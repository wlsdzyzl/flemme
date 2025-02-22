
from .common import *
from mamba_ssm import Mamba, Mamba2
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from einops import rearrange, repeat

def zsort(xyz):
    z = xyz[:, :, 2]
    return torch.argsort(z, dim = -1)
def ysort(xyz):
    y = xyz[:, :, 1]
    return torch.argsort(y, dim = -1)
def xsort(xyz):
    x = xyz[:, :, 0]
    return torch.argsort(x, dim = -1)
def centersort(xyz):
    m = torch.mean(xyz, dim = 1, keepdim = True)
    dist_to_m = torch.norm(xyz - m, dim = -1)
    return torch.argsort(dist_to_m, dim = -1)
def nonsort(xyz):
    return torch.arange(xyz.shape[1]).unsqueeze(0).expand(xyz.shape[0], -1)
### to recover the original tensor
def resort(sorted_index):
    return torch.argsort(sorted_index, dim = -1)

def get_scanners(scanners):
    res = []
    if scanners is None:
      return res
    if type(scanners) == str:
      scanners = [scanners,]
    assert type(scanners) == list or type(scanners) == tuple, \
        'scan strategies should be a str, list or tuple.'
    for scanner in scanners:
        if scanner == 'z_order':
            res.append(zsort)
        elif scanner == 'y_order':
            res.append(ysort)
        elif scanner == 'x_order':
            res.append(xsort)
        elif scanner == 'center_dist':
            res.append(centersort)
        elif scanner == 'nonsort':
            res.append(nonsort)
        else:
            logger.error(f'Unsupported scan strategy for point cloud: {scanner}.')
            exit(1)
    return res

def get_psmamba_block(name, **kwargs):
    logger.debug('building block parameters: {}'.format(kwargs))
    if name == 'pmamba':
        return partial(PointScanMambaBlock, **kwargs)
    elif name == 'pmamba2':
        return partial(PointScanMamba2Block, **kwargs)

class PointMambaBlock(NormBlock):
    def __init__(self, in_channel, out_channel = None, 
        time_channel = 0, state_channel = 64, 
        conv_kernel_size = 4, inner_factor = 2,  
        head_channel = 64,
        conv_bias=True, bias=False,
        chunk_size=256,
        dt_min=0.001, A_init_range=(1, 16),
        dt_max=0.1, dt_init_floor=1e-4, 
        dt_rank = None, dt_scale = 1.0,
        dropout = None, skip_connection = True, mamba = 'Mamba', 
        activation = 'relu', norm='batch', num_norm_groups = -1, 
        post_normalization = False, **kwargs):
        
        super().__init__(_channel_dim = -1)
        self.in_channel = in_channel
        self.out_channel = out_channel or in_channel
        inner_factor = int(inner_factor)
        if mamba == 'Mamba':
            dt_rank = dt_rank or math.ceil(in_channel / 16)
            self.mamba = Mamba(d_model = in_channel, d_state = state_channel, 
                        d_conv = conv_kernel_size, expand = inner_factor, 
                        bias = bias, conv_bias = conv_bias,                         
                        dt_max = dt_max, dt_min = dt_min, dt_rank = dt_rank,
                        dt_init_floor = dt_init_floor, dt_scale = dt_scale)
        elif mamba == 'Mamba2':
            # assert (in_channel * inner_factor / head_channel) % 8 == 0, 'Mamba2 problem: Make sure that (in_channel * inner_factor / head_channel) % 8 == 0'
            self.mamba = Mamba2(d_model = in_channel, d_state = state_channel, 
                        d_conv = conv_kernel_size, expand = inner_factor, 
                        headdim = head_channel,
                        bias = bias, conv_bias = conv_bias, chunk_size = chunk_size,
                        use_mem_eff_path = False,
                        dt_max = dt_max, dt_min = dt_min, 
                        dt_init_floor = dt_init_floor,
                        A_init_range = A_init_range)
        self.post_normalization = post_normalization
        if post_normalization:
            self.norm, self.norm_type = get_norm(norm, out_channel, 1, num_norm_groups)
        else:
            self.norm, self.norm_type = get_norm(norm, in_channel, 1, num_norm_groups)
        self.act = get_act(activation)
        self.time_channel = time_channel
        self.skip_connection = skip_connection
        self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p=dropout)
        if self.time_channel > 0:
            self.hyper_bias = nn.Linear(self.time_channel, out_channel, bias=False)
            self.hyper_gate = nn.Linear(self.time_channel, out_channel)

    def forward(self, x, t = None):
        res = self.mamba(x)
        res = self.dropout(res)

        if not self.post_normalization:
            res = self.act(self.normalize(res))
        
        if self.skip_connection:
            res = x + res
        x = self.dense(res)

        if self.post_normalization:
            x = self.act(self.normalize(x))
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x
    @staticmethod
    def is_sequence_modeling():
        return True

### contains different scan strategies
class PointScanMambaBaseBlock(NormBlock):
    def __init__(
        self, in_channel, num_scan, 
        out_channel, state_channel, 
        conv_kernel_size, inner_factor, 
        dropout, conv_bias, bias, activation,
        norm, num_norm_groups, 
        skip_connection,
        post_normalization):
        super().__init__(_channel_dim = -1)
        self.in_channel = in_channel
        ### out channel should be equal to in channel
        self.out_channel = out_channel or in_channel
        self.state_channel = state_channel or math.ceil(self.in_channel / 6)
        self.conv_kernel_size = conv_kernel_size
        self.inner_channel = int(inner_factor * self.in_channel) 
        self.in_proj = nn.Linear(self.in_channel, self.inner_channel * 2, bias=bias)

        self.conv = nn.Conv1d(
            self.inner_channel,
            self.inner_channel,
            groups=self.inner_channel,
            bias=conv_bias,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size - 1,
        )
        self.K = num_scan
        self.act = get_act(activation)
        self.skip_connection = skip_connection
        self.post_normalization = post_normalization
        if post_normalization:
            self.norm, self.norm_type = get_norm(norm, out_channel, 1, num_norm_groups)
        else:
            self.norm, self.norm_type = get_norm(norm, in_channel, 1, num_norm_groups)

        self.out_proj = nn.Linear(self.inner_channel, self.in_channel, bias=bias)

        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p=dropout)

        ### to make sure the output is equal to input
        #### run reconstruction needed.
        self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()

    def scan(self, xyz_features, sorted_index_list):
        
        assert len(sorted_index_list) == self.K, \
          f'Mis-matched scan number and scan results: {len(sorted_index_list)} and {self.K}.'
        c = xyz_features.shape[1]
        sorted_feature_list = []
        # B, C, L
        for idx in sorted_index_list:
            index = idx.unsqueeze(1).expand(-1, c, -1)
            sorted_feature = torch.gather(xyz_features, dim = -1, index = index)
            sorted_feature_list.append(sorted_feature)

        # B C L -> B K C L
        new_feature = torch.stack(sorted_feature_list, dim = 1)
        return new_feature
    
    def scan_proj(self, new_feature, sorted_index_list):
        # B K C L 
        res = []
        c = new_feature.shape[2]
        for idx, sorted_idx in enumerate(sorted_index_list):
            rsidx = resort(sorted_idx)
            index = rsidx.unsqueeze(1).expand(-1, c, -1)
            recovered_feature = torch.gather( new_feature[:, idx, ...], dim = -1,  index = index)
            res.append(recovered_feature)
        y = sum(res)
        return y
    def ssm(self, xs):
        raise NotImplementedError
    
    def forward(self, x, sorted_index_list):
        ### scan mamba block
        # print(x.shape)
        shortcut = x
        L = x.shape[1]
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) 
        x = channel_recover(x)
        x = self.act(self.conv(x))[... , :L]
        xs = self.scan(x, sorted_index_list)
        out_y = self.ssm(xs)
        y = self.scan_proj(out_y, sorted_index_list)
        y = channel_transfer(y)
        y = y * self.act(z)
        out = self.out_proj(y)
        out = self.dropout(out)
        
        if not self.post_normalization:
            out = self.act(self.normalize(out))

        if self.skip_connection:
            out = shortcut + out
        out = self.dense(out)

        if self.post_normalization:
            out = self.act(self.normalize(out))
        return out

class PointScanMambaBlock(PointScanMambaBaseBlock):
    def __init__(
        self, in_channel, num_scan, out_channel = None, time_channel = 0, 
        state_channel=None, conv_kernel_size=4,
        inner_factor = 2.0,  dt_rank=None, dt_min=0.001, 
        dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, 
        dropout=0., conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_norm_groups = 0, 
        skip_connection = True,
        post_normalization = False, **kwargs):
        super().__init__(in_channel = in_channel, 
            num_scan = num_scan,
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            dropout=dropout, 
            conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_norm_groups = num_norm_groups, 
            skip_connection = skip_connection,
            post_normalization = post_normalization)
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.dt_rank = dt_rank or math.ceil(self.in_channel / 16)        
        self.x_proj = tuple(
            nn.Linear(self.inner_channel, (self.dt_rank + self.state_channel * 2), bias=False) for _ in range(self.K))
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj
        self.dt_projs = tuple(
            self.dt_init(self.dt_rank, self.inner_channel, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(self.K))
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.state_channel, self.inner_channel, copies=self.K, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.inner_channel, copies=self.K, merge=True) # (K=4, D, N)
        self.selective_scan = selective_scan_fn
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.hyper_bias = nn.Linear(time_channel, out_channel, bias=False)
            self.hyper_gate = nn.Linear(time_channel, out_channel)
    @staticmethod
    def dt_init(dt_rank, inner_channel, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, inner_channel, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(inner_channel) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(state_channel, inner_channel, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, state_channel + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=inner_channel,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(inner_channel, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(inner_channel, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    def ssm(self, xs):
        B, K, _, L = xs.shape
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        ### split on channel
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.state_channel, self.state_channel], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * c, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, state_channel, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, state_channel, l)
        
        Ds = self.Ds.float().view(-1) # (k * c)
        As = -torch.exp(self.A_logs.float()).view(-1, self.state_channel)  # (k, inner_channel, state_channel)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * c)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        return out_y
    def forward(self, x, it):
        sorted_index_list, t = it
        x = super().forward(x, sorted_index_list)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x

class PointScanMamba2Block(PointScanMambaBaseBlock):
    def __init__(
        self, in_channel, num_scan, out_channel = None, 
        time_channel = 0, state_channel=None, conv_kernel_size=4,
        inner_factor = 2.0,  
        head_channel = 64, 
        learnable_init_states = True, chunk_size=256,
        dt_min=0.001, A_init_range=(1, 16),
        dt_max=0.1, dt_init_floor=1e-4, 
        dropout=0., conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_norm_groups = 0, 
        skip_connection = True,
        post_normalization = False, **kwargs):

        super().__init__(in_channel = in_channel, 
            num_scan = num_scan,
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            dropout=dropout, 
            conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_norm_groups = num_norm_groups, 
            skip_connection = skip_connection,
            post_normalization = post_normalization)
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.head_channel = head_channel
        assert self.inner_channel % self.head_channel == 0, \
            'inner channel should be divisible by head_channel'
        
        self.num_heads_each_K = self.inner_channel // self.head_channel
        self.num_heads = self.K * self.num_heads_each_K
        self.init_states = None
        if learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.num_heads, self.head_channel, self.state_channel))
            self.init_states._no_weight_decay = True
            # initialization

        self.x_proj = tuple(
            nn.Linear(self.inner_channel, self.state_channel * 2 + self.num_heads_each_K, bias=False) for _ in range(self.K))
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        
        del self.x_proj
        
        self.A_logs = self.A_log_init(self.num_heads, A_init_range)
        self.Ds = self.D_init(self.num_heads)
        self.dt_bias = self.dt_bias_init(self.num_heads, dt_min=dt_min, 
                                        dt_max=dt_max, dt_init_floor=dt_init_floor)

        self.selective_scan = mamba_chunk_scan_combined
        self.chunk_size = chunk_size
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.hyper_bias = nn.Linear(time_channel, out_channel, bias=False)
            self.hyper_gate = nn.Linear(time_channel, out_channel)
    @staticmethod
    def dt_bias_init(num_heads, dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        dt_bias._no_weight_decay = True
        return dt_bias
    @staticmethod
    def A_log_init(num_heads, A_init_range):
        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(num_heads).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=A.dtype)
        A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.num_heads, dtype=torch.float32, device=device), persistent=True)
        A_log._no_weight_decay = True
        return A_log
    @staticmethod
    def D_init(num_heads):
        # D "skip" parameter
        D = nn.Parameter(torch.ones(num_heads))
        D._no_weight_decay = True
        return D
    def ssm(self, xs):
        B, K, _, L = xs.shape
        
        BCdt = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # b l k d
        xs = rearrange(xs, "b k c l -> b l k c")
        # b l k c (c = num_head + state + state)
        BCdt = rearrange(BCdt, "b k c l -> b l k c")
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        ### split on channel
        As = -torch.exp(self.A_logs)  # (num_heads) or (inner_channel, state_channel)
        initial_states=repeat(self.init_states, "... -> b ...", b=B) \
                            if self.init_states is not None else None
        
        Bs, Cs, dt = torch.split(BCdt, [self.state_channel, self.state_channel, self.num_heads_each_K], dim=-1)
        xs = xs.float().view(B, L, self.num_heads, -1) # (b, l, h p)
        Bs = Bs.float().view(B, L, K, -1) 
        Cs = Cs.float().view(B, L, K, -1) # (b, l, k, state_channel)
        dt = dt.float().view(B, L, -1)
        dt = F.softplus(dt + self.dt_bias)  # (B, L, num_heads)
        out_y = self.selective_scan(
                xs,
                dt,
                As,
                Bs,
                Cs, 
                chunk_size=self.chunk_size,
                D=self.Ds,
                z=None,
                seq_idx=None,
                initial_states=initial_states
            )
        out_y = channel_recover(rearrange(out_y, "b l h p -> b l (h p)")).view(B, K, -1, L)
        return out_y
    def forward(self, x, it):
        sorted_index_list, t = it
        x = super().forward(x, sorted_index_list)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x
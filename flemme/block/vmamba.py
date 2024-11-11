from .common import *
from flemme.logger import get_logger
### blocks about mambaimport time
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat

logger = get_logger('mamba_block')
### permute and reconver table for 3D image
### the first column is all possible permuted orders
### the second colume is the permute order to recover permuted tensor back to original shape 
cross_per_rec_table = {
        ### 2D
        2:
        [
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]]
         ],
        ### 3D
        3:
        [
            [[0, 1, 2], [0, 1, 2]],
            [[0, 2, 1], [0, 2, 1]],
            [[1, 0, 2], [1, 0, 2]],
            [[1, 2, 0], [2, 0, 1]],
            [[2, 0, 1], [1, 2, 0]],
            [[2, 1, 0], [2, 1, 0]]
        ]
        }
simplifed_per_rec_table = {
        ### 2D
        2:
        [
            [[0, 1], [0, 1]],
            [[1, 0], [1, 0]]
         ],
        ### 3D
        3:
        [
            [[0, 1, 2], [0, 1, 2]],
            [[2, 1, 0], [2, 1, 0]]
        ]
        }
single_per_rec_table = {
        ### 2D
        2:
        [
            [[0, 1], [0, 1]]
         ],
        ### 3D
        3:
        [
            [[0, 1, 2], [0, 1, 2]]
        ]
        }
class MBaseBlock(nn.Module):
    def __init__(
        self, dim, in_channel, out_channel = None, state_channel=None, 
        conv_kernel_size=3, inner_factor = 2.0,  mlp_hidden_ratio=[4.0,], 
        dropout=0., drop_path=0.0, conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_groups = 0, scan_mode = 'single', flip_scan = False):
        super().__init__()
        self.in_channel = in_channel
        ### out channel should be equal to in channel
        self.out_channel = out_channel or in_channel
        self.state_channel = state_channel or math.ceil(self.in_channel / 6)
        self.conv_kernel_size = conv_kernel_size
        self.inner_channel = int(inner_factor * self.in_channel) 
        self.in_proj = nn.Linear(self.in_channel, self.inner_channel * 2, bias=bias)
        self.dim = dim
        self.K = 1
        self.scan_mode = scan_mode
        assert scan_mode in ['cross', 'single', 'simplified'], 'Unsupported scan model for vision mamba.'
        
        if self.dim == 2:
            self.conv = nn.Conv2d(
                self.inner_channel,
                self.inner_channel,
                groups=self.inner_channel,
                bias=conv_bias,
                kernel_size=conv_kernel_size,
                padding=(conv_kernel_size - 1) // 2
            )
            if not self.scan_mode == 'single':
                self.K = 2

        if self.dim == 3:
            self.conv = nn.Conv3d(
                self.inner_channel,
                self.inner_channel,
                groups=self.inner_channel,
                bias=conv_bias,
                kernel_size=conv_kernel_size,
                padding=(conv_kernel_size - 1) // 2
            )         
            if self.scan_mode == 'cross':
                self.K = 6   
            elif self.scan_mode == 'simplified':
                self.K = 2
        self.flip_scan = flip_scan
        if self.flip_scan:
            self.K *= 2
        self.act = get_act(activation)

        self.norm1 = NormBlock(*(get_norm(norm, in_channel, self.dim, num_groups) + (-1,)))
        self.norm2 = NormBlock(*(get_norm(norm, self.inner_channel, self.dim, num_groups) + (-1,)))
        self.out_proj = nn.Linear(self.inner_channel, self.in_channel, bias=bias)

        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p=dropout)
        self.drop_path = DropPath(
                    drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_hidden_ratio = mlp_hidden_ratio
        mlp_hidden_channels = [int(in_channel * r) for r in mlp_hidden_ratio]
        self.mlp = MultiLayerPerceptionBlock(in_channel=in_channel, out_channel=self.out_channel, 
                        hidden_channels=mlp_hidden_channels,
                        activation=activation, dropout=dropout)

        ### to make sure the output is equal to input
        #### run reconstruction needed.
        self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()
        # self.dense = DenseBlock(in_channel = in_channel, out_channel = self.out_channel, 
        #             activation = None) if in_channel != self.out_channel else nn.Identity()
        self.per_rec_table = single_per_rec_table
        if self.scan_mode == 'simplified':
            self.per_rec_table = simplifed_per_rec_table
        elif self.scan_mode == 'cross':
            self.per_rec_table = cross_per_rec_table
    def cross_scan(self, x):
        ### for 2D image, D = -1
        D = -1
        input_shape = x.shape
        x = channel_recover(x)
        x = self.act(self.conv(x)) # (b, d, h, w)

        if self.dim == 2:
            B, _, H, W = x.shape
            L = H * W
            if self.scan_mode == 'single':
                x_cross = x.view(B, 1, -1, L)
            else:
                ### cross scan module
                ### can be extend to 3D: the most simple way is to use reverse scan
                x_cross = torch.stack([x.view(B, -1, L), 
                                    torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        else:
            B, _, H, W, D = x.shape
            L = H * W * D
            if self.scan_mode == 'single':
                x_cross = x.view(B, 1, -1, L)
            elif self.scan_mode == 'simplified':
                x_cross = torch.stack([x.view(B, -1, L), 
                                        x.permute(0, 1, 4, 3, 2).contiguous().view(B, -1, L)
                                        ], dim=1).view(B, 2, -1, L)
            else:
                x_cross = torch.stack([x.view(B, -1, L), 
                                        x.permute(0, 1, 2, 4, 3).contiguous().view(B, -1, L),
                                        x.permute(0, 1, 3, 2, 4).contiguous().view(B, -1, L),
                                        x.permute(0, 1, 3, 4, 2).contiguous().view(B, -1, L),
                                        x.permute(0, 1, 4, 2, 3).contiguous().view(B, -1, L),
                                        x.permute(0, 1, 4, 3, 2).contiguous().view(B, -1, L)
                                        ], dim=1).view(B, 6, -1, L)
        xs = x_cross
        if self.flip_scan:
            xs = torch.cat([x_cross, torch.flip(x_cross, dims=[-1])], dim=1) # (b, k, c, l)
        return xs, input_shape
    
    def scan_proj(self, out_y, input_shape):
        ## input_shape: B, H, W, C
        B, K, _, L = out_y.shape
        if self.flip_scan:
            inv_y = torch.flip(out_y[:, K // 2: K], dims=[-1]).view(B, K // 2, -1, L)
        ## mapping dim index to shape
        res = []
        for ik in range(self.K // (2**self.flip_scan)):
            per_shape = [B, -1] + [ input_shape[d+1] for d in self.per_rec_table[self.dim][ik][0]]
            rec_shape = [0, 1] + [d + 2 for d in self.per_rec_table[self.dim][ik][1]]
            res.append(out_y[:, ik].view(*per_shape).permute(*rec_shape).contiguous().view(B, -1, L))
            if self.flip_scan:
                res.append(inv_y[:, ik].view(*per_shape).permute(*rec_shape).contiguous().view(B, -1, L))
        y = torch.transpose(sum(res), dim0=1, dim1=2).contiguous().view(input_shape)
        y = self.norm2.normalize(y)
        return y
    def ssm(self, xs):
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm1.normalize(x)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)
        xs, shape = self.cross_scan(x)
        out_y = self.ssm(xs)
        y = self.scan_proj(out_y, shape)
        y = y * self.act(z)
        out = self.out_proj(y)
        out = self.dropout(out)
        ### keep same with swin transformer block
        out = shortcut + self.drop_path(out)
        out = self.dense(out) + self.drop_path(self.mlp(out))
        return out
    
class VMambaBlock(MBaseBlock):
    def __init__(
        self, dim, in_channel, out_channel = None, time_channel = 0, 
        state_channel=None, conv_kernel_size=3,
        inner_factor = 2.0,  mlp_hidden_ratio=[4.0,], dt_rank=None, dt_min=0.001, 
        dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, 
        dropout=0., drop_path=0.0, conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_groups = 0, scan_mode = 'single', flip_scan = False, **kwargs):
        super().__init__(dim = dim, in_channel = in_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio,
            dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
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

        xs = xs.float().view(B, -1, L) # (b, k * c, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * c, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, state_channel, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, state_channel, l)
        
        Ds = self.Ds.float().view(-1) # (k * c)
        As = -torch.exp(self.A_logs.float()).view(-1, self.state_channel)  # (k * c, state_channel)
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
    def forward(self, x, t = None):
        x = super().forward(x)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x
    
class DoubleVMambaBlock(nn.Module):
    def __init__(
        self, dim, in_channel, 
        out_channel = None, time_channel = 0, 
        state_channel=None, conv_kernel_size=3,
        inner_factor = 2.0,  mlp_hidden_ratio=[4.0,], dt_rank=None, dt_min=0.001, 
        dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, 
        dropout=0., drop_path=0.0, conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_groups = 0, scan_mode = 'single', flip_scan = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.mamba1 = VMambaBlock(dim = dim, in_channel = in_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio, dt_rank=dt_rank, dt_min=dt_min, 
            dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, 
            dt_init_floor=dt_init_floor, dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
        self.mamba2 = VMambaBlock(dim = dim, in_channel = out_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio, dt_rank=dt_rank, dt_min=dt_min, 
            dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, 
            dt_init_floor=dt_init_floor, dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.time_emb = nn.Linear(time_channel, out_channel)

    def forward(self, x, t = None):
        x = self.mamba1(x)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            x = x + expand_as(self.time_emb(t), x, channel_dim=-1)
        x = self.mamba2(x)
        return x
    
class ResVMambaBlock(nn.Module):
    def __init__(
        self, dim, in_channel, out_channel = None, time_channel = 0, 
        state_channel=None, conv_kernel_size=3,
        inner_factor = 2.0,  mlp_hidden_ratio=[4.0,], dt_rank=None, dt_min=0.001, 
        dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, 
        dropout=0., drop_path=0.0, conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_groups = 0, scan_mode = 'single', flip_scan = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.mamba1 = VMambaBlock(dim = dim, in_channel = in_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio, dt_rank=dt_rank, dt_min=dt_min, 
            dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, 
            dt_init_floor=dt_init_floor, dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
        self.mamba2 = VMambaBlock(dim = dim, in_channel = out_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio, dt_rank=dt_rank, dt_min=dt_min, 
            dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, 
            dt_init_floor=dt_init_floor, dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = None, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
        self.act = get_act(activation)  
        self.shortcut = nn.Linear(in_channel, out_channel) if not in_channel == out_channel else nn.Identity()
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.time_emb = nn.Linear(time_channel, out_channel)
    def forward(self, x, t = None):
        h = self.mamba1(x)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            h = h + expand_as(self.time_emb(t), h, channel_dim=-1)
        h = self.mamba2(h)
        out = self.act(h + self.shortcut(x))
        return out
    
class VMamba2Block(MBaseBlock):
    def __init__(
        self, dim, in_channel, out_channel = None, 
        time_channel = 0, state_channel=None, conv_kernel_size=3,
        inner_factor = 2.0,  mlp_hidden_ratio=[4.0,], 
        head_channel = 64, 
        learnable_init_states = True, chunk_size=256,
        dt_min=0.001, A_init_range=(1, 16),
        dt_max=0.1, dt_init_floor=1e-4, 
        dropout=0., drop_path=0.0, conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_groups = 0, scan_mode = 'single', flip_scan = False, **kwargs):

        super().__init__(dim = dim, in_channel = in_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio,
            dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
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
        # b l g d
        xs = rearrange(xs, "b k c l -> b l k c")
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
    def forward(self, x, t = None):
        x = super().forward(x)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x
class DoubleVMamba2Block(nn.Module):
    def __init__(
        self, dim, in_channel, out_channel = None, time_channel = 0, 
        state_channel=None, conv_kernel_size=3,
        inner_factor = 2.0,  mlp_hidden_ratio=[4.0,], 
        head_channel = 128, 
        learnable_init_states = True, chunk_size=256,
        dt_min=0.001, A_init_range=(1, 16),
        dt_max=0.1, dt_init="random", dt_init_floor=1e-4, 
        dropout=0., drop_path=0.0, conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_groups = 0, scan_mode = 'single', flip_scan = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.mamba1 = VMamba2Block(dim = dim, in_channel = in_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio, head_channel = head_channel,
            learnable_init_states = learnable_init_states, chunk_size = chunk_size,
            A_init_range=A_init_range,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, 
            dt_init_floor=dt_init_floor, dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
        self.mamba2 = VMamba2Block(dim = dim, in_channel = out_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio, head_channel = head_channel,
            learnable_init_states = learnable_init_states, chunk_size = chunk_size,
            A_init_range=A_init_range,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, 
            dt_init_floor=dt_init_floor, dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.time_emb = nn.Linear(time_channel, out_channel)
    def forward(self, x, t = None):
        x = self.mamba1(x)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            x = x + expand_as(self.time_emb(t), x, channel_dim=-1)
        x = self.mamba2(x)
        return x
    
class ResVMamba2Block(nn.Module):
    def __init__(
        self, dim, in_channel, out_channel = None, time_channel = 0, 
        state_channel=None, conv_kernel_size=3,
        inner_factor = 2.0,  mlp_hidden_ratio=[4.0,], 
        head_channel = 128, 
        learnable_init_states = True, chunk_size=256,
        dt_min=0.001, A_init_range=(1, 16),
        dt_max=0.1, dt_init="random", dt_init_floor=1e-4, 
        dropout=0., drop_path=0.0, conv_bias=True, bias=False, activation = 'silu',
        norm = None, num_groups = 0, scan_mode = 'single', flip_scan = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.mamba1 = VMamba2Block(dim = dim, in_channel = in_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio, head_channel = head_channel,
            learnable_init_states = learnable_init_states, chunk_size = chunk_size,
            A_init_range=A_init_range,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, 
            dt_init_floor=dt_init_floor, dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = activation, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
        self.mamba2 = VMamba2Block(dim = dim, in_channel = out_channel, 
            out_channel = out_channel, state_channel=state_channel, 
            conv_kernel_size=conv_kernel_size, inner_factor = inner_factor,  
            mlp_hidden_ratio=mlp_hidden_ratio, head_channel = head_channel,
            learnable_init_states = learnable_init_states, chunk_size = chunk_size,
            A_init_range=A_init_range,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, 
            dt_init_floor=dt_init_floor, dropout=dropout, 
            drop_path=drop_path, conv_bias=conv_bias, bias=bias, 
            activation = None, norm = norm, 
            num_groups = num_groups, scan_mode = scan_mode, flip_scan = flip_scan)
        self.act = get_act(activation)  
        self.shortcut = nn.Linear(in_channel, out_channel) if not in_channel == out_channel else nn.Identity()
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.time_emb = nn.Linear(time_channel, out_channel)
    def forward(self, x, t = None):
        h = self.mamba1(x)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.' 
            h = h + expand_as(self.time_emb(t), h, channel_dim=-1)
        h = self.mamba2(h)
        out = self.act(h + self.shortcut(x))
        return out
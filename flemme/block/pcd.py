from .common import *
from mamba_ssm import Mamba, Mamba2

## insight from PCT: https://arxiv.org/pdf/2012.09688
class OffSetAttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, in_channel, num_heads = 3, d_k = None, 
        qkv_bias = True, qk_scale = None, atten_dropout = None, 
        dropout = None, skip_connection = False):
        """
        * `in_channel` is the number of channel in the input
        * `num_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        self.d_k = d_k or int(in_channel / num_heads)

        if atten_dropout is None or atten_dropout <= 0:
            self.atten_dropout = nn.Identity()
        else:
            self.atten_dropout = nn.Dropout(p=atten_dropout)

        # Project x to query, key and values
        self.qkv = nn.Linear(in_channel, num_heads * self.d_k * 3, bias = qkv_bias)
        # project to original space
        self.proj = nn.Linear(num_heads * self.d_k, in_channel)
        # offset transformation
        self.offset_trans = nn.Linear(in_channel, in_channel)
        self.num_heads = num_heads
        self.skip_connection = skip_connection
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p=dropout)
    
    def attention(self, q, k, v):
        # [batch_size, num_heads, seq, d_k]
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        atten = torch.einsum('bhik,bhjk->bhij', q, k)
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        atten = atten.softmax(dim=-1)
        atten = atten / (1e-8 + atten.sum(dim=1, keepdims = True))
        atten = self.atten_dropout(atten)
        res = torch.einsum('bhik,bhkj->bhij', atten, v)
        return res
    
    def forward(self, x: torch.Tensor):
        # Get shape
        
        batch_size, in_channel = x.shape[0], x.shape[-1]
        # Change `x` to shape `[batch_size, seq, in_channel]`
        x = x.reshape(batch_size, -1, in_channel)
        # Get query, key, and values (concatenated) and shape it to `[3, batch_size, num_heads, seq, d_k]`
        qkv = self.qkv(x).reshape(batch_size, -1, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # Split query, key, and values.
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        res = self.attention(q, k, v)

        # Reshape to `[batch_size, seq, num_heads * d_k]`
        res = res.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)
        # Transform to `[batch_size, seq, in_channel]`
        res = self.proj(res)
        res = self.offset_trans(x - res)
        res = self.dropout(res)
        # Add skip connection
        if self.skip_connection:
            res = res + x
        return res

class PointTransformerBlock(NormBlock):
    def __init__(self, in_channel, out_channel = None, 
        time_channel = 0, num_heads = 3, d_k = None, 
        qkv_bias = True, qk_scale = None, atten_dropout = None, 
        dropout = None, residual_attention = False, 
        skip_connection = True, attention = 'SA', 
        activation = 'relu', norm='batch', num_groups = -1, **kwargs):
        
        super().__init__(_channel_dim = -1)
        self.in_channel = in_channel
        self.out_channel = out_channel or in_channel
        if attention == 'SA':
            self.atten = SelfAttentionBlock(in_channel=in_channel, num_heads=num_heads, d_k = d_k, 
            qkv_bias = qkv_bias, qk_scale = qk_scale, atten_dropout = atten_dropout, 
            dropout = dropout, skip_connection = residual_attention, channel_dim = -1)
        elif attention == 'OA':
            self.atten = OffSetAttentionBlock(in_channel=in_channel, num_heads=num_heads, d_k = d_k, 
            qkv_bias = qkv_bias, qk_scale = qk_scale, atten_dropout = atten_dropout, 
            dropout = dropout, skip_connection = residual_attention)

        self.norm, self.norm_type = get_norm(norm, in_channel, 1, num_groups)
        self.act = get_act(activation)
        self.time_channel = time_channel
        self.skip_connection = skip_connection
        self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()

        if self.time_channel > 0:
            self.hyper_bias = nn.Linear(self.time_channel, out_channel, bias=False)
            self.hyper_gate = nn.Linear(self.time_channel, out_channel)
    def forward(self, x, t = None):
        res = self.atten(x)
        res = self.act(self.normalize(res))
        if self.skip_connection:
            res = x + res
        x = self.dense(res)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x

class PointMambaBlock(NormBlock):
    def __init__(self, in_channel, out_channel = None, 
        time_channel = 0, state_channel = 64, 
        conv_kernel_size = 4, inner_factor = 2.0,  
        head_channel = 64,
        conv_bias=True, bias=False,
        learnable_init_states = True, chunk_size=256,
        dt_min=0.001, A_init_range=(1, 16),
        dt_max=0.1, dt_init_floor=1e-4, 
        dt_rank = None, dt_scale = 1.0,
        dropout = None, skip_connection = True, mamba = 'Mamba', 
        activation = 'relu', norm='batch', num_groups = -1, **kwargs):
        
        super().__init__(_channel_dim = -1)
        self.in_channel = in_channel
        self.out_channel = out_channel or in_channel
        if mamba == 'Mamba':
            dt_rank = dt_rank or math.ceil(in_channel / 16)
            self.mamba = Mamba(d_model = in_channel, d_state = state_channel, 
                        d_conv = conv_kernel_size, expand = inner_factor, 
                        bias = bias, conv_bias = conv_bias,                         
                        dt_max = dt_max, dt_min = dt_min, dt_rank = dt_rank,
                        dt_init_floor = dt_init_floor, dt_scale = dt_scale)
        elif mamba == 'Mamba2':
            self.mamba = Mamba2(d_model = in_channel, d_state = state_channel, 
                        d_conv = conv_kernel_size, expand = inner_factor, 
                        headdim = head_channel,
                        bias = bias, conv_bias = conv_bias, chunk_size = chunk_size,
                        learnable_init_states = learnable_init_states, 
                        dt_max = dt_max, dt_min = dt_min, 
                        dt_init_floor = dt_init_floor,
                        A_init_range = A_init_range)

        self.norm, self.norm_type = get_norm(norm, in_channel, 1, num_groups)
        self.act = get_act(activation)
        self.time_channel = time_channel
        self.skip_connection = skip_connection
        self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()

        if self.time_channel > 0:
            self.hyper_bias = nn.Linear(self.time_channel, out_channel, bias=False)
            self.hyper_gate = nn.Linear(self.time_channel, out_channel)

    def forward(self, x, t = None):
        res = self.mamba(x)
        res = self.act(self.normalize(res))
        if self.skip_connection:
            res = x + res
        x = self.dense(res)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x

## part of this code is adopted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def knn(x, k):
    # x: B*N*D, inner: B*N*N
    inner = 2*torch.matmul(x, x.transpose(1, 2))
    # x^2: B*N*1
    xx = torch.sum(x**2, dim=-1, keepdim=True)
    # negative distance
    pairwise_distance = -(xx.transpose(1, 2) - inner + xx)
    # return the closest k indices, idx: B*N*K
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx

def get_graph_feature(x, k=20, idx=None):
    # B * N * D
    batch_size, num_points, num_dims = x.shape
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)
    # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    # feature: B*N*K*D 
    feature = feature.view(batch_size, num_points, k, num_dims) 
    # x: B*N*K*D
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # feature: B*N*K*2D
    feature = torch.cat((feature-x, x), dim=3)
    return feature


class LocalGraphLayer(nn.Module):
    def __init__(self, k, 
                in_channel, 
                out_channel, 
                BuildingBlock, 
                is_seq = True,
                **kwargs):
        super().__init__()
        self.is_seq = is_seq
        self.k = k
        if not self.is_seq:
            self.bb = BuildingBlock(in_channel = 2*in_channel,
                                    out_channel = out_channel) 
        else:
            self.bb = BuildingBlock(in_channel = 2*in_channel * k,
                                    out_channel = out_channel * k) 
    def forward(self, x, t=None):
        ## x: B*N*D -> B * N * K * 2D
        x = get_graph_feature(x, k = self.k)
        if not self.is_seq:
            x = self.bb(x, t)
        ## sequence modeling
        else:
            B, N, K, _ = x.shape
            # B * N * K * 2D -> B * N * (K*2D)
            x = x.reshape(B, N, -1)
            x = self.bb(x, t)
            # B * N * (K * 2D) -> B * N * K * D_out
            x = x.reshape(B, N, K, -1)
        ## x: B * N * D
        x, _ = x.max(dim=2)
        return x
        

class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel, 
                out_channel, 
                hidden_channels, 
                BuildingBlock,
                **kwargs):
                
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        module_sequence = []
        for hc in hidden_channels:
            module_sequence.append(BuildingBlock(in_channel = in_channel, 
                    out_channel = hc))         
            in_channel = hc
        # module_sequence.append(DenseBlock(in_channel = in_channel, 
        #         out_channel = out_channel, activation = None))
        self.layers = SequentialT(*module_sequence)
        self.final = nn.Linear(in_channel, out_channel)
    ## codewords
    def forward(self, shapes, codewords, t = None):
        """
        Parameters
        ----------
            codewords = B * N * D
            shapes: shapes or intermediate reconstructed point clouds
        """
        assert shapes.shape[-1] + codewords.shape[-1] == self.in_channel,\
            f"channel of grid + channel of codewords should be equal to the channel of input, we get: {shapes.shape[-1]} {codewords.shape[-1]} and {self.in_channel}."
        # concatenate
        x = torch.cat([shapes, codewords], dim=-1)
        x, _ = self.layers(x, t)
        x = self.final(x)
        return x

# class FoldingLayer(nn.Module):
#     """
#     The folding operation of FoldingNet
#     """

#     def __init__(self, in_channel, 
#                     out_channel, 
#                 hidden_channels,
#                 **kwargs):
#         super(FoldingLayer, self).__init__()

#         layers = []
#         for oc in hidden_channels:
#             # # conv = nn.Conv1d(in_channel, oc,1)
#             # conv = nn.Linear(in_channel, oc)
#             # # bn = NormBlock(norm = nn.BatchNorm1d(oc), norm_type = Norm.BATCH, _channel_dim = -1)
#             # bn = NormBlock(norm = nn.GroupNorm(1, oc), norm_type = Norm.GROUP, _channel_dim = -1)
#             # active = nn.LeakyReLU()
#             # layers.extend([conv, bn, active])
            
#             layers.append(DenseBlock(in_channel, oc, norm='batch', activation='relu'))
#             in_channel = oc
#         out_layer = nn.Linear(in_channel, out_channel)
#         layers.append(out_layer)
        
#         self.layers = nn.Sequential(*layers)

#     def forward(self, shapes, codewords, t = None):
#         """
#         Parameters
#         ----------
#             shapes: reshaped 2D shapes or intermediam reconstructed point clouds
#         """
#         # concatenate
#         # print(shapes.shape, codewords.shape)
#         x = torch.cat([shapes, codewords], dim=-1)
#         # shared mlp
#         x = self.layers(x)
#         return x
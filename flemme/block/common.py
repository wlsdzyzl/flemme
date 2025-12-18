# some building blocks
import torch
from torch.nn import functional as F, init
from torch import nn
from torch import fft
import math
from enum import Enum, auto
from functools import partial
from flemme.logger import get_logger

logger = get_logger('block')
class Norm(Enum):
    BATCH = auto()
    GROUP = auto()
    LAYER = auto()
    INSTANCE = auto()
    NONE = auto()
### transfer channel to the last dim
def channel_transfer(x):
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.transpose(1, 2).contiguous()
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1).contiguous()
    if x.ndim == 5:
        return x.permute(0, 2, 3, 4, 1).contiguous()
### transfer channel to the second dim
def channel_recover(x):
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.transpose(1, 2).contiguous()
    if x.ndim == 4:
        return x.permute(0, 3, 1, 2).contiguous()
    if x.ndim == 5:
        return x.permute(0, 4, 1, 2, 3).contiguous()
## expand tensor ignoring batch_size and feature.
def expand_as(x, as_y, channel_dim):
    if x.ndim == as_y.ndim: return x
    assert x.ndim == 2, 'expand_as need the tensor ndim to be 2.' 
    if channel_dim == 1:
        if len(as_y.shape) == 3:
            res = x[:, :, None]
        # 2D image
        elif len(as_y.shape) == 4:
            res = x[:, :, None, None]
        # 3D image
        elif len(as_y.shape) == 5:
            res = x[:, :, None, None, None]
        expand_size = (-1, -1) + as_y.shape[2:]
    elif channel_dim == -1:
        if len(as_y.shape) == 3:
            res = x[:, None, :]
        # 2D image
        elif len(as_y.shape) == 4:
            res = x[:, None, None, :]
        # 3D image
        elif len(as_y.shape) == 5:
            res = x[:, None, None, None, :]
        expand_size = (-1,) + as_y.shape[1: -1] + (-1, )
    else:
        logger.error('Unsupported feature dimension.')
        raise NotImplementedError
    return res.expand(*expand_size)

### add is one way to merge x and embedding.
def new_add(x, c_emb, channel_dim = 1):
    if x is None: return c_emb
    if c_emb is None: return x
    assert c_emb.shape[0] == x.shape[0], \
        "Batch size inconsistency."
    assert c_emb.shape[channel_dim] == x.shape[channel_dim], \
        "Number of channels of x and condition encoder should be the same, get {} and {}".format(x.shape[1], c_emb.shape[1])
    if len(c_emb.shape) == 2:
        c_emb = expand_as(c_emb, x, channel_dim)
    # assert c_emb.shape == x.shape, f"c_emb and x should have the same shape, get {c_emb.shape} and {x.shape}."
    return x + c_emb

def new_cat(x, c_emb, channel_dim = 1):
    if x is None: return c_emb
    if c_emb is None: return x
    assert c_emb.shape[0] == x.shape[0], \
        "Batch size inconsistency."
    ## for concat, we don't need the number of channels to be the same.
    if len(c_emb.shape) == 2:
        c_emb = expand_as(c_emb, x, channel_dim)
    # assert c_emb.shape == x.shape, f"c_emb and x should have the same shape, get {c_emb.shape} and {x.shape}."
    return torch.concat([x, c_emb], dim = channel_dim)

# get activation
def get_act(act_name):
    assert act_name in ['relu', 'lrelu', 'elu', 'silu', 'swish', 'gelu',None], \
        'activation need to be one of [\'relu\', \'lrelu\', \'elu\', \'silu\' or \'swish\', None ]'
    if act_name == 'relu':
        return nn.ReLU()
    elif act_name == 'lrelu':
        return nn.LeakyReLU()
    elif act_name == 'elu':
        return nn.ELU()
    elif act_name == 'swish' or act_name == 'silu':
        return nn.SiLU()
    elif act_name == 'gelu':
        return nn.GELU()
    else:
        return nn.Identity()    
# get batch normalization
def get_bn(bn_dim, channel):
        if bn_dim == 1:
            return nn.BatchNorm1d(channel)
        elif bn_dim== 2:
            return nn.BatchNorm2d(channel)
        elif bn_dim == 3:
            return nn.BatchNorm3d(channel)       
def get_in(in_dim, channel):
        if in_dim == 1:
            return nn.InstanceNorm1d(channel)
        elif in_dim== 2:
            return nn.InstanceNorm2d(channel)
        elif in_dim == 3:
            return nn.InstanceNorm3d(channel)   
def get_norm(norm_name, norm_channel, dim = -1, num_norm_groups = 0):
        ### normalization layer
        if norm_name == 'batch' and dim > 0:
            return get_bn(dim, norm_channel), Norm.BATCH
        elif norm_name == 'group' and num_norm_groups > 0:
            return nn.GroupNorm(num_norm_groups, norm_channel), Norm.GROUP
        elif norm_name == 'layer':
            return nn.LayerNorm(norm_channel), Norm.LAYER
        elif norm_name == 'instance':
            return get_in(dim, norm_channel), Norm.INSTANCE
        elif norm_name == 'rms':
            assert hasattr(nn, 'RMSNorm'), "Cannot find 'RMSNorm' in torch.nn, please make sure that torch >= 2.4.0."
            return nn.RMSNorm(norm_channel), Norm.LAYER
        else:
            return nn.Identity(), Norm.NONE 
def get_ca(method, **kwargs):
    if method == 'ca':
        return CABlock(**kwargs)
    elif method == 'eca':
        return ECABlock(**kwargs)
    elif method == 'eca-ns':
        return ECANSBlock(**kwargs)
    else:
        logger.error(f'Unrecognized channel attention method: {method}, should be one of [ca, eca, eca-ns].')
        exit(1)
def get_context_injection(method, context_channel, data_channel, channel_dim):
    if method == 'gb' or method == 'gate_bias':
        return GateBiasBlock(context_channel, data_channel, channel_dim=channel_dim)
    elif method == 'gate':
        return GateBlock(context_channel, data_channel, channel_dim=channel_dim)
    elif method == 'bias':
        return BiasBlock(context_channel, data_channel, channel_dim=channel_dim)
    elif method == 'ca' or method == 'cross_attention' or method == 'cross_atten':
        return CrossAttentionBlock(in_channel = data_channel, 
                                        context_channel = context_channel,
                                        channel_dim = channel_dim, 
                                        skip_connection = True)
    else:
        logger.error('Unknown method for context injection, should be one of ["gate_bias", "gate", "bias", "cross_atten"].')
        exit(1)
def get_n_x_conv_atten(atten, in_channel, 
            num_heads = 3, 
            d_k = None, 
            qkv_bias = True, qk_scale = None, 
            atten_dropout = None, 
            dropout = None,
            skip_connection = True, 
            dim = None):
    if not atten:
        return None
    assert 'atten' in atten, r'atten should be a string like ${num}x${atten_type}, such as 10xatten.'
    atten = atten.split('x')
    num_atten = 1
    if len(atten) == 2:
        num_atten = int(atten[0])
        atten = atten[1]
    else:
        atten = atten[0]
    
    if atten == 'atten':
        return nn.Sequential(*[SelfAttentionBlock(in_channel=in_channel, num_heads=num_heads, d_k = d_k, 
        qkv_bias = qkv_bias, qk_scale = qk_scale, atten_dropout = atten_dropout, 
        dropout = dropout, skip_connection = skip_connection, channel_dim = 1) for _ in range(num_atten) ])
    elif atten == 'fft_atten':
        return nn.Sequential(*[FFTAttenBlock(dim = dim, in_channel=in_channel, num_heads=num_heads, 
        d_k = d_k, qkv_bias = qkv_bias, qk_scale = qk_scale, 
        atten_dropout = atten_dropout, dropout = dropout, 
        skip_connection = skip_connection) for _ in range(num_atten) ])
    else:
        logger.error(f'Unsupported attention block, should be one of ["atten", "fft_atten"], we got "{atten}".')
        exit(1)
def get_middle_channel(in_channel, out_channel, unit_channel = 16):
    tmp_channel = int(max(max(in_channel, out_channel) // 2, min(in_channel, out_channel)))
    return int(math.ceil(tmp_channel / unit_channel) * unit_channel)
def drop_path(x: torch.Tensor, prob: float = 0.0, inplace: bool = False) -> torch.Tensor:
    mask_shape: tuple[int] = (x.shape[0],) + (1,) * (x.ndim - 1) 
    # remember tuples have the * operator -> (1,) * 3 = (1,1,1)
    mask: torch.Tensor = x.new_empty(mask_shape).bernoulli_(1 - prob)
    # rescale down the activation of the input during training
    mask.div_(1 - prob)
    if inplace:
        x.mul_(mask)
    else:
        x = x * mask
    return x

class DropPath(nn.Module):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x
    
class SequentialT(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)
    def forward(self, x, *args, **kwargs): 
        for module in self._modules.values():
            x = module(x, *args, **kwargs)
        return x
    
## use pooling to achieve down-sampling
## alternatively, we can use convolution to achieve down-samping
## default: new_size = old_size / 2
class NormBlock(nn.Module):
    def __init__(self, norm = nn.Identity(), 
                norm_type = Norm.NONE, _channel_dim = 1):
        super().__init__()
        self._channel_dim = _channel_dim
        self.norm = norm
        self.norm_type = norm_type
    def normalize(self, x, t = None):
        if (self._channel_dim == -1 and \
            (self.norm_type == Norm.GROUP or self.norm_type == Norm.BATCH \
             or self.norm_type == Norm.INSTANCE)):
            x = channel_recover(x)
            x = self.norm(x)
            x = channel_transfer(x)
        elif self._channel_dim == 1 and self.norm_type == Norm.LAYER:
            x = channel_transfer(x)
            x = self.norm(x)
            x = channel_recover(x)
        else:
            x = self.norm(x)
        return x
    def forward(self, x):
        return self.normalize(x)

class MultipleBuildingBlocks(nn.Module):
    def __init__(self, BuildingBlock, in_channel, 
            out_channel = None, 
            n = 1, 
            hidden_channels = None,
            kwargs_list = {}, **kwargs):
        super().__init__()
        assert n is not None and n >= 1 or type(hidden_channels) == list, "Number of layers is not specified."
        out_channel = out_channel or in_channel
        if not type(hidden_channels) == list:
            hidden_channel = get_middle_channel(in_channel, out_channel)
            hidden_channels = [hidden_channel,] * (n - 1)
        channels = [in_channel,] + hidden_channels + [out_channel, ]
        building_blocks = []
        for i in range(len(channels) - 1):
            for k, v in kwargs_list.items():
                kwargs[k] = v[i]
            building_blocks.append(BuildingBlock(in_channel = channels[i], 
                                              out_channel = channels[i+1],
                                              **kwargs))
        self.building_blocks = SequentialT(*building_blocks)
    def forward(self, x, *args, **kwargs):
        x = self.building_blocks(x, *args, **kwargs)
        return x
        
class DownSamplingBlock(nn.Module):
    def __init__(self, dim, in_channel=None, out_channel=None, func = 'conv', scale_factor = 2):
        super().__init__()
        self.pooling = None
        self.conv = None
        self.scale_factor = scale_factor
        out_channel = out_channel or in_channel
        if func == 'conv':
            stride = scale_factor
            padding = scale_factor // 2
            kernel_size = 2 * scale_factor - scale_factor % 2
            if dim == 1:
                self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding = padding)
            if dim == 2:
                self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding = padding)
            if dim == 3:
                self.conv = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding = padding)
        else:
            pooling_size = scale_factor
            if func == 'max_pooling':
                if dim == 1:
                    self.pooling = nn.MaxPool1d(pooling_size)
                elif dim == 2:
                    self.pooling = nn.MaxPool2d(pooling_size)
                elif dim == 3:
                    self.pooling = nn.MaxPool3d(pooling_size)
            elif func == 'avg_pooling':
                if dim == 1:
                    self.pooling = nn.AvgPool1d(pooling_size)
                elif dim == 2:
                    self.pooling = nn.AvgPool2d(pooling_size)
                elif dim == 3:
                    self.pooling = nn.AvgPool3d(pooling_size)
            if in_channel != out_channel:
                if dim == 1:
                    self.proj = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0, bias = False)
                if dim == 2:
                    self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, bias = False)
                if dim == 3:
                    self.proj = nn.Conv3d(in_channel, out_channel, kernel_size=1, padding=0, bias = False)
            else:
                self.proj = nn.Identity()
    def forward(self, x, _ = None):
        if self.conv is not None:
            return self.conv(x)
        if self.pooling is not None:
            x = self.pooling(x)
        else:
            x = F.interpolate(x, size=tuple(s // self.scale_factor for s in x.shape[2:]))
        return self.proj(x)
    
## use transposed convolution to achieve up-sampling
## alternatively, we can use F.interpolate to achieve up-samping
## default: new_size = old_size * 2
class UpSamplingBlock(nn.Module):
    def __init__(self, dim, in_channel=None, out_channel=None, func = 'conv', scale_factor = 2):
        super().__init__()
        self.conv = None
        self.scale_factor = scale_factor
        out_channel = out_channel or in_channel
        if func == 'conv' and scale_factor > 1:
            stride = scale_factor
            padding = scale_factor // 2
            kernel_size = 2 * scale_factor - scale_factor % 2
            if dim == 1:
                self.conv = nn.ConvTranspose1d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding = padding)
            if dim == 2:
                self.conv = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding = padding)
            if dim == 3:
                self.conv = nn.ConvTranspose3d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding = padding)

        if in_channel != out_channel:
            if dim == 1:
                self.proj = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)
            if dim == 2:
                self.proj = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
            if dim == 3:
                self.proj = nn.Conv3d(in_channel, out_channel, kernel_size=1, padding=0)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x, _ = None):
        if self.conv is not None:
            x = self.conv(x)
        else:
            x = F.interpolate(x, scale_factor=self.scale_factor)
        return self.proj(x)

class PositionEmbeddingBlock(nn.Module):
    """
    ### Embeddings for time step t and position
    """
    ## channel dim has to be -1
    def __init__(self, out_channel, activation = 'silu', in_channel = None):
        """
        * `out_channel` is the number of embedding dimensions
        """
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        ## freq length = self.out_channel // 4
        embed_channel = self.in_channel * self.out_channel // 4 if self.in_channel else self.out_channel // 4
        self.dense1 = DenseBlock(embed_channel, self.out_channel, activation=activation)
        self.dense2 = DenseBlock(self.out_channel, self.out_channel, activation=None)

    def forward(self, pos: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        half_dim = self.out_channel // 8
        emb = math.log(1e4) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=pos.device) * -emb)
        emb = pos.unsqueeze(-1) * emb.reshape(pos.ndim *(1, ) + (-1, ))
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.in_channel:
            emb = emb.flatten(start_dim=-2)
        # Transform with the MLP
        emb = self.dense1(emb)
        emb = self.dense2(emb)
        return emb
class GateBlock(nn.Module):
    def __init__(self, context_channel, data_channel, channel_dim):
        super().__init__()
        self.hyper_gate = nn.Linear(context_channel, data_channel)
        self.channel_dim = channel_dim
    def forward(self, x, t):
        if t is None: return x
        gate = expand_as(self.hyper_gate(t), x, channel_dim = self.channel_dim)
        return (gate + 1) * x
class BiasBlock(nn.Module):
    def __init__(self, context_channel, data_channel, channel_dim):
        super().__init__()
        self.hyper_bias = nn.Linear(context_channel, data_channel, bias=False)
        self.channel_dim = channel_dim
    def forward(self, x, t):
        if t is None: return x
        bias = expand_as(self.hyper_bias(t), x, channel_dim = self.channel_dim)
        return x + bias

class GateBiasBlock(nn.Module):
    def __init__(self, context_channel, data_channel, channel_dim):
        super().__init__()
        self.hyper_bias = nn.Linear(context_channel, data_channel, bias=False)
        self.hyper_gate = nn.Linear(context_channel, data_channel)
        self.channel_dim = channel_dim
    def forward(self, x, t):
        if t is None: return x
        gate = expand_as(self.hyper_gate(t), x, channel_dim = self.channel_dim)
        bias = expand_as(self.hyper_bias(t), x, channel_dim = self.channel_dim)
        x = x * (1 + gate) + bias
        return x
## similar to learnable parameters in LayerNorm
class ScaleShiftBlock(nn.Module):
    def __init__(self, feature_shape):
        super().__init__()
        self.scale = nn.Parameter(
            torch.ones(feature_shape)
        )
        self.shift = nn.Parameter(
            torch.zeros(feature_shape)
        )
    def forward(self, x):
        return x * self.scale + self.shift
## transfer class label to one-hot vector which is encoded through FC block.
class OneHotEmbeddingBlock(nn.Module):
    def __init__(self, num_classes, out_channel, activation, apply_onehot = True):
        super().__init__()
        if num_classes == 2: num_classes = 1
        self.num_classes = num_classes
        self.apply_onehot = apply_onehot
        middle_channel = get_middle_channel(num_classes, out_channel) 
        #### old
        # middle_channel = min(int( max(self.num_classes, self.out_channel) / 2), 
        #                       self.num_classes, self.out_channel)
        self.dense1 = DenseBlock(self.num_classes, middle_channel, activation=activation)
        self.dense2 = DenseBlock(middle_channel, out_channel, activation=None)
    def forward(self, x):
        if self.apply_onehot and self.num_classes > 1:
            x = F.one_hot(x, self.num_classes).float()
        return self.dense2(self.dense1(x))
    
class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, scale, atten_dropout = None):
        """
        * `d_k` is the number of dimensions in each head
        """
        super().__init__()
        # Scale for dot-product attention
        self.scale = scale
        if atten_dropout is None or atten_dropout <= 0:
            self.atten_dropout = nn.Identity()
        else:
            self.atten_dropout = nn.Dropout(p=atten_dropout)

    def attention(self, q, k, v):
        # [batch_size, num_heads, seq, d_k]
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        atten = torch.einsum('bhik,bhjk->bhij', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        atten = atten.softmax(dim=-1)
        atten = self.atten_dropout(atten)
        res = torch.einsum('bhik,bhkj->bhij', atten, v)
        return res
    
class SelfAttentionBlock(AttentionBlock):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, in_channel, num_heads = 3, d_k = None, 
        qkv_bias = True, qk_scale = None, atten_dropout = None, 
        dropout = None, skip_connection = False, channel_dim = -1):
        """
        * `in_channel` is the number of channel in the input
        * `num_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        self.d_k = d_k or int(in_channel / num_heads) or 1
        scale = qk_scale or self.d_k ** -0.5
        super().__init__(scale=scale, atten_dropout=atten_dropout)
        self.in_channel = in_channel
        # Project x to query, key and values
        self.qkv = nn.Linear(in_channel, num_heads * self.d_k * 3, bias = qkv_bias)
        # project to original space
        self.proj = nn.Linear(num_heads * self.d_k, in_channel)
        self.num_heads = num_heads
        self.skip_connection = skip_connection
        self.channel_dim = channel_dim
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        # Get shape
        x_shape = x.shape
        if self.channel_dim == 1:
            x = channel_transfer(x)
        batch_size = x.shape[0]
        # Change `x` to shape `[batch_size, seq, in_channel]`
        x = x.reshape(batch_size, -1, self.in_channel)
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
        res = self.dropout(res)
        # Add skip connection
        if self.skip_connection:
            res = res + x
        if self.channel_dim == 1:
            res = channel_recover(res)
        # Change to shape `[batch_size, in_channel, height, width]`
        res = res.reshape(x_shape)
        return res
### might be useful when we have a context input
class CrossAttentionBlock(AttentionBlock):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, in_channel, context_channel = None, num_heads = 3, d_k = None, 
        qkv_bias = False, qk_scale = None, atten_dropout = None, 
        dropout = None, skip_connection = False, channel_dim = -1):
        self.d_k = d_k or int(in_channel / num_heads)
        scale = qk_scale or self.d_k ** -0.5
        super().__init__(scale=scale, atten_dropout=atten_dropout)

        # Default `d_k`
        # Project x to query, key and values
        self.in_channel = in_channel
        self.context_channel = context_channel or self.in_channel
        self.q = nn.Linear(in_channel, num_heads * self.d_k, bias = qkv_bias)
        self.kv = nn.Linear(context_channel, num_heads * self.d_k * 2, bias = qkv_bias)
        # project to original space
        self.proj = nn.Linear(num_heads * self.d_k, in_channel)
        self.num_heads = num_heads
        self.skip_connection = skip_connection
        self.channel_dim = channel_dim
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p=dropout)
    ### key and value are compute from x, query is compute from y
    ### usually y is a batch of vector
    def forward(self, x, y = None):
        if y is None: return x
        # Get shape
        x_shape = x.shape
        if self.channel_dim == 1:
            x = channel_transfer(x)
            # y = channel_transfer(y)
        batch_size = x.shape[0]
        # Change `x` to shape `[batch_size, seq, in_channel]`
        x = x.reshape(batch_size, -1, self.in_channel)
        y = y.reshape(batch_size, -1, self.context_channel)
        # assert x.shape[:-1] == y.shape[:-1], 'x and y should have the same shapes except for channel dimension.'
        # Get query, key, and values (concatenated) and shape it to `[2, batch_size, num_heads, seq, d_k]`
        kv = self.kv(y).reshape(batch_size, -1, 2, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, seq, d_k]
        q = self.q(x).reshape(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        # Split query, key, and values. 
        k, v = kv[0], kv[1]
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        res = self.attention(q, k, v)
        # Reshape to `[batch_size, seq, num_heads * d_k]`
        res = res.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.d_k)
        # Transform to `[batch_size, seq, in_channel]`
        res = self.proj(res)
        res = self.dropout(res)
        # Add skip connection
        if self.skip_connection:
            res = res + x

        if self.channel_dim == 1:
            res = channel_recover(res)
        # Change to shape `[batch_size, in_channel, height, width]`
        res = res.reshape(x_shape)
        return res

### FFT with attention: only for images
class FFTAttenBlock(nn.Module):
    def __init__(self, dim, in_channel: int, num_heads: int = 1, d_k: int = None, 
        qkv_bias = True, qk_scale = None, atten_dropout = None, 
        dropout = None, skip_connection = False):
        super().__init__()
        self.atten = SelfAttentionBlock(in_channel=in_channel, num_heads=num_heads, d_k = d_k, 
            qkv_bias = qkv_bias, qk_scale = qk_scale, atten_dropout = atten_dropout, 
            dropout = dropout, skip_connection = False, channel_dim = 1)
        self.dim = tuple(range(2, 2 + dim))
        self.fft = partial(fft.rfftn, dim = self.dim)
        self.ifft = partial(fft.irfftn, dim = self.dim)
        self.skip_connection = skip_connection
    def forward(self, x):
        res = self.fft(x)
        res = torch.view_as_real(res)
        res = self.atten(res)
        res = torch.view_as_complex(res.contiguous())
        res = self.ifft(res)
        if self.skip_connection:
            res = res + x
        return res

class ContextInjectionBlock(nn.Module):
    def __init__(self, time_channel, condition_channel, out_channel, 
                 time_injection, condition_injection, 
                 channel_dim, condition_first = False):
        super().__init__()
        self.merger = None
        if time_channel > 0 and condition_channel > 0:
            if condition_injection == 'cat_to_time':
                time_channel = time_channel + condition_channel
                condition_channel = 0
                self.merger = partial(new_cat, channel_dim = channel_dim)
            elif condition_injection == 'add_to_time':
                assert time_channel == condition_channel, \
                    'Time and condition embedding need to have the same channel dimension for addition.'
                condition_channel = 0
                self.merger = partial(new_add, channel_dim = channel_dim)
        self.time = None
        self.cond = None
        if time_channel > 0:
            self.time = get_context_injection(time_injection, time_channel, out_channel, channel_dim=channel_dim)
        if condition_channel > 0:
            self.cond = get_context_injection(condition_injection, condition_channel, out_channel, channel_dim=channel_dim)
        self.condition_first = condition_first
    def forward(self, x, t, c):
        if self.merger:
            t = self.merger(t, c)
            c = None
        if not self.condition_first:
            if self.time:
                x = self.time(x, t)
            if self.cond:
                x = self.cond(x, c)
        else:
            if self.cond:
                x = self.cond(x, c)
            if self.time:
                x = self.time(x, t)
        return x

class DenseBlock(NormBlock):
    def __init__(self, in_channel, out_channel, time_channel = 0, 
                norm = None, num_norm_groups = 0, bias = True,
                activation = 'relu', dropout=None, 
                time_injection = 'gate_bias', 
                condition_channel = 0, 
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(_channel_dim = -1)
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.linear = nn.Linear(in_channel, out_channel, bias = bias)
        ## normalization layer

        self.norm, self.norm_type = get_norm(norm, out_channel, 1, num_norm_groups)
        if dropout is None or dropout <= 0:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p=dropout)
        # activation function
        self.act = get_act(activation)
        self.cinj = None
        if time_channel > 0 or condition_channel > 0:
            self.cinj = ContextInjectionBlock(time_channel = time_channel,
                condition_channel = condition_channel,
                out_channel = out_channel,
                time_injection=time_injection,
                condition_injection=condition_injection,
                channel_dim = -1,
                condition_first = condition_first)
    def forward(self, x, t = None, c = None):
        size = x.shape[1:-1]
        if len(size) > 1:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.linear(x)
        x = self.normalize(x)
        x = self.dropout(self.act(x))
        if self.cinj:
            x = self.cinj(x, t, c)
        if len(size) > 1:
            x = x.reshape(*((x.shape[0], ) + size + (x.shape[-1], )))
        return x
    @staticmethod
    def is_sequence_modeling():
        return False
    
## Multi Layer Perception block
class MultiLayerPerceptionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, 
                n = 1, hidden_channels = None, 
                time_channel = 0, norm = None, 
                bias = True,
                num_norm_groups = 0, 
                activation = 'relu', dropout=None, 
                final_activation = True,
                time_injection = 'gate_bias',
                condition_channel = 0, 
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        assert n is not None and n >= 1 or type(hidden_channels) == list, \
            "Number of layers is not specified."
        if not type(hidden_channels) == list or len(hidden_channels) == 0:
            hidden_channel = get_middle_channel(in_channel, out_channel)
            hidden_channels = [hidden_channel,] * (n - 1)
        channels = [in_channel, ] + hidden_channels + [out_channel, ]
        module_sequence = [DenseBlock( in_channel = channels[idx], 
                    out_channel = channels[idx+1], time_channel = time_channel, 
                    time_injection = time_injection, norm = norm,
                    num_norm_groups = num_norm_groups, 
                    activation = activation, dropout = dropout, 
                    bias = bias,
                    condition_channel = condition_channel,
                    condition_injection = condition_injection,
                    condition_first = condition_first,
                    ) for idx in range(len(channels) - 2)]
        if not final_activation:
            module_sequence = module_sequence + [DenseBlock( in_channel = channels[-2], 
                        out_channel = channels[-1], 
                        bias = bias,
                        norm = None,
                        activation = None, 
                        ), ]
        else:
            module_sequence = module_sequence + [DenseBlock( in_channel = channels[-2], 
                    out_channel = channels[-1], time_channel = time_channel, 
                    time_injection = time_injection, norm = norm,
                    num_norm_groups = num_norm_groups, 
                    activation = activation, dropout = dropout, 
                    bias = bias,
                    condition_channel = condition_channel,
                    condition_injection = condition_injection,
                    condition_first = condition_first,
                    ), ]
        self.mlp = SequentialT(*module_sequence)
    def forward(self, x, t = None, c = None):
        x = self.mlp(x, t, c)
        return x 

class DoubleDenseBlock(nn.Module):
    def __init__(self, in_channel, out_channel, 
        time_channel = 0, norm = None, num_norm_groups = 0, 
        activation = 'relu', dropout=None, bias = True,
        time_injection = 'gate_bias', 
        condition_channel = 0, condition_injection = 'gate_bias',
        condition_first = False,
        **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.dense1 = DenseBlock(in_channel = in_channel, out_channel = out_channel, 
            norm=norm, num_norm_groups = num_norm_groups, activation = activation, 
            dropout = dropout, 
            bias = bias,
            time_channel = time_channel, 
            time_injection = time_injection,
            condition_channel = condition_channel, 
            condition_injection = condition_injection,
            condition_first=condition_first)
        self.dense2 = DenseBlock(in_channel = out_channel, out_channel = out_channel, 
            norm=norm, num_norm_groups = num_norm_groups, activation = activation, 
            dropout = dropout, 
            bias=bias)    
        self.time_channel = time_channel
        self.condition_channel = condition_channel
    def forward(self, x, t = None, c = None):
        x = self.dense1(x, t, c)
        x = self.dense2(x)
        return x
    @staticmethod
    def is_sequence_modeling():
        return False
    
class ResDenseBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_channel = 0, norm = None,
        num_norm_groups = 0, activation = 'relu', dropout=None, bias = True,
        time_injection = 'gate_bias', 
        condition_channel = 0, condition_injection = 'gate_bias', 
        condition_first = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.dense1 = DenseBlock(in_channel = in_channel, out_channel = out_channel, 
            norm=norm, num_norm_groups = num_norm_groups, activation = activation, 
            dropout = dropout, bias=bias,
            time_channel = time_channel,
            time_injection = time_injection,
            condition_channel = condition_channel,
            condition_injection = condition_injection,
            condition_first=condition_first)
        self.dense2 = DenseBlock(in_channel = out_channel, out_channel = out_channel, 
            norm=norm, num_norm_groups = num_norm_groups, activation = None, 
            dropout = dropout, bias=bias)  
        self.act = get_act(activation)  
        if in_channel != out_channel:
            ## without normalization
            self.shortcut = DenseBlock(in_channel=in_channel, out_channel=out_channel, 
                                      activation=None, bias=False,
                                      norm=norm, num_norm_groups=num_norm_groups, )
        else:
            self.shortcut = nn.Identity()    
        self.time_channel = time_channel
        self.condition_channel = condition_channel
    def forward(self, x, t = None, c = None):
        h = self.dense1(x, t, c)
        h = self.dense2(h)
        out = self.act(h + self.shortcut(x))
        return out
    @staticmethod
    def is_sequence_modeling():
        return False

class CABlock(nn.Module):
    """Constructs a CA module.

    Args:
        channel: Number of channels of the input feature map
        k: Adaptive selection of kernel size
    """
    def __init__(self, dim, channel_dim = 1, **kwargs):
        super().__init__()
        if dim == 1:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif dim == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dim = dim
        self.atten = SelfAttentionBlock(in_channel=1, channel_dim=-1, **kwargs)
        self.sigmoid = nn.Sigmoid()
        self.channel_dim = channel_dim
        assert channel_dim == 1 or channel_dim == -1, "Channel dim should be 1 or -1."
    def forward(self, x):
        if self.channel_dim == -1:
            x = channel_recover(x)
        # feature descriptor on the global spatial information
        B, C = x.shape[0], x.shape[1]

        y = self.avg_pool(x)
        y = y.reshape(B, C, 1)
        # Two different branches of ECA module
        y = self.atten(y).squeeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x * expand_as(y, x, channel_dim=1)
        if self.channel_dim == -1:
            x = channel_transfer(x)
        return x
## efficient channel attention from ECANet (https://arxiv.org/pdf/1910.03151)
class ECABlock(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k: Adaptive selection of kernel size
    """
    def __init__(self, dim, kernel_size=3, channel_dim = 1, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        if dim == 1:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif dim == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dim = dim
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
        self.channel_dim = channel_dim
        assert channel_dim == 1 or channel_dim == -1, "Channel dim should be 1 or -1."
    def forward(self, x):
        # feature descriptor on the global spatial information
        if self.channel_dim == -1:
            x = channel_recover(x)
        B, C = x.shape[0], x.shape[1]

        y = self.avg_pool(x)
        y = y.reshape(B, C, 1)
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).squeeze(1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x * expand_as(y, x, channel_dim=1)
        if self.channel_dim == -1:
            x = channel_transfer(x)
        return x
## eca-ns, convolution only processed in k neighbor channel
class ECANSBlock(nn.Module):
    def __init__(self, dim, channel, kernal_size = 3, channel_dim = 1, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        if dim == 1:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif dim == 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.dim = dim
        self.k = kernal_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernal_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()
        self.channel_dim = channel_dim
        assert channel_dim == 1 or channel_dim == -1, "Channel dim should be 1 or -1."
        self.unfold=partial(nn.functional.unfold, kernel_size=(self.k,), padding=((self.k - 1) // 2, ))
    def forward(self, x):
        if self.channel_dim == -1:
            x = channel_recover(x)
        B, C = x.shape[0], x.shape[1]
        y = self.avg_pool(x)
        y = y.reshape(B, C, 1, 1)
        y = nn.functional.unfold(y.transpose(-1, -3), kernel_size=(1, self.k), padding=(0, (self.k - 1) // 2 ))
        y = self.conv(y.transpose(-1, -2)).squeeze(-1)
        y = self.sigmoid(y)
        x = x * expand_as(y, x, channel_dim=1)
        if self.channel_dim == -1:
            x = channel_transfer(x)
        return x

##### convolution block and its variations
class ConvBlock(NormBlock):
    def __init__(self, dim, in_channel, out_channel, 
            time_channel = 0, kernel_size = 3, 
            padding = 1, depthwise = False, bias = True, activation = 'relu', 
            norm='batch', num_norm_groups = 0, time_injection = 'gate_bias',
            condition_channel = 0, condition_injection = 'gate_bias',
            atten = None, num_heads = 1, d_k = None, 
            qkv_bias = True, qk_scale = None, atten_dropout = None, 
            dropout = None, skip_connection = True, 
            condition_first = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        ### convolution layer
        groups = 1
        conv_out_channel = out_channel
        if depthwise:
            groups = in_channel
            conv_out_channel = in_channel
        if dim ==1:
            self.conv = nn.Conv1d(in_channel, conv_out_channel, kernel_size, padding = padding, groups = groups,
                                  bias = bias)
        elif dim== 2:
            self.conv = nn.Conv2d(in_channel, conv_out_channel, kernel_size, padding = padding, groups = groups,
                                  bias = bias)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channel, conv_out_channel, kernel_size, padding = padding, groups = groups,
                                  bias = bias)
        self.proj = nn.Identity()
        if not conv_out_channel == out_channel:
            if dim ==1:
                self.proj = nn.Conv1d(conv_out_channel, out_channel, kernel_size=1, bias = False)
            elif dim== 2:
                self.proj = nn.Conv2d(conv_out_channel, out_channel, kernel_size=1, bias = False)
            elif dim == 3:
                self.proj = nn.Conv3d(conv_out_channel, out_channel, kernel_size=1, bias = False)

        self.norm, self.norm_type = get_norm(norm, out_channel, dim, num_norm_groups)
        self.atten = get_n_x_conv_atten(atten, out_channel, num_heads=num_heads, d_k=d_k, 
                               qkv_bias = qkv_bias, qk_scale = qk_scale, 
                               atten_dropout = atten_dropout, 
                               dropout = dropout, 
                               skip_connection = skip_connection, dim = dim)
        # activation function
        self.act = get_act(activation)

        self.cinj = None
        if time_channel > 0 or condition_channel > 0:
            self.cinj = ContextInjectionBlock(time_channel = time_channel,
                condition_channel = condition_channel,
                out_channel = out_channel,
                time_injection=time_injection,
                condition_injection=condition_injection,
                channel_dim = 1,
                condition_first = condition_first)       
    def forward(self, x, t = None, c = None):
        x = self.proj(self.conv(x))
        x = self.act(self.normalize(x))
        if self.atten:
            x = self.atten(x)

        if self.cinj:
            x = self.cinj(x, t, c)
        return x
## double convolution
class DoubleConvBlock(nn.Module):
    def __init__(self, dim, in_channel, out_channel, 
            time_channel = 0, kernel_size = 3, 
            padding = 1, depthwise = False, bias = True, 
            activation = 'relu', norm='batch', num_norm_groups = 0, 
            time_injection = 'gate_bias', 
            atten = None, num_heads = 1, d_k = None, 
            qkv_bias = True, qk_scale = None, atten_dropout = None, 
            dropout = None, skip_connection = True,  
            condition_channel = 0, condition_injection = 'gate_bias',
            condition_first = False,
            **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.conv1 = ConvBlock(dim = dim, in_channel=in_channel, out_channel=out_channel, 
                               kernel_size=kernel_size, padding=padding, 
                               depthwise = depthwise, bias = bias, activation=activation, 
                               norm=norm, num_norm_groups=num_norm_groups,
                               time_channel=time_channel, time_injection=time_injection,
                               condition_channel = condition_channel, condition_injection=condition_injection,
                               condition_first=condition_first)
        self.conv2 = ConvBlock(dim = dim, in_channel=out_channel, out_channel=out_channel, 
                               kernel_size=kernel_size, padding=padding, 
                               depthwise = depthwise, bias = bias, activation=activation,
                               norm=norm, num_norm_groups=num_norm_groups)
        self.atten = get_n_x_conv_atten(atten, out_channel, num_heads=num_heads, d_k=d_k, 
                               qkv_bias = qkv_bias, qk_scale = qk_scale, 
                               atten_dropout = atten_dropout, 
                               dropout = dropout, 
                               skip_connection = skip_connection, dim = dim)
        self.time_channel = time_channel
        self.condition_channel = condition_channel
    def forward(self, x, t = None, c = None):
        # First convolution layer
        x = self.conv1(x, t, c)
        # Second convolution layer
        x = self.conv2(x)
        if self.atten:
            x = self.atten(x)
        return x 

# residual double convolution
class ResConvBlock(nn.Module):
    def __init__(self, dim, in_channel, out_channel, 
        time_channel = 0, kernel_size = 3, 
        padding = 1, depthwise = False, bias = True, 
        activation = 'relu', norm='batch', num_norm_groups = 0, 
        atten = None, num_heads = 1, d_k = None,
        qkv_bias = True, qk_scale = None, atten_dropout = None, 
        dropout = None, skip_connection = True, 
        time_injection = 'gate_bias', 
        condition_channel = 0, condition_injection = 'gate_bias',
        condition_first = False,
        **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.conv1 = ConvBlock(dim = dim, in_channel=in_channel, out_channel=out_channel, 
                               kernel_size=kernel_size, padding=padding, 
                               depthwise = depthwise, bias = bias, activation=activation, 
                               norm=norm, num_norm_groups=num_norm_groups, 
                               time_channel=time_channel, time_injection=time_injection,
                               condition_channel = condition_channel,
                               condition_injection = condition_injection,
                               condition_first=condition_first)
        
        self.conv2 = ConvBlock(dim = dim, in_channel=out_channel, out_channel=out_channel, 
                               kernel_size=kernel_size, padding=padding, 
                               depthwise = depthwise, bias = bias, activation=None,
                               norm=norm, num_norm_groups=num_norm_groups, )
        self.act = get_act(activation)
        self.atten = get_n_x_conv_atten(atten, out_channel, num_heads=num_heads, d_k=d_k, 
                               qkv_bias = qkv_bias, qk_scale = qk_scale, 
                               atten_dropout = atten_dropout, 
                               dropout = dropout, 
                               skip_connection = skip_connection, dim = dim)
        # If the number of input channel is not equal to the number of output channel we have to
        # project the shortcut connection
        if in_channel != out_channel:
            ## with un-biased normalization
            self.shortcut = ConvBlock(dim=dim, in_channel=in_channel, out_channel=out_channel, 
                                      kernel_size=1, padding = 0, activation=None, bias=False,
                                      norm=norm, num_norm_groups=num_norm_groups, )
        else:
            self.shortcut = nn.Identity()        
        self.time_channel = time_channel
        self.condition_channel = condition_channel

    def forward(self, x, t = None, c = None):
        # First convolution layer
        _x = self.conv1(x, t, c)
        # Second convolution layer
        _x = self.conv2(_x)
        out = self.act( _x + self.shortcut(x))
        if self.atten:
            out = self.atten(out)
        return out


## Combine inputs with different shapes
class CombineLayer(nn.Module):
    def __init__(self, in_channels, out_channel, target_size, mode='nearest',
                combine = 'cat', position_embedding = True, apply_fft = False):
        super().__init__()
        assert type(in_channels) == list and len(in_channels) > 2, \
            'the input of combine layer should be a list whose length is larger than 1'
        
        self.length = len(in_channels)
        self.dim = len(target_size)
        self.resize = partial(F.interpolate, size=target_size, mode=mode)
        if combine == 'cat':
            self.combine = partial(torch.cat, dim = 1)
            in_channel = sum(in_channels)
        elif combine == 'add':
            self.combine = sum
            ### make sure all the input has the same channel
            in_channel = in_channels[0]
        self.target_size = target_size
        self.apply_fft = apply_fft
        if self.apply_fft:
            fft_dim = tuple(range(2, 2 + self.dim))
            self.fft = partial(fft.rfftn, dim = fft_dim, norm='ortho')
            self.ifft = partial(fft.irfftn, dim = fft_dim, norm='ortho')

        self.pos_emb = None
        if position_embedding:
            pos_emb_shapes = [[1, ic] + list(self.target_size) for ic in in_channels ]
            if apply_fft: 
                for i in range(len(pos_emb_shapes)):
                    pos_emb_shapes[i][-1] = pos_emb_shapes[i][-1] // 2 + 1
                    pos_emb_shapes[i] = pos_emb_shapes[i] + [2,]
            self.pos_emb = nn.ParameterList([nn.Parameter(torch.zeros(pos_emb_shapes[i])) for i in range(self.length)])
            for i in range(self.length):
                nn.init.trunc_normal_(self.pos_emb[i], std=.02)
        self.final_conv = ConvBlock(dim = self.dim, in_channel = in_channel, out_channel = out_channel, 
                                    activation=None, kernel_size=1, padding=0, norm=None)

    def forward(self, x):
        assert (isinstance(x, list) or isinstance(x, tuple)) \
            and len(x) == self.length, \
            'In consistent length between input and combine layer.'
        for i in range(len(x)):
            x[i] = self.resize(x[i])
            if self.pos_emb:
                if self.apply_fft:
                    ### fft to time frequency space
                    fx = self.fft(x[i])
                    w = torch.view_as_complex(self.pos_emb[i])
                    fx = fx * w
                    ### back to pixel space
                    x[i] = self.ifft(fx)
                else:
                    x[i] = x[i] + self.pos_emb[i]
        com_x = self.combine(x)
        return self.final_conv(com_x)


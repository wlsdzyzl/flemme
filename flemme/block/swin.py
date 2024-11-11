### some parts of the code are adopted from https://github.com/ziyangwang007/Mamba-UNet/blob/main/code/networks/swin_transformer_unet_skip_expand_decoder_sys.py
from .common import *
import torch
import torch.nn as nn
import numpy as np
from flemme.logger import get_logger
from einops import rearrange
import math
logger = get_logger('swin_block')
### blocks about swin transformer
### patch related blocks

### use convolution to transfer image to patches embedding
class PatchConstructionBlock(NormBlock):
    def __init__(self, dim, patch_size, in_channel, out_channel, norm = 'layer', num_groups = 4, 
                 activation = None, order="cn"):
        super().__init__()
        self.dim = dim
        ### convolution layer
        if dim ==1:
            self.conv = nn.Conv1d(in_channel, out_channel, patch_size, stride = patch_size)
        elif dim== 2:
            self.conv = nn.Conv2d(in_channel, out_channel, patch_size, stride = patch_size)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channel, out_channel, patch_size, stride = patch_size)

        norm_channel = out_channel
        if order.index('n') < order.index('c'):
            norm_channel = in_channel
        ### normalization layer
        self.norm, self.norm_type = get_norm(norm, norm_channel, self.dim, num_groups)
        # activation function
        self.act = get_act(activation)
        self.order = order
    def forward(self, x, _ = None):
        for m in self.order:
            if m == 'n':
                x = self.normalize(x)
            elif m == 'c':
                x = self.conv(x)
        x = self.act(x)
        ## B * C_in * H * W -> B * Ph * Pw * C_out
        return channel_transfer(x).contiguous()

### use convolution to transfer image to patches embedding
class PatchRecoveryBlock(NormBlock):
    def __init__(self, dim, patch_size, 
                 in_channel, out_channel, norm = None, num_groups = 4, 
                 activation = None, order="en"):
        super().__init__(_channel_dim = -1)
        self.expand = PatchExpansionBlock(dim = dim,
                                    in_channel = in_channel, out_channel = in_channel,
                                    factor = patch_size, norm = norm, num_groups = num_groups, 
                                    activation = activation, order = "en")
        self.dim = dim
        if dim ==1:
            self.conv = nn.Conv1d(in_channel, out_channel, 1)
        elif dim== 2:
            self.conv = nn.Conv2d(in_channel, out_channel, 1)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channel, out_channel, 1)
        norm_channel = in_channel
        ## norm is after convolution
        if order.index('n') > order.index('e'):
            norm_channel = out_channel
            self._channel_dim = 1
            
        ### normalization layer
        self.norm, self.norm_type = get_norm(norm, norm_channel, self.dim, num_groups)
        # activation function
        self.act = get_act(activation)
        self.order = order
    def forward(self, x, _ = None):
        for m in self.order:
            if m == 'n':
                x = self.normalize(x)
            elif m == 'e':
                x = self.expand(x)
                x = channel_recover(x)
                x = self.conv(x)
        x = self.act(x)
        ## B , H * W, C -> B, C, H, W
        return x
### not that in the rest of this file, the channel should be the last dimension    
class PatchMergingBlock(NormBlock):

    def __init__(self, dim, in_channel, out_channel = None, 
                 factor = 2, norm = None, num_groups = 4, 
                 activation = None, order="mn"):
        super().__init__(_channel_dim = -1)
        self.dim = dim
        self.in_channel = in_channel
        self.out_channel = out_channel or 2 * self.in_channel
        assert factor > 0 and (factor & (factor - 1)) == 0, \
            "Patch merging factor should be a power of 2."
        self.merge_times = int(math.log2(factor))
        self.reduction = nn.Linear((2 ** self.dim)**self.merge_times * in_channel, self.out_channel, bias=False)

        norm_channel = self.out_channel
        if order.index('n') < order.index('m'):
            norm_channel = in_channel
        ### normalization layer
        self.norm, self.norm_type = get_norm(norm, norm_channel, self.dim, num_groups)
        # activation function
        self.act = get_act(activation)
        self.order = order
    def merge(self, x):
        for _ in range(self.merge_times):
            if self.dim == 2:
                B, H, W, C = x.shape
                assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

                x0 = x[:, 0::2, 0::2, :]  
                x1 = x[:, 1::2, 0::2, :]  
                x2 = x[:, 0::2, 1::2, :]  
                x3 = x[:, 1::2, 1::2, :]  
                x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
            else:
                B, H, W, D, C = x.shape
                assert H % 2 == 0 and W % 2 == 0 and D % 2 == 0, f"x size ({H}*{W}*{D}) are not even."

                x0 = x[:, 0::2, 0::2, 0::2, :]  
                x1 = x[:, 1::2, 0::2, 0::2, :]  
                x2 = x[:, 0::2, 1::2, 0::2, :]  
                x3 = x[:, 1::2, 1::2, 0::2, :]  
                x4 = x[:, 0::2, 0::2, 1::2, :]  
                x5 = x[:, 1::2, 0::2, 1::2, :]  
                x6 = x[:, 0::2, 1::2, 1::2, :]  
                x7 = x[:, 1::2, 1::2, 1::2, :]  
                x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 D/2 8*C
        x = self.reduction(x)
        return x
    def forward(self, x, _ = None):
        """
        x: B, H*W, C
        """
        for m in self.order:
            if m == 'm':
                x = self.merge(x)
            if m == 'n':
                x = self.normalize(x)
        x = self.act(x)
        return x

### speficy the out channel of patch expansion
### final expand can be implemented using this block
class PatchExpansionBlock(NormBlock):
    def __init__(self, dim, in_channel, out_channel = None, factor = 2, 
                norm = None, num_groups = 4, activation = None, order="en"):
        super().__init__(_channel_dim = -1)
        self.dim = dim
        self.in_channel = in_channel
        self.out_channel = out_channel or in_channel // factor
        l_out_channel = self.out_channel * (factor ** self.dim)
        ## expansion factor
        self.factor = int(factor)
        assert self.factor > 1 and self.factor % 2 == 0 or self.factor == 1, \
            "Patch expansion factor should be an even number or 1."
        self.expand = nn.Linear(
            in_channel, l_out_channel, bias=False) if in_channel != l_out_channel else nn.Identity()

        norm_channel = self.out_channel
        if order.index('n') < order.index('e'):
            norm_channel = in_channel
        ### normalization layer
        self.norm, self.norm_type = get_norm(norm, norm_channel, self.dim, num_groups)
        # activation function
        self.act = get_act(activation)
        self.order = order

    def forward(self, x, _ = None):
        """
        for 2D:
        input: x: B, H*W, C
        outout: x: B, H*2, W*2, C, 
        """
        for m in self.order:
            if m == 'e':
                x = self.expand(x)
                
                if self.dim == 2:
                    C = x.shape[-1]
                    x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c',
                                p1=self.factor, p2=self.factor, c=C//(self.factor ** self.dim))
                else:
                    C = x.shape[-1]
                    x = rearrange(x, 'b h w d (p1 p2 p3 c)-> b (h p1) (w p2) (d p3) c',
                                p1=self.factor, p2=self.factor, p3=self.factor, c=C//(self.factor ** self.dim))
            elif m == 'n':
                x = self.normalize(x)
        x = self.act(x)
        return x

### only suitable for 2D or 3D images
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    if x.ndim == 4:
        B, H, W, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0],
                W // window_size[1], window_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
        ).view(-1, window_size[0], window_size[1], C)
    elif x.ndim == 5:
        B, H, W, D, C = x.shape
        x = x.view(B, H // window_size[0], window_size[0],
                W // window_size[1], window_size[1], D // window_size[2], window_size[2], C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous(
        ).view(-1, window_size[0], window_size[1], window_size[2], C)        
    else:
        logger.error('Only 2D or 3D images are supported for window operation')
    return windows


def window_reverse(windows, window_size, patch_image_size):
    """
    Args:
        windows: (num_windows*B, window_size, C)
        window_size (int): Window size
    Returns:
        x: (B, H, W, C)
    """
    
    if len(window_size) == 2:
        H, W = patch_image_size
        B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
        x = windows.view(B, H // window_size[0], W // window_size[1],
                        window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    else:
        H, W, D = patch_image_size
        B = int(windows.shape[0] / (H * W * D / window_size[0] / window_size[1] / window_size[2]))
        x = windows.view(B, H // window_size[0], W // window_size[1], D // window_size[2],
                        window_size[0], window_size[1], window_size[2], -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
    return x

class WindowAttentionBlock(SelfAttentionBlock):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width and (depth) of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, in_channel, window_size, num_heads = 3, 
        d_k = None, qkv_bias=True, qk_scale = None, skip_connection = False, 
        atten_dropout=None, dropout = None):

        super().__init__(in_channel = in_channel, num_heads = num_heads, d_k = d_k, qkv_bias = qkv_bias, 
            qk_scale = qk_scale, atten_dropout = atten_dropout, dropout = dropout, 
            skip_connection = skip_connection)
        assert isinstance(window_size, list), "window_size should be a list."
        self.window_size = window_size  # Wh, Ww
        self.window_length = np.prod(self.window_size)
        # Default `d_k`
        self.dim = len(self.window_size)
        
        if self.dim == 2:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])

            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - \
                coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(
                1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        elif self.dim == 3:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2 *Wd-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords_d = torch.arange(self.window_size[2])

            coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d], indexing='ij'))  # 3, Wh, Ww, Wd
            coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
            relative_coords = coords_flatten[:, :, None] - \
                coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd
            relative_coords = relative_coords.permute(
                1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
            # shift to start from 0
            # the max value would be (2M-2), 
            relative_coords[:, :, 0] += self.window_size[0] - 1  
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1

            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wd, Wh*Ww*Wd

        self.register_buffer("relative_position_index",
                             relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    def attention(self, q, k, v, mask):
        ### B, nH, N, d_k
        B, N = q.shape[0], q.shape[2]
        atten = torch.einsum('bhik,bhjk->bhij', q, k) * self.scale
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_length, self.window_length, -1)  # Wh*Ww,Wh*Ww,nH
        
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        atten = atten + relative_position_bias.unsqueeze(0)
        ## if mask is not None
        if mask is not None:
            nW = mask.shape[0]
            atten = atten.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            atten = atten.view(-1, self.num_heads, N, N)
        atten = atten.softmax(dim=-1)
        atten = self.atten_dropout(atten)
        res = torch.einsum('bhik,bhkj->bhij', atten, v)
        return res
    def forward(self, x, mask = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
            note that N = Wh*Ww*Wd
        """
        B = x.shape[0]
        # 3 B nH N d_k
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]
        res = self.attention(q, k, v, mask).transpose(1, 2).reshape(B, -1, self.num_heads * self.d_k)
        res = self.proj(res)
        res = self.dropout(res)
        if self.skip_connection:
            res = res + x
        return res

class VisionTransformerBlock(nn.Module):
    r""" Vision Transformer Block.

    Args:
        in_channel (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden in_channel to embedding in_channel.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_in_channel ** -0.5 if set.
        dropout (float, optional): Dropout rate. Default: 0.0
        atten_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, in_channel, out_channel = None, 
                time_channel = 0, num_heads = 1, 
                window_size=[7, 7], shift_size=0,
                mlp_hidden_ratio=[4.0,], qkv_bias=True, 
                qk_scale=None, dropout=0., atten_dropout=0., drop_path=0.,
                activation = 'silu', norm = 'layer', num_groups = 4, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.in_channel = in_channel
        self.out_channel = out_channel or in_channel
        self.num_heads = num_heads
        self.dim = dim
        if self.dim <2 or self.dim > 3:
            logger.error('Vision-transformer block only support 2D / 3D images.')
            exit()
        ## window size is a single number
        ## latter we need to transfer window size to tuple.
        
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.norm1 = NormBlock(*(get_norm(norm, in_channel, self.dim, num_groups) + (-1,)))
        self.norm2 = NormBlock(*(get_norm(norm, in_channel, self.dim, num_groups) + (-1,)))
        self.atten = SelfAttentionBlock(
            in_channel, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, atten_dropout=atten_dropout, dropout=dropout,
            channel_dim = -1)

        self.drop_path = DropPath(
            p = drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_channels = [int(in_channel * r) for r in mlp_hidden_ratio]
        self.mlp = MultiLayerPerceptionBlock(in_channel=in_channel, out_channel=self.out_channel, 
                        hidden_channels=mlp_hidden_channels,
                        activation=activation, dropout=dropout)
        self.dense = nn.Linear(self.in_channel, self.out_channel) if self.in_channel != self.out_channel else nn.Identity()
        # self.dense = DenseBlock(in_channel = in_channel, out_channel = self.out_channel, 
        #             activation = None) if in_channel != self.out_channel else nn.Identity()
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.hyper_bias = nn.Linear(time_channel, out_channel, bias=False)
            self.hyper_gate = nn.Linear(time_channel, out_channel)
    def forward(self, x, t = None):
        shortcut = x
        x = self.norm1.normalize(x)
        x = self.atten(x)
        x = shortcut + self.drop_path(x)
        x = self.dense(x) + self.drop_path(self.mlp(self.norm2.normalize(x)))
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x


## swin transformer block doesn't channge the input channel
class SwinTransformerBlock(VisionTransformerBlock):
    r""" Swin Transformer Block.

    Args:
        in_channel (int): Number of input channels.
        patch_image_size (list[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size. We only support square window
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden in_channel to embedding in_channel.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_in_channel ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        atten_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, patch_image_size, in_channel, 
                out_channel = None, time_channel = 0, 
                num_heads = 1, window_size=[7, 7], shift_size=0,
                mlp_hidden_ratio=[4.0,], 
                qkv_bias=True, qk_scale=None, dropout=0., 
                atten_dropout=0., drop_path=0.,
                activation = 'silu', norm = 'layer', num_groups = 4, **kwargs):
        super().__init__(dim = len(patch_image_size), in_channel = in_channel, out_channel = out_channel, 
                 num_heads = num_heads, mlp_hidden_ratio=mlp_hidden_ratio, 
                 qkv_bias=qkv_bias, qk_scale=qk_scale, 
                 dropout=dropout, atten_dropout=atten_dropout, 
                 drop_path=drop_path, activation=activation, 
                 norm = norm, num_groups = num_groups)
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        self.patch_image_size = patch_image_size
        assert isinstance(patch_image_size, list), "patch_image_size should be a list."
        self.shift_size = shift_size
        if not type(shift_size) == list and not type(shift_size) == tuple:
            self.shift_size = [self.shift_size, ] * self.dim
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.window_size = window_size
        pmw = [p - w for w, p in zip(self.window_size, self.patch_image_size)]
        assert min(pmw) >= 0, f"patch image size {self.patch_image_size} can't be smaller than window size {self.window_size}"
        ### if patch image size == window size, we don't need to shift this dim
        self.shift_size = [(p > 0 ) * s for p, s in zip(pmw, self.shift_size)]
        self.window_shift = max(self.shift_size) > 0
        assert sum([ p % w for w, p in zip(self.window_size, self.patch_image_size)]) == 0, f"patch image size {self.patch_image_size} must be divisible by window size {self.window_size}"
        assert sum([0 <= ss < ws 
            for ss, ws in zip(self.shift_size, self.window_size)] ) == self.dim, "shift_size must in [0, window_size)"

        self.window_length = np.prod(self.window_size)
        self.atten = WindowAttentionBlock(
            in_channel, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, atten_dropout=atten_dropout, dropout=dropout)

        ### shift window for both 2D and 3D
        if self.window_shift:
            if self.dim == 2:
                # calculate attention mask for SW-MSA
                H, W = self.patch_image_size
                img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size[0]),
                            slice(-self.window_size[0], -self.shift_size[0]),
                            slice(-self.shift_size[0], None))
                w_slices = (slice(0, -self.window_size[1]),
                            slice(-self.window_size[1], -self.shift_size[1]),
                            slice(-self.shift_size[1], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
            else:
                # calculate attention mask for SW-MSA
                H, W, D = self.patch_image_size
                img_mask = torch.zeros((1, H, W, D, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size[0]),
                            slice(-self.window_size[0], -self.shift_size[0]),
                            slice(-self.shift_size[0], None))
                w_slices = (slice(0, -self.window_size[1]),
                            slice(-self.window_size[1], -self.shift_size[1]),
                            slice(-self.shift_size[1], None))
                d_slices = (slice(0, -self.window_size[2]),
                            slice(-self.window_size[2], -self.shift_size[2]),
                            slice(-self.shift_size[2], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        for d in d_slices:
                            img_mask[:, h, w, d, :] = cnt
                            cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_length)
            atten_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            atten_mask = atten_mask.masked_fill(
                atten_mask != 0, float(-100.0)).masked_fill(atten_mask == 0, float(0.0))
        else:
            atten_mask = None

        self.register_buffer("atten_mask", atten_mask)
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.hyper_bias = nn.Linear(time_channel, out_channel, bias=False)
            self.hyper_gate = nn.Linear(time_channel, out_channel)
    def forward(self, x, t = None):
        ### patches to shifted window
        shortcut = x
        x = self.norm1.normalize(x)
        space_dims = tuple(range(1, self.dim+1))
        # cyclic shift
        if self.window_shift:
            shifted_x = torch.roll(
                x, 
                shifts=tuple(-s for s in self.shift_size), 
                dims=space_dims)
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        
        # nW*B, window_length, C
        x_windows = x_windows.view(-1, self.window_length, self.in_channel)
        # W-MSA/SW-MSA
        # nW*B, window_length, C
        atten_windows = self.atten(x_windows, mask=self.atten_mask)
        ### windows to patches
        # merge windows
        atten_windows = atten_windows.view(*([-1, ] + self.window_size + [self.in_channel, ]))
        
        shifted_x = window_reverse(
            atten_windows, self.window_size, self.patch_image_size)  # B H' W' C
        # reverse cyclic shift
        if self.window_shift:
            x = torch.roll(
                shifted_x, 
                shifts = self.shift_size, 
                dims=space_dims)
        else:
            x = shifted_x
        x = shortcut + self.drop_path(x)
        x = self.dense(x) + self.drop_path(self.mlp(self.norm2.normalize(x)))
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            gate = expand_as(torch.sigmoid(self.hyper_gate(t)), x, channel_dim=-1)
            bias = expand_as(self.hyper_bias(t), x, channel_dim = -1)
            x = x * gate + bias
        return x
## embed time step into swin transformer and vison mamba
    
class DoubleSwinTransformerBlock(nn.Module):
    def __init__(self, patch_image_size, in_channel, 
                out_channel = None, time_channel = 0, 
                num_heads = 1, window_size=7, shift_size=0,
                mlp_hidden_ratio=[4.0,], qkv_bias=True, qk_scale=None, 
                dropout=0., atten_dropout=0., drop_path=0.,
                activation = 'silu', norm = 'layer', num_groups = 4, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters:{}".format(kwargs))
        shift_size_sum = shift_size
        if isinstance(shift_size, list) or isinstance(shift_size, tuple):
            shift_size = sum(shift_size)
        self.st1 = SwinTransformerBlock(patch_image_size = patch_image_size, in_channel = in_channel, 
                out_channel = out_channel, num_heads = num_heads, window_size=window_size, 
                shift_size=shift_size, mlp_hidden_ratio=mlp_hidden_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                dropout=dropout, atten_dropout=atten_dropout, drop_path=drop_path,
                activation=activation, norm = norm, num_groups = num_groups)
        self.st2 = SwinTransformerBlock(patch_image_size = patch_image_size, in_channel = out_channel, 
                out_channel = out_channel, num_heads = num_heads, window_size=window_size, 
                shift_size=min(window_size) // 2 if shift_size_sum == 0 else 0, 
                mlp_hidden_ratio=mlp_hidden_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                dropout=dropout, atten_dropout=atten_dropout, drop_path=drop_path,
                activation=activation, norm = norm, num_groups = num_groups)
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.time_emb = nn.Linear(time_channel, out_channel)
    def forward(self, x, t = None):
        x = self.st1(x)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            x = x + expand_as(self.time_emb(t), x, channel_dim=-1)
        x = self.st2(x)
        return x
    
class ResSwinTransformerBlock(nn.Module):
    def __init__(self, patch_image_size, in_channel, out_channel = None, time_channel = 0, 
                 num_heads = 1, window_size=7, shift_size=0,
                 mlp_hidden_ratio=[4.0,], qkv_bias=True, qk_scale=None, dropout=0., atten_dropout=0., drop_path=0.,
                 activation = 'silu', norm = 'layer', num_groups = 4, **kwargs):
        super().__init__()
        shift_size_sum = shift_size
        if isinstance(shift_size, list) or isinstance(shift_size, tuple):
            shift_size = sum(shift_size)
        self.st1 = SwinTransformerBlock(patch_image_size = patch_image_size, in_channel = in_channel, 
                out_channel = out_channel, num_heads = num_heads, window_size=window_size, 
                shift_size=shift_size, mlp_hidden_ratio=mlp_hidden_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                dropout=dropout, atten_dropout=atten_dropout, drop_path=drop_path,
                activation=activation, norm = norm, num_groups = num_groups)
        self.st2 = SwinTransformerBlock(patch_image_size = patch_image_size, in_channel = out_channel, 
                out_channel = out_channel, num_heads = num_heads, window_size=window_size, 
                shift_size=min(window_size) // 2 if shift_size_sum == 0 else 0, 
                mlp_hidden_ratio=mlp_hidden_ratio, 
                qkv_bias=qkv_bias, qk_scale=qk_scale, 
                dropout=dropout, atten_dropout=atten_dropout, drop_path=drop_path,
                activation=None, norm = norm, num_groups = num_groups)
        self.shortcut = nn.Linear(in_channel, out_channel) if not in_channel == out_channel else nn.Identity()
        self.act = get_act(activation)
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.time_emb = nn.Linear(time_channel, out_channel)

    def forward(self, x, t = None):
        h = self.st1(x)
        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            h = h + expand_as(self.time_emb(t), h, channel_dim=-1)
        h = self.st2(h)
        out = self.act(h + self.shortcut(x))
        return out
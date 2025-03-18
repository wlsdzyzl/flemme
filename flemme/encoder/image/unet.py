# encoder for 2D image
# it actually encode image to a feature map instead of a vector embedding
from flemme.config import module_config
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import *
from flemme.logger import get_logger
from .cnn import CNNEncoder, CNNDecoder


logger = get_logger('encoder.image.unet')
### UNet encoder
### Possible improvement: A more elegant implementation is that Unet encoder can be inherited from CNN encoder
class UNetEncoder(CNNEncoder):
    def __init__(self, image_size, image_channel = 3,  patch_channel = 32, patch_size = 2, down_channels = [64, 128], 
                 down_attens = [None, None], shape_scaling = [2, 2], middle_channels = [256, 256], middle_attens = [None, None],
                 kernel_size = 3, depthwise = False, time_channel = 0, dsample_function = 'conv', num_blocks = 2,
                 building_block='res_t', normalization = 'group', num_norm_groups = 8, cn_order = 'cn',
                 activation = 'relu', dropout = 0., num_heads = 1, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 abs_pos_embedding = False, channel_attention = None, **kwargs):
        '''
        ## check down channels and middle channels
        ## each value in down channels corresponds to a downsample layer and a convolution layer
        ## each value in middle channels corresponds to a convolution layer, without changing the input size.
        '''
        super().__init__(image_size = image_size, image_channel = image_channel, time_channel = time_channel,
            patch_channel = patch_channel, patch_size=patch_size, kernel_size = kernel_size, 
            down_channels = down_channels, down_attens = down_attens, 
            shape_scaling=shape_scaling, middle_channels = middle_channels,
            middle_attens = middle_attens, depthwise = depthwise, dsample_function = dsample_function,
            num_blocks = num_blocks, building_block = building_block, normalization = normalization,
            num_norm_groups = num_norm_groups, cn_order = cn_order, activation = activation, dropout = dropout,
            num_heads = num_heads, d_k = d_k, qkv_bias = qkv_bias, qk_scale = qk_scale, 
            abs_pos_embedding=abs_pos_embedding, atten_dropout = atten_dropout,
            z_count = 1, dense_channels = [], return_feature_list = True,
            channel_attention = channel_attention)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
    
class UNetDecoder(CNNDecoder):
    def __init__(self, image_size, image_channel = 3, in_channel = 256, time_channel = 0, patch_size = 2,
                 up_channels = [128, 64], up_attens = [None, None], shape_scaling = [2, 2],
                 final_channels = [], final_attens = [], depthwise = False, usample_function = 'conv', 
                 kernel_size = 3, building_block='res_t', normalization = 'group',
                 num_norm_groups = 8, cn_order = 'cn',
                 num_blocks = 2,
                 activation = 'relu', dropout = 0., 
                 num_heads = 1, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 return_feature_list = False, 
                 channel_attention = None, **kwargs):
        super().__init__(image_size = image_size, image_channel = image_channel, 
            in_channel = in_channel, time_channel = time_channel, patch_size=patch_size,
            up_channels = up_channels, up_attens = up_attens, 
            shape_scaling=shape_scaling, final_channels = final_channels, kernel_size = kernel_size, 
            final_attens = final_attens, depthwise = depthwise, usample_function = usample_function,
            num_blocks = num_blocks, building_block = building_block, normalization = normalization,
            num_norm_groups = num_norm_groups, cn_order = cn_order, activation = activation, dropout = dropout,
            num_heads = num_heads, d_k = d_k, qkv_bias = qkv_bias, qk_scale = qk_scale, 
            return_feature_list = return_feature_list, 
            atten_dropout = atten_dropout, dense_channels = [],
            channel_attention=channel_attention)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        up_channels = [in_channel, ] + up_channels
        self.up = nn.ModuleList([UpSamplingBlock(dim=self.dim, in_channel=up_channels[i], out_channel=up_channels[i + 1], 
                                                func=usample_function, scale_factor=shape_scaling[i]) for i in range(len(up_channels)-1)])
        self.u_conv = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        dim=self.dim, in_channel=up_channels[i+1] * 2, 
                                                        out_channel=up_channels[i+1], 
                                                        atten=up_attens[i]) for i in range(len(up_channels) - 1) ])
    def forward(self, x, t = None):
        x, x_features = x
        x_features = x_features[::-1]
        ## x, t
        res = []
        for i, (up, u_conv) in enumerate(zip(self.up, self.u_conv)):
            x = u_conv(torch.cat((x_features[i], up(x)), dim = 1), t)
            if hasattr(self, 'uca'):
                x = self.uca[i](x)
            res = res + [x,]
        for fid, f_conv in enumerate(self.final):
            x = f_conv(x, t)
            if hasattr(self, 'fca'):
                x = self.fca[fid](x)
        x = self.image_back_proj(x)
        if self.return_feature_list:
            return x, res
        return x
if module_config['transformer']:
    from .vit import ViTEncoder, ViTDecoder
    from .swin import SwinEncoder, SwinDecoder

    class ViTUNetEncoder(ViTEncoder):
        def __init__(self, image_size, image_channel = 3, 
                    patch_size = 2, patch_channel = 32,
                    building_block = 'vit',
                    time_channel = 0,
                    mlp_hidden_ratio=[4., ], qkv_bias=True, qk_scale=None, 
                    down_channels = [128, 256], middle_channels = [256, 256], 
                    down_num_heads = [3, 3], middle_num_heads = [3, 3],
                    dropout=0., atten_dropout=0., drop_path=0.1, 
                    normalization = 'group', num_norm_groups = 8, 
                    num_blocks = 2, activation = 'silu', 
                    abs_pos_embedding = False,
                    channel_attention = None, 
                    **kwargs):
            super().__init__(image_size = image_size, image_channel = image_channel,
                patch_size = patch_size, patch_channel = patch_channel, time_channel = time_channel,
                building_block = building_block, mlp_hidden_ratio = mlp_hidden_ratio, qkv_bias = qkv_bias,
                qk_scale = qk_scale, down_channels = down_channels, middle_channels = middle_channels,
                down_num_heads = down_num_heads, middle_num_heads = middle_num_heads,
                dropout = dropout, atten_dropout = atten_dropout, drop_path = drop_path,
                normalization = normalization, num_norm_groups = num_norm_groups, 
                num_blocks = num_blocks, activation = activation, 
                abs_pos_embedding = abs_pos_embedding, 
                z_count = 1, dense_channels = [], 
                return_feature_list = True,
                channel_attention = channel_attention)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))
    class ViTUNetDecoder(ViTDecoder):
        def __init__(self, image_size, image_channel = 3, 
                    patch_size = 2, in_channel = 64,
                    time_channel = 0,
                    building_block = 'vit',
                    mlp_hidden_ratio=[4., ], qkv_bias=True, qk_scale=None, 
                    up_channels = [128, 64], final_channels = [64, 64], 
                    up_num_heads = [3, 3], final_num_heads = [3, 3],
                    dropout=0., atten_dropout=0., drop_path=0.1, 
                    normalization = 'group', num_norm_groups = 8, 
                    num_blocks = 2, activation = 'silu', 
                    return_feature_list = False, 
                    channel_attention = None, **kwargs):
            super().__init__(image_size = image_size, image_channel = image_channel,
                patch_size = patch_size, in_channel = in_channel, time_channel = time_channel,
                building_block = building_block, mlp_hidden_ratio = mlp_hidden_ratio, qkv_bias = qkv_bias,
                qk_scale = qk_scale, up_channels = up_channels, final_channels = final_channels,
                up_num_heads = up_num_heads, final_num_heads = final_num_heads,
                dropout = dropout, atten_dropout = atten_dropout, drop_path = drop_path,
                normalization = normalization, num_norm_groups = num_norm_groups, 
                num_blocks = num_blocks, activation = activation, dense_channels = [],
                return_feature_list = return_feature_list,
                channel_attention = channel_attention)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))

            up_channels = [in_channel] + up_channels
            self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i] // 2,
                                                        norm = normalization, 
                                                        num_norm_groups = num_norm_groups) for i in range(self.u_depth)])
            ### building block
            self.u_trans = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            num_heads = up_num_heads[i],
                                                            in_channel = up_channels[i+1] * 2,
                                                            out_channel = up_channels[i+1],
                                                            kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.u_depth) ])
        def forward(self, x, t = None):
            
            x, x_features = x
            x_features = x_features[::-1]
            ## x, t
            res = []
            for i, (up, u_trans) in enumerate(zip(self.up, self.u_trans)):
                ### channel information 
                x = u_trans(torch.cat((x_features[i], up(x)), dim = -1), t)
                if hasattr(self, 'uca'):
                    x = self.uca[i](x)
                res = res + [x, ]
            for fid, f_trans in enumerate(self.final):
                x = f_trans(x, t)
                if hasattr(self, 'fca'):
                    x = self.fca[fid](x)
            x = self.patch_recov(x)
            if self.return_feature_list:
                return x, res
            return x
    class SwinUNetEncoder(SwinEncoder):
        def __init__(self, image_size, image_channel = 3, 
                    window_size = 8, time_channel = 0,
                    patch_size = 2, patch_channel = 32,
                    building_block = 'swin',
                    mlp_hidden_ratio=[4., ], qkv_bias=True, qk_scale=None, 
                    down_channels = [128, 256], middle_channels = [256, 256], 
                    down_num_heads = [3, 3], middle_num_heads = [3, 3],
                    dropout=0., atten_dropout=0., drop_path=0.1, 
                    normalization = 'group', num_norm_groups = 8, 
                    num_blocks = 2, activation = 'silu', 
                    abs_pos_embedding = False,
                    channel_attention = None,
                    **kwargs):
            super().__init__(image_size = image_size, image_channel = image_channel,
                window_size = window_size, time_channel = time_channel,
                patch_size = patch_size, patch_channel = patch_channel,
                building_block = building_block, mlp_hidden_ratio = mlp_hidden_ratio, qkv_bias = qkv_bias,
                qk_scale = qk_scale, down_channels = down_channels, middle_channels = middle_channels,
                down_num_heads = down_num_heads, middle_num_heads = middle_num_heads,
                dropout = dropout, atten_dropout = atten_dropout, drop_path = drop_path,
                normalization = normalization, num_norm_groups = num_norm_groups, 
                num_blocks = num_blocks, activation = activation, 
                abs_pos_embedding = abs_pos_embedding, 
                z_count = 1, dense_channels = [], return_feature_list = True,
                channel_attention = channel_attention)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))
    class SwinUNetDecoder(SwinDecoder):
        def __init__(self, image_size, image_channel = 3, 
                    window_size = 8, time_channel = 0,
                    patch_size = 2, in_channel = 64,
                    building_block = 'swin',
                    mlp_hidden_ratio=[4., ], qkv_bias=True, qk_scale=None, 
                    up_channels = [128, 64], final_channels = [64, 64], 
                    up_num_heads = [3, 3], final_num_heads = [3, 3],
                    dropout=0., atten_dropout=0., drop_path=0.1, 
                    normalization = 'group', num_norm_groups = 8, 
                    num_blocks = 2, activation = 'silu', 
                    return_feature_list = False, 
                    channel_attention = None, **kwargs):
            super().__init__(image_size = image_size, image_channel = image_channel,
                window_size = window_size, time_channel = time_channel,
                patch_size = patch_size, in_channel = in_channel,
                building_block = building_block, mlp_hidden_ratio = mlp_hidden_ratio, qkv_bias = qkv_bias,
                qk_scale = qk_scale, up_channels = up_channels, final_channels = final_channels,
                up_num_heads = up_num_heads, final_num_heads = final_num_heads,
                dropout = dropout, atten_dropout = atten_dropout, drop_path = drop_path,
                normalization = normalization, num_norm_groups = num_norm_groups, 
                num_blocks = num_blocks, activation = activation, dense_channels = [],
                return_feature_list = return_feature_list,
                channel_attention = channel_attention)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))

            up_channels = [in_channel] + up_channels
            self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i] // 2,
                                                        norm = normalization, 
                                                        num_norm_groups = num_norm_groups) for i in range(self.u_depth)])
            ### building block
            self.u_trans = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            num_heads = up_num_heads[i],
                                                            in_channel = up_channels[i+1] * 2,
                                                            out_channel = up_channels[i+1],
                                                            patch_image_size = self.patch_image_pyramid[i+1],
                                                            kwargs_list = {"shift_size":  [(ni % 2) * (min(self.window_size) // 2) for ni in range(self.num_blocks) ],
                                                                        "drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.u_depth) ])
        def forward(self, x, t = None):
            x, x_features = x
            x_features = x_features[::-1]
            ## x, t
            res = []
            for i, (up, u_trans) in enumerate(zip(self.up, self.u_trans)):
                ### channel information 
                x = u_trans(torch.cat((x_features[i], up(x)), dim = -1), t)
                if hasattr(self, 'uca'):
                    x = self.uca[i](x)
                res = res + [x, ]
            for fid, f_trans in enumerate(self.final):
                x = f_trans(x, t)
                if hasattr(self, 'fca'):
                    x = self.fca[fid](x)
            x = self.patch_recov(x)
            if self.return_feature_list:
                return x, res
            return x
if module_config['mamba']:
    from .vmamba import VMambaEncoder, VMambaDecoder
    class VMambaUNetEncoder(VMambaEncoder):
        def __init__(self, image_size, image_channel = 3, 
                    patch_size = 2, patch_channel = 32,
                    time_channel = 0,
                    down_channels = [128, 256], middle_channels = [256, 256], 
                    mlp_hidden_ratio=[4., ], state_channel=None, 
                    building_block = 'vmamba',
                    conv_kernel_size=3,
                    inner_factor = 2.0,
                    dt_rank=None, dt_min=0.001, 
                    dt_max=0.1, dt_init="random", dt_scale=1.0, 
                    dt_init_floor=1e-4, 
                    conv_bias=True, bias=False,             
                    dropout=0., drop_path=0.1, 
                    normalization = 'group', num_norm_groups = 8, 
                    num_blocks = 2, activation = 'silu', 
                    scan_mode = 'single', flip_scan = True,
                    abs_pos_embedding = False,
                    head_channel = 64, 
                    learnable_init_states = True, 
                    chunk_size=256,
                    channel_attention = None,
                    **kwargs):
            super().__init__(image_size, image_channel = image_channel, 
                    time_channel = time_channel,
                    patch_size = patch_size, patch_channel = patch_channel,
                    mlp_hidden_ratio=mlp_hidden_ratio,
                    down_channels = down_channels, middle_channels = middle_channels, 
                    state_channel=state_channel, building_block = building_block,
                    conv_kernel_size=conv_kernel_size,
                    inner_factor = inner_factor,   
                    dt_rank=dt_rank, dt_min=dt_min, 
                    dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, 
                    dt_init_floor=dt_init_floor, 
                    conv_bias=conv_bias, bias=bias,             
                    dropout=dropout, drop_path=drop_path, 
                    head_channel = head_channel, 
                    learnable_init_states = learnable_init_states, 
                    chunk_size=chunk_size,
                    normalization = normalization, num_norm_groups = num_norm_groups, 
                    num_blocks = num_blocks, activation = activation, 
                    scan_mode = scan_mode, flip_scan=flip_scan, 
                    abs_pos_embedding = abs_pos_embedding,
                    z_count = 1, dense_channels = [],
                    return_feature_list = True,
                    channel_attention = channel_attention)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))
    class VMambaUNetDecoder(VMambaDecoder):
        def __init__(self, image_size, image_channel = 3, 
                    patch_size = 2, in_channel = 64,
                    time_channel = 0,
                    mlp_hidden_ratio=[4., ],
                    up_channels = [128, 64], final_channels = [64, 64], 
                    state_channel=None, 
                    building_block = 'vmamba',
                    conv_kernel_size=3,
                    inner_factor = 2.0,  
                    dt_rank=None, dt_min=0.001, 
                    dt_max=0.1, dt_init="random", dt_scale=1.0, 
                    dt_init_floor=1e-4, 
                    conv_bias=True, bias=False,             
                    dropout=0., drop_path=0.1, 
                    head_channel = 64, 
                    learnable_init_states = True, 
                    chunk_size=256,
                    normalization = 'group', num_norm_groups = 8, 
                    num_blocks = 2, activation = 'silu', 
                    scan_mode = 'single', flip_scan = True, 
                    return_feature_list = False, 
                    channel_attention = None, 
                    **kwargs):
            super().__init__(image_size, image_channel = image_channel, 
                    patch_size = patch_size, in_channel = in_channel,
                    time_channel = time_channel,
                    mlp_hidden_ratio=mlp_hidden_ratio,
                    up_channels = up_channels, final_channels = final_channels, 
                    state_channel=state_channel, building_block = building_block,
                    conv_kernel_size=conv_kernel_size,
                    inner_factor = inner_factor,   
                    dt_rank=dt_rank, dt_min=dt_min, 
                    dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, 
                    dt_init_floor=dt_init_floor, 
                    conv_bias=conv_bias, bias=bias,             
                    dropout=dropout, drop_path=drop_path, 
                    head_channel = head_channel, 
                    learnable_init_states = learnable_init_states, 
                    chunk_size=chunk_size,
                    normalization = normalization, num_norm_groups = num_norm_groups, 
                    num_blocks = num_blocks, activation = activation, 
                    scan_mode = scan_mode, flip_scan=flip_scan, 
                    dense_channels = [], return_feature_list = return_feature_list,
                    channel_attention = channel_attention)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))
            ### use patch expansion block for up sampling
            up_channels = [in_channel] + up_channels
            self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i] // 2,
                                                        norm = normalization, 
                                                        num_norm_groups = num_norm_groups) for i in range(self.u_depth)])
            ### building block
            self.u_ssm = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            in_channel = up_channels[i+1]*2,
                                                            out_channel = up_channels[i+1],
                                                            kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.u_depth) ])

        def forward(self, x, t = None):
            x, x_features = x
            x_features = x_features[::-1]
            ## x, t
            res = []
            for i, (up, u_ssm) in enumerate(zip(self.up, self.u_ssm)):
                ### channel information 
                x = u_ssm(torch.cat((x_features[i], up(x)), dim = -1), t)
                if hasattr(self, 'uca'):
                    x = self.uca[i](x)
                res = res + [x,]
            for fid, f_ssm in enumerate(self.final):
                x = f_ssm(x, t)
                if hasattr(self, 'fca'):
                    x = self.fca[fid](x)
            x = self.patch_recov(x)
            if self.return_feature_list:
                return x, res
            return x
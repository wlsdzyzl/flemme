# encoder for 2D image
# it actually encode image to a feature map instead of a vector embedding
from flemme.config import module_config
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import *
from flemme.logger import get_logger
from .cnn import CNNEncoder, CNNDecoder


logger = get_logger('encoder.image.dnet')
### DenseNet encoder
class DNetEncoder(CNNEncoder):
    def __init__(self, image_size, image_channel = 3,  patch_channel = 32, patch_size = 2, down_channels = [64, 128], 
                    down_attens = [None, None], shape_scaling = [2, 2], middle_channels = [256, 256], middle_attens = [None, None],
                    kernel_size = 3, depthwise = False, time_channel = 0, dsample_function = 'conv', num_blocks = 2,
                    building_block='res_t', normalization = 'group', num_norm_groups = 8, cn_order = 'cn',
                    activation = 'relu', dropout = 0., num_heads = 1, d_k = None, 
                    qkv_bias = True, qk_scale = None, atten_dropout = None, 
                    abs_pos_embedding = False, 
                    channel_attention = None, 
                    time_injection = 'gate_bias', 
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    condition_first = False,
                    **kwargs):
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
            channel_attention = channel_attention,
            time_injection = time_injection,
            condition_channel = condition_channel,
            condition_injection=condition_injection,
            condition_first = condition_first)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        down_channels = [self.image_patch_channel, ] + down_channels

        self.d_conv = nn.ModuleList( [MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        dim=self.dim, in_channel=down_channels[i] if i < 2 else sum(down_channels[1:i+1]), 
                                                        out_channel=down_channels[i+1], 
                                                        atten=down_attens[i]) for i in range(len(down_channels) - 1) ])

        middle_channels = [sum(down_channels[1:]) if len(down_channels) >= 2 else down_channels[-1], ] + middle_channels 
        module_sequence = [MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                          dim=self.dim, in_channel=middle_channels[i], 
                                          out_channel=middle_channels[i+1], 
                                          atten = middle_attens[i]) for i in range(len(middle_channels) - 1)]
        
        self.middle = nn.ModuleList(module_sequence)

        ## down sample layers for dense connections
        self.down_for_dense = nn.ModuleList([
          nn.ModuleList( [  DownSamplingBlock(dim=self.dim, in_channel=down_channels[j], 
                                func=dsample_function, scale_factor=np.prod(shape_scaling[j-1:i]) ) for j in range(1, i)])
        for i in range(2, len(down_channels) - 1) ] )
        ### for middle layers
        self.down_for_dense.append(nn.ModuleList( [  DownSamplingBlock(dim=self.dim, in_channel=down_channels[i], 
                                func=dsample_function, scale_factor=np.prod(shape_scaling[i-1:]) ) for i in range(1, len(down_channels) - 1)]))
        

    def forward(self, x, t = None, c = None):
        x = self.image_proj(x)
        res = []
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed

        for i, (d_conv, down) in enumerate(zip(self.d_conv, self.down)):
            concat_feature = []
            if i >= 2:
                for r, dd in zip(res[:-1], self.down_for_dense[i - 2]):
                    concat_feature.append(dd(r))
                x = torch.cat(concat_feature + [x], dim=1)
            x = d_conv(x, t, c)
            if hasattr(self, 'dca'):
                x = self.dca[i](x)

            res = res + [x,]
            x = down(x)
        
        concat_feature = [dd(r) for r, dd in zip(res[:-1], self.down_for_dense[-1])]
        x = torch.cat(concat_feature + [x], dim=1)
        for mid, m_conv in enumerate(self.middle):
            x = m_conv(x, t, c)
            if hasattr(self, 'mca'):
                x = self.mca[mid](x)

        
        if self.return_feature_list:
            return x, res
        return x
    
class DNetDecoder(CNNDecoder):
    def __init__(self, image_size, image_channel = 3, in_channel = 256, time_channel = 0, patch_size = 2,
                 up_channels = [128, 64], up_attens = [None, None], shape_scaling = [2, 2],
                 final_channels = [], final_attens = [], depthwise = False, 
                 usample_function = 'conv', dsample_function = 'conv',
                 kernel_size = 3, building_block='res_t', normalization = 'group',
                 num_norm_groups = 8, cn_order = 'cn',
                 num_blocks = 2,
                 activation = 'relu', dropout = 0., 
                 num_heads = 1, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 return_feature_list = False, 
                 channel_attention = None, 
                 time_injection = 'gate_bias', 
                 condition_channel = 0,
                 condition_injection = 'gate_bias',
                 condition_first = False,
                 **kwargs):
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
            channel_attention = channel_attention,
            time_injection = time_injection,
            condition_channel = condition_channel,
            condition_injection = condition_injection,
            condition_first = condition_first)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        
        down_channels = up_channels[::-1]
        down_shape_scaling = shape_scaling[::-1]
        up_channels = [in_channel, ] + up_channels
        self.up = nn.ModuleList([UpSamplingBlock(dim=self.dim, in_channel=up_channels[i], out_channel=up_channels[i + 1], 
                                                func=usample_function, scale_factor=shape_scaling[i]) for i in range(len(up_channels)-1)])
        self.u_conv = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        dim=self.dim, in_channel=(len(down_channels) + i + 1) * up_channels[i+1]  , 
                                                        out_channel=up_channels[i+1], 
                                                        atten=up_attens[i]) for i in range(len(up_channels) - 1) ])

        ## down / up sample layers for dense connections
        self.sample_for_dense = nn.ModuleList([])
        for i in range(0, len(up_channels) - 1):
            tmp_module = []
            ## features from encoder
            for j in range(0, len(down_channels)):
                up_shape_scaling = np.prod(shape_scaling[0:i+1]) / np.prod(down_shape_scaling[j:])
                ## should be down sampling
                if up_shape_scaling < 1:
                    tmp_module.append(DownSamplingBlock(dim=self.dim, in_channel=down_channels[j], 
                                    out_channel = up_channels[i + 1],
                                    func=dsample_function, 
                                    scale_factor=int(np.prod(down_shape_scaling[j:]) / np.prod(shape_scaling[0:i+1]))) )
                else:
                    tmp_module.append(UpSamplingBlock(dim=self.dim, in_channel=down_channels[j], 
                                        out_channel=up_channels[i + 1], 
                                        func=usample_function, 
                                        scale_factor=int(up_shape_scaling)))
            ## features from decoder
            for j in range(0, i):
                tmp_module.append(UpSamplingBlock(dim=self.dim, 
                                        in_channel=up_channels[j], 
                                        out_channel=up_channels[i + 1], 
                                        func=usample_function, 
                                        scale_factor=np.prod(shape_scaling[j:i+1])))
            self.sample_for_dense.append(nn.ModuleList(tmp_module))

    def forward(self, x, t = None, c = None):
        ## existing features but without x
        x, existing_features = x
        ## x, t
        res = []
        for i, (up, u_conv) in enumerate(zip(self.up, self.u_conv)):
            concat_features = []
            for tf, dus in zip(existing_features, self.sample_for_dense[i]):
                concat_features.append(dus(tf))
            concat_features.append(up(x))
            existing_features = existing_features + [x,]
            x = u_conv(torch.cat(concat_features, dim = 1), t, c)
            if hasattr(self, 'uca'):
                x = self.uca[i](x)
            res = res + [x,]
        for fid, f_conv in enumerate(self.final):
            x = f_conv(x, t, c)
            if hasattr(self, 'fca'):
                x = self.fca[fid](x)

        x = self.image_back_proj(x)
        if self.return_feature_list:
            return x, res
        return x

if module_config['transformer']:
    from .vit import ViTEncoder, ViTDecoder
    from .swin import SwinEncoder, SwinDecoder

    class ViTDNetEncoder(ViTEncoder):
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
                    time_injection = 'gate_bias', 
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    condition_first = False,
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
                z_count = 1, dense_channels = [], return_feature_list = True,
                channel_attention = channel_attention,
                time_injection = time_injection,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = condition_first)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))

            down_channels = [self.patch_channel, ] + down_channels
            self.d_trans = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            num_heads = down_num_heads[i],
                                                            in_channel=down_channels[i] if i < 2 else sum(down_channels[1:i+1]), 
                                                            out_channel = down_channels[i+1],
                                                            kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.d_depth) ])


            middle_channels = [sum(down_channels[1:]) if len(down_channels) >= 2 else down_channels[-1], ] + middle_channels 
            self.middle = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            num_heads = middle_num_heads[i],
                                                            in_channel = middle_channels[i],
                                                            out_channel = middle_channels[i+1],
                                                            kwargs_list = {"drop_path": [self.drop_path[(i + self.d_depth)*self.num_blocks + ni] 
                                                                                        for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.m_depth) ])
                

            ## down sample layers for dense connections
            self.down_for_dense = nn.ModuleList([
            nn.ModuleList( [  PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[j],
                                    out_channel=down_channels[j],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2 ** (i - j + 1) ) for j in range(1, i)])
            for i in range(2, len(down_channels) - 1) ] )
            ### for middle layers
            self.down_for_dense.append(nn.ModuleList( [  
                                    PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[i],
                                    out_channel=down_channels[i],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2 ** (self.d_depth - i + 1) ) for i in range(1, len(down_channels) - 1)]))

        def forward(self, x, t = None, c = None):

            x = self.patch_emb(x)
            res = []
            if self.absolute_pos_embed is not None:
                x = x + self.absolute_pos_embed
            for i, (d_trans, down) in enumerate(zip(self.d_trans, self.down)):
                concat_feature = []
                if i >= 2:
                    for r, dd in zip(res[:-1], self.down_for_dense[i - 2]):
                        concat_feature.append(dd(r))
                    x = torch.cat(concat_feature + [x], dim=-1)
                x = d_trans(x, t, c)
                if hasattr(self, 'dca'):
                    x = self.dca[i](x)

                res = res + [x, ]
                x = down(x)
            concat_feature = [dd(r) for r, dd in zip(res[:-1], self.down_for_dense[-1])]
            x = torch.cat(concat_feature + [x], dim=-1)

            for mid, m_trans in enumerate(self.middle):
                x = m_trans(x, t, c)
                if hasattr(self, 'mca'):
                    x = self.mca[mid](x)


            if self.return_feature_list:
                return x, res
            return x

    class ViTDNetDecoder(ViTDecoder):
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
                    channel_attention = None, 
                    time_injection = 'gate_bias', 
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    condition_first = False,
                    **kwargs):
            super().__init__(image_size = image_size, image_channel = image_channel,
                patch_size = patch_size, in_channel = in_channel, time_channel = time_channel,
                building_block = building_block, mlp_hidden_ratio = mlp_hidden_ratio, qkv_bias = qkv_bias,
                qk_scale = qk_scale, up_channels = up_channels, final_channels = final_channels,
                up_num_heads = up_num_heads, final_num_heads = final_num_heads,
                dropout = dropout, atten_dropout = atten_dropout, drop_path = drop_path,
                normalization = normalization, num_norm_groups = num_norm_groups, 
                num_blocks = num_blocks, activation = activation, dense_channels = [],
                return_feature_list = return_feature_list,
                channel_attention = channel_attention,
                time_injection = time_injection,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = condition_first)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))

            down_channels = up_channels[::-1]
            up_channels = [in_channel] + up_channels
            self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i + 1], 
                                                        norm = normalization, 
                                                        num_norm_groups = num_norm_groups) for i in range(self.u_depth)])
                
            self.u_trans = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            num_heads = up_num_heads[i],
                                                            in_channel=(len(down_channels) + i + 1) * up_channels[i+1], 
                                                            out_channel=up_channels[i+1], 
                                                            kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.u_depth) ])

            ## down / up sample layers for dense connections
            self.sample_for_dense = nn.ModuleList([])
            for i in range(0, len(up_channels) - 1):
                tmp_module = []
                ## features from encoder
                for j in range(0, len(down_channels)):
                    up_times = i+1 - self.u_depth + j
                    if up_times < 0:
                        tmp_module.append(PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[j],
                                    out_channel=up_channels[i + 1],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2**(-up_times)))
                    else:
                        tmp_module.append(PatchExpansionBlock(dim = self.dim,
                                            in_channel=down_channels[j],
                                            out_channel=up_channels[i + 1], 
                                            norm = normalization, 
                                            num_norm_groups = num_norm_groups,
                                            factor=2**up_times))
                ## features from decoder
                for j in range(0, i):
                    tmp_module.append(PatchExpansionBlock(dim = self.dim,
                                            in_channel=up_channels[j],
                                            out_channel=up_channels[i + 1], 
                                            norm = normalization, 
                                            num_norm_groups = num_norm_groups,
                                            factor=2**(i+1-j)))
                self.sample_for_dense.append(nn.ModuleList(tmp_module))

        def forward(self, x, t = None, c = None):
            ## existing features but without x
            x, existing_features = x
            ## x, t
            res = []
            for i, (up, u_trans) in enumerate(zip(self.up, self.u_trans)):
                concat_features = []
                for tf, dus in zip(existing_features, self.sample_for_dense[i]):
                    concat_features.append(dus(tf))
                concat_features.append(up(x))
                existing_features = existing_features + [x,]
                x = u_trans(torch.cat(concat_features, dim = -1), t, c)
                if hasattr(self, 'uca'):
                    x = self.uca[i](x)

                res = res + [x,]
            for fid, f_trans in enumerate(self.final):
                x = f_trans(x, t, c)
                if hasattr(self, 'fca'):
                    x = self.fca[fid](x)

            x = self.patch_recov(x)
            if self.return_feature_list:
                return x, res
            return x 

    class SwinDNetEncoder(SwinEncoder):
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
                    time_injection = 'gate_bias', 
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    condition_first = False,
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
                channel_attention = channel_attention,
                time_injection = time_injection,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = condition_first)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))
            down_channels = [self.patch_channel, ] + down_channels
            self.d_trans = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            num_heads = down_num_heads[i],
                                                            in_channel=down_channels[i] if i < 2 else sum(down_channels[1:i+1]), 
                                                            out_channel = down_channels[i+1],
                                                            patch_image_size = self.patch_image_pyramid[i],
                                                            kwargs_list = {"shift_size":  [(ni % 2) * (min(self.window_size) // 2) for ni in range(self.num_blocks) ],
                                                                        "drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.d_depth) ])


            middle_channels = [sum(down_channels[1:]) if len(down_channels) >= 2 else down_channels[-1], ] + middle_channels 
            self.middle = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            num_heads = middle_num_heads[i],
                                                            in_channel = middle_channels[i],
                                                            out_channel = middle_channels[i+1],
                                                            patch_image_size = self.patch_image_pyramid[-1],
                                                            kwargs_list = {"shift_size":  [(ni % 2) * (min(self.window_size) // 2) for ni in range(self.num_blocks) ],
                                                                        "drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.m_depth) ])
                

            ## down sample layers for dense connections
            self.down_for_dense = nn.ModuleList([
            nn.ModuleList( [  PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[j],
                                    out_channel=down_channels[j],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2 ** (i - j + 1) ) for j in range(1, i)])
            for i in range(2, len(down_channels) - 1) ] )
            ### for middle layers
            self.down_for_dense.append(nn.ModuleList( [  
                                    PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[i],
                                    out_channel=down_channels[i],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2 ** (self.d_depth - i + 1) ) for i in range(1, len(down_channels) - 1)]))

        def forward(self, x, t = None, c = None):

            x = self.patch_emb(x)
            res = []
            if self.absolute_pos_embed is not None:
                x = x + self.absolute_pos_embed
            for i, (d_trans, down) in enumerate(zip(self.d_trans, self.down)):
                concat_feature = []
                if i >= 2:
                    for r, dd in zip(res[:-1], self.down_for_dense[i - 2]):
                        concat_feature.append(dd(r))
                    x = torch.cat(concat_feature + [x], dim=-1)
                x = d_trans(x, t, c)
                if hasattr(self, 'dca'):
                    x = self.dca[i](x)

                res = res + [x, ]
                x = down(x)
            concat_feature = [dd(r) for r, dd in zip(res[:-1], self.down_for_dense[-1])]
            x = torch.cat(concat_feature + [x], dim=-1)
            for mid, m_trans in enumerate(self.middle):
                x = m_trans(x, t, c)
                if hasattr(self, 'mca'):
                    x = self.mca[mid](x)

            if self.return_feature_list:
                return x, res
            return x
    class SwinDNetDecoder(SwinDecoder):
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
                    channel_attention = None, 
                    time_injection = 'gate_bias', 
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    condition_first = False,
                    **kwargs):
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
                channel_attention=channel_attention,
                time_injection=time_injection,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = condition_first)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))

            down_channels = up_channels[::-1]
            up_channels = [in_channel] + up_channels
            self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i + 1], 
                                                        norm = normalization, 
                                                        num_norm_groups = num_norm_groups) for i in range(self.u_depth)])
                
            self.u_trans = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            num_heads = up_num_heads[i],
                                                            in_channel=(len(down_channels) + i + 1) * up_channels[i+1], 
                                                            out_channel=up_channels[i+1], 
                                                            patch_image_size = self.patch_image_pyramid[i+1],
                                                            kwargs_list = {"shift_size":  [(ni % 2) * (min(self.window_size) // 2) for ni in range(self.num_blocks) ],
                                                                        "drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.u_depth) ])

            ## down / up sample layers for dense connections
            self.sample_for_dense = nn.ModuleList([])
            for i in range(0, len(up_channels) - 1):
                tmp_module = []
                ## features from encoder
                for j in range(0, len(down_channels)):
                    up_times = i+1 - self.u_depth + j
                    if up_times < 0:
                        tmp_module.append(PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[j],
                                    out_channel=up_channels[i + 1],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2**(-up_times)))
                    else:
                        tmp_module.append(PatchExpansionBlock(dim = self.dim,
                                            in_channel=down_channels[j],
                                            out_channel=up_channels[i + 1], 
                                            norm = normalization, 
                                            num_norm_groups = num_norm_groups,
                                            factor=2**up_times))
                ## features from decoder
                for j in range(0, i):
                    tmp_module.append(PatchExpansionBlock(dim = self.dim,
                                            in_channel=up_channels[j],
                                            out_channel=up_channels[i + 1], 
                                            norm = normalization, 
                                            num_norm_groups = num_norm_groups,
                                            factor=2**(i+1-j)))
                self.sample_for_dense.append(nn.ModuleList(tmp_module))

        def forward(self, x, t = None, c = None):
            ## existing features but without x
            x, existing_features = x
            ## x, t
            res = []
            for i, (up, u_trans) in enumerate(zip(self.up, self.u_trans)):
                concat_features = []
                for tf, dus in zip(existing_features, self.sample_for_dense[i]):
                    concat_features.append(dus(tf))
                concat_features.append(up(x))
                existing_features = existing_features + [x,]
                x = u_trans(torch.cat(concat_features, dim = -1), t, c)
                if hasattr(self, 'uca'):
                    x = self.uca[i](x)

                res = res + [x,]
            for fid, f_trans in enumerate(self.final):
                x = f_trans(x, t, c)
                if hasattr(self, 'fca'):
                    x = self.fca[fid](x)

            x = self.patch_recov(x)
            if self.return_feature_list:
                return x, res
            return x 

if module_config['mamba']:
    from .vmamba import VMambaEncoder, VMambaDecoder
    class VMambaDNetEncoder(VMambaEncoder):
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
                    time_injection = 'gate_bias', 
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    condition_first = False,
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
                    channel_attention=channel_attention,
                    time_injection=time_injection,
                    condition_channel = condition_channel,
                    condition_injection = condition_injection,
                    condition_first = condition_first)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))

            down_channels = [self.patch_channel, ] + down_channels
            self.d_ssm = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            in_channel=down_channels[i] if i < 2 else sum(down_channels[1:i+1]), 
                                                            out_channel = down_channels[i+1],
                                                            kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.d_depth) ])


            middle_channels = [sum(down_channels[1:]) if len(down_channels) >= 2 else down_channels[-1], ] + middle_channels 
            self.middle = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            in_channel = middle_channels[i],
                                                            out_channel = middle_channels[i+1],
                                                            kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.m_depth) ])
                

            ## down sample layers for dense connections
            self.down_for_dense = nn.ModuleList([
            nn.ModuleList( [  PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[j],
                                    out_channel=down_channels[j],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2 ** (i - j + 1) ) for j in range(1, i)])
            for i in range(2, len(down_channels) - 1) ] )
            ### for middle layers
            self.down_for_dense.append(nn.ModuleList( [  
                                    PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[i],
                                    out_channel=down_channels[i],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2 ** (self.d_depth - i + 1) ) for i in range(1, len(down_channels) - 1)]))

        def forward(self, x, t = None, c = None):

            x = self.patch_emb(x)
            res = []
            if self.absolute_pos_embed is not None:
                x = x + self.absolute_pos_embed
            for i, (d_ssm, down) in enumerate(zip(self.d_ssm, self.down)):
                concat_feature = []
                if i >= 2:
                    for r, dd in zip(res[:-1], self.down_for_dense[i - 2]):
                        concat_feature.append(dd(r))
                    x = torch.cat(concat_feature + [x], dim=-1)
                x = d_ssm(x, t, c)
                if hasattr(self, 'dca'):
                    x = self.dca[i](x)

                res = res + [x, ]
                x = down(x)
            concat_feature = [dd(r) for r, dd in zip(res[:-1], self.down_for_dense[-1])]
            x = torch.cat(concat_feature + [x], dim=-1)
            for mid, m_ssm in enumerate(self.middle):
                x = m_ssm(x, t, c)
                if hasattr(self, 'mca'):
                    x = self.mca[mid](x)


            if self.return_feature_list:
                return x, res
            return x

    class VMambaDNetDecoder(VMambaDecoder):
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
                    time_injection = 'gate_bias', 
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    condition_first = False,
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
                    channel_attention=channel_attention,
                    time_injection=time_injection,
                    condition_channel = condition_channel,
                    condition_injection = condition_injection,
                    condition_first = condition_first)
            if len(kwargs) > 0:
                logger.debug("redundant parameters:{}".format(kwargs))

            down_channels = up_channels[::-1]
            up_channels = [in_channel] + up_channels
            self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i + 1], 
                                                        norm = normalization, 
                                                        num_norm_groups = num_norm_groups) for i in range(self.u_depth)])

            self.u_ssm = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                            in_channel=(len(down_channels) + i + 1) * up_channels[i+1], 
                                                            out_channel=up_channels[i+1], 
                                                            kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                            ) for i in range(self.u_depth) ])

            ## down / up sample layers for dense connections
            self.sample_for_dense = nn.ModuleList([])
            for i in range(0, len(up_channels) - 1):
                tmp_module = []
                ## features from encoder
                for j in range(0, len(down_channels)):
                    up_times = i+1 - self.u_depth + j
                    if up_times < 0:
                        tmp_module.append(PatchMergingBlock(dim=self.dim, 
                                    in_channel=down_channels[j],
                                    out_channel=up_channels[i + 1],
                                    norm = normalization,
                                    num_norm_groups = num_norm_groups,
                                    factor=2**(-up_times)))
                    else:
                        tmp_module.append(PatchExpansionBlock(dim = self.dim,
                                            in_channel=down_channels[j],
                                            out_channel=up_channels[i + 1], 
                                            norm = normalization, 
                                            num_norm_groups = num_norm_groups,
                                            factor=2**up_times))
                ## features from decoderup
                for j in range(0, i):
                    tmp_module.append(PatchExpansionBlock(dim = self.dim,
                                            in_channel=up_channels[j],
                                            out_channel=up_channels[i + 1], 
                                            norm = normalization, 
                                            num_norm_groups = num_norm_groups,
                                            factor=2**(i+1-j)))
                self.sample_for_dense.append(nn.ModuleList(tmp_module))

        def forward(self, x, t = None, c = None):
            ## existing features but without x
            x, existing_features = x
            ## x, t
            res = []
            for i, (up, u_ssm) in enumerate(zip(self.up, self.u_ssm)):
                concat_features = []
                for tf, dus in zip(existing_features, self.sample_for_dense[i]):
                    concat_features.append(dus(tf))
                concat_features.append(up(x))
                existing_features = existing_features + [x,]
                x = u_ssm(torch.cat(concat_features, dim = -1), t, c)
                if hasattr(self, 'uca'):
                    x = self.uca[i](x)

                res = res + [x,]
            for fid, f_ssm in enumerate(self.final):
                x = f_ssm(x, t, c)
                if hasattr(self, 'fca'):
                    x = self.fca[fid](x)

            x = self.patch_recov(x)
            if self.return_feature_list:
                return x, res
            return x 
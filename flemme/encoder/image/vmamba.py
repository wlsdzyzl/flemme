### mamba-encoder: window and patch based vision SSM
import torch
import torch.nn.functional as F
from torch import nn
import math
from flemme.block import PatchConstructionBlock, PatchRecoveryBlock,\
    PatchExpansionBlock, PatchMergingBlock, get_ca, \
    get_building_block, MultipleBuildingBlocks, DenseBlock
from flemme.logger import get_logger
import copy
logger = get_logger("encoder.image.vmamba")

class VMambaEncoder(nn.Module):
    def __init__(self, image_size, image_channel = 3, 
                patch_size = 2, patch_channel = 32,
                time_channel = 0,
                down_channels = [128, 256], middle_channels = [256, 256], 
                mlp_hidden_ratio=[4., ], state_channel=None, 
                building_block = 'vmamba', dense_channels = [256], 
                conv_kernel_size=3,
                inner_factor = 2.0,
                dt_rank=None, dt_min=0.001, 
                dt_max=0.1, dt_init="random", dt_scale=1.0, 
                dt_init_floor=1e-4, 
                conv_bias=True, bias=False,             
                dropout=0., drop_path=0.1, 
                normalization = 'group', num_norm_groups = 8, 
                num_blocks = 2, activation = 'silu', 
                scan_mode = 'single',
                flip_scan = True,
                head_channel = 64, 
                learnable_init_states = True, 
                chunk_size=256,
                abs_pos_embedding = False,
                last_activation = True,
                return_feature_list = False,
                z_count = 1, 
                channel_attention = None, 
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.image_size = image_size
        self.image_channel = image_channel
        self.dim = len(image_size)
        self.z_count = z_count
        self.d_depth = len(down_channels)
        self.m_depth = len(middle_channels)
        self.vector_embedding = isinstance(dense_channels, list) and len(dense_channels) > 0

        self.activation = activation
        self.dropout = dropout
        self.num_blocks = num_blocks
        # stochastic depth decay rule
        self.drop_path = [x.item() for x in torch.linspace(0, drop_path,
                                                (self.d_depth + self.m_depth) * self.num_blocks)]
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.patch_size = patch_size
        self.patch_channel = patch_channel

        assert 'vmamba' in building_block, 'VMambaEncoder only support mamba-related building blocks.'
        ### building block: mamba SSM block
        self.BuildingBlock = get_building_block(building_block, dim = self.dim,
                                time_channel = time_channel,
                                mlp_hidden_ratio = mlp_hidden_ratio,
                                inner_factor = inner_factor, 
                                conv_kernel_size = conv_kernel_size,
                                dt_min = dt_min, dt_max = dt_max,
                                dt_init_floor = dt_init_floor,
                                dt_rank = dt_rank,
                                dt_scale = dt_scale,
                                dt_init = dt_init,
                                conv_bias = conv_bias, bias = bias,
                                dropout = dropout,
                                norm = normalization, num_norm_groups = num_norm_groups,
                                state_channel = state_channel,
                                head_channel = head_channel, 
                                learnable_init_states = learnable_init_states, 
                                chunk_size=chunk_size,
                                activation = activation, 
                                scan_mode = scan_mode,
                                flip_scan = flip_scan)
        ### construct patch
        self.patch_emb = PatchConstructionBlock(dim = self.dim, 
                                                patch_size = self.patch_size,
                                                in_channel = self.image_channel,
                                                out_channel = self.patch_channel,
                                                norm = None)
        self.absolute_pos_embed = None
        if abs_pos_embedding:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros([1,] + [s // self.patch_size for s in self.image_size] + [self.patch_channel,]))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        ### use patch merging block for down sampling
        down_channels = [self.patch_channel, ] + down_channels
        ### building block
        self.d_ssm = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        in_channel = down_channels[i],
                                                        out_channel = down_channels[i+1],
                                                        kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] 
                                                                                     for ni in range(self.num_blocks) ] }
                                                        ) for i in range(self.d_depth) ])
        self.down = nn.ModuleList([PatchMergingBlock(dim = self.dim,
                                                     in_channel = down_channels[i+1],
                                                     out_channel = down_channels[i+1],
                                                     norm = normalization, 
                                                     num_norm_groups = num_norm_groups)  for i in range(self.d_depth)])
        self.down_path = [self.image_channel, ] + down_channels        
        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = self.dim, channel = down_channels[i+1], channel_dim = -1, **channel_attention) 
                           for i in range(len(down_channels) - 1)]
            self.dca = nn.ModuleList(ca_sequence)        
        ## middle layer
        dense_channels = [ middle_channels[-1], ] + dense_channels
        self.dense_path = dense_channels.copy()
        self.middle_path = [down_channels[-1], ] + middle_channels 
        if not self.vector_embedding:
            middle_channels = [mc * self.z_count for mc in middle_channels]
        ### middle SSM
        middle_channels = [down_channels[-1], ] + middle_channels
        self.middle = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        in_channel = middle_channels[i],
                                                        out_channel = middle_channels[i+1],
                                                        kwargs_list = {"drop_path": [self.drop_path[(i + self.d_depth)*self.num_blocks + ni] 
                                                                                     for ni in range(self.num_blocks) ] }
                                                        ) for i in range(self.m_depth) ])
        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = self.dim, channel = middle_channels[i+1], channel_dim = -1, **channel_attention) 
                           for i in range(self.m_depth)]
            self.mca = nn.ModuleList(ca_sequence)
        ### fully connected layers
        if self.vector_embedding:
            dense_channels[0] = int( math.prod(self.image_size) / ((2**self.d_depth * self.patch_size)**self.dim ) *dense_channels[0])
            if last_activation:
                dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                                norm = normalization, num_norm_groups=num_norm_groups, 
                                                activation = self.activation) for i in range(len(dense_channels) - 1)]
            else:
                dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                                norm = normalization, num_norm_groups=num_norm_groups, 
                                                activation = self.activation) for i in range(len(dense_channels) - 2)]
                dense_sequence = dense_sequence + [DenseBlock(dense_channels[-2], dense_channels[-1], norm=None, activation = None), ]
            self.dense = nn.ModuleList([nn.Sequential(*(copy.deepcopy(dense_sequence)) ) for _ in range(z_count) ])

        ## set out_channel
        self.out_channel = dense_channels[-1]
        self.return_feature_list = return_feature_list
    def forward(self, x, t = None):
        x = self.patch_emb(x)
        res = []
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        for did, (d_ssm, down) in enumerate(zip(self.d_ssm, self.down)):
            x = d_ssm(x, t)
            if hasattr(self, 'dca'):
                x = self.dca[did](x)
            res = res + [x,]
            x = down(x)

        for mid, m_ssm in enumerate(self.middle):
            x = m_ssm(x, t)
            if hasattr(self, 'mca'):
                x = self.mca[mid](x)
        ### The last dimension is feature channel
        if self.vector_embedding:
            x = x.reshape(x.shape[0], -1)
            x = [ self.dense[i](x) for i in range(self.z_count) ]
        else:
            x = list(torch.chunk(x, self.z_count, dim = -1))
        if self.z_count == 1:
            x = x[0]
        if self.return_feature_list:
            return x, res
        return x

    def __str__(self):
        _str = ''
        # print down sampling
        _str += 'Patch merging and mamba SSM layers: '
        for c in self.down_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.down_path[-1])
        _str += '\n'

        if len(self.middle_path) > 1:
            _str += 'Middle mamba SSM layers: '
            for c in self.middle_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.middle_path[-1])
            _str += '\n'

        if self.vector_embedding:
            _str = _str + 'Dense layers: '
            for c in self.dense_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.dense_path[-1])
            _str += '\n'
        return _str 
    
class VMambaDecoder(nn.Module):
    def __init__(self, image_size, image_channel = 3, 
                patch_size = 2, in_channel = 64,
                mlp_hidden_ratio=[4., ], dense_channels = [32], 
                up_channels = [128, 64], final_channels = [64, 64], 
                time_channel = 0,
                building_block = 'vmamba',
                state_channel=None, 
                conv_kernel_size=3,
                inner_factor = 2.0, 
                dt_rank=None, dt_min=0.001, 
                dt_max=0.1, dt_init="random", dt_scale=1.0, 
                dt_init_floor=1e-4, 
                conv_bias=True, bias=False,
                head_channel = 64, 
                learnable_init_states = True, 
                chunk_size=256,             
                dropout=0., drop_path=0.1, 
                normalization = 'group', num_norm_groups = 8, 
                num_blocks = 2, activation = 'silu', 
                scan_mode = 'single', flip_scan = True, 
                return_feature_list = False,
                channel_attention = None,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.image_size = image_size
        self.image_channel = image_channel
        self.dim = len(image_size)
        self.u_depth = len(up_channels)
        self.f_depth = len(final_channels)
        self.activation = activation
        self.dropout = dropout
        self.num_blocks = num_blocks
        # stochastic depth decay rule
        self.drop_path = [x.item() for x in torch.linspace(0, drop_path,
                                                (self.u_depth + self.f_depth) * self.num_blocks)]
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.vector_embedding = isinstance(dense_channels, list) and len(dense_channels) > 0

        ### building block: mamba SSM block
        assert 'vmamba' in building_block, 'VMambaDecoder only support mamba-related building blocks.'
        self.BuildingBlock = get_building_block(building_block, dim = self.dim,
                                mlp_hidden_ratio = mlp_hidden_ratio,
                                time_channel = time_channel,
                                inner_factor = inner_factor, 
                                conv_kernel_size = conv_kernel_size,
                                dt_rank = dt_rank,
                                dt_min = dt_min, dt_max = dt_max,
                                dt_init = dt_init,
                                dt_scale = dt_scale,
                                dt_init_floor = dt_init_floor,
                                head_channel = head_channel, 
                                learnable_init_states = learnable_init_states, 
                                chunk_size=chunk_size,
                                conv_bias = conv_bias, bias = bias,
                                dropout = dropout,
                                norm = normalization, num_norm_groups = num_norm_groups,
                                state_channel = state_channel,
                                activation = activation,
                                scan_mode = scan_mode, flip_scan = flip_scan)
        ### use patch expansion block for up sampling
        ## fully connected layer
        dense_channels = [in_channel, ] + dense_channels 
        if not sum([im_size % (self.patch_size * (2** self.u_depth)) for im_size in self.image_size ]) == 0:
            logger.error('Please check your image size, patch size and downsample depth to make sure the image size can be divisible.')
            exit(1)
        if self.vector_embedding:
            # used for view (reshape)
            self.view_shape = [-1, ] +[int(im_size // (self.patch_size * (2** self.u_depth))) for im_size in self.image_size ]  + [int( dense_channels[-1]),]
            module_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                                norm = normalization, num_norm_groups=num_norm_groups,
                                                activation = self.activation) for i in range(len(dense_channels) - 2)]
            # to construct image shape
            # if there is not fc layer, then we also don't need this step
            module_sequence.append(DenseBlock(dense_channels[-2],  
                                                       int( dense_channels[-1] * math.prod(self.image_size) / ((2**self.u_depth * self.patch_size)**self.dim  )), 
                                                        norm = normalization, num_norm_groups=num_norm_groups, 
                                                        activation = self.activation))
            self.dense = nn.Sequential(*module_sequence)  
        self.dense_path = dense_channels
        ### use patch expansion block for up sampling
        up_channels = [dense_channels[-1], ] + up_channels

        self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                     in_channel = up_channels[i],
                                                     out_channel = up_channels[i],
                                                     norm = normalization, 
                                                     num_norm_groups = num_norm_groups) for i in range(self.u_depth)])
        ### building block
        self.u_ssm = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i+1],
                                                        kwargs_list = {"drop_path": [self.drop_path[i*self.num_blocks + ni] for ni in range(self.num_blocks) ] }
                                                        ) for i in range(self.u_depth) ])
        self.up_path = up_channels
        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = self.dim, channel = up_channels[i+1], channel_dim = -1, **channel_attention) 
                           for i in range(len(up_channels) - 1)]
            self.uca = nn.ModuleList(ca_sequence)
        final_channels = [up_channels[-1]] + final_channels
        self.final = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        in_channel = final_channels[i],
                                                        out_channel = final_channels[i+1],
                                                        kwargs_list = {"drop_path": [self.drop_path[(i + self.u_depth)*self.num_blocks + ni] 
                                                                                     for ni in range(self.num_blocks) ] }
                                                        ) for i in range(self.f_depth) ])
        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = self.dim, channel = final_channels[i+1], channel_dim = -1, **channel_attention) 
                           for i in range(len(final_channels) - 1)]
            self.fca = nn.ModuleList(ca_sequence)
        ### from patch to image: up_sample
        self.patch_recov = PatchRecoveryBlock(dim = self.dim, 
                                                patch_size = self.patch_size,
                                                in_channel = final_channels[-1],
                                                out_channel = self.image_channel,
                                                norm = None)
        self.final_path = final_channels + [self.image_channel]
        self.return_feature_list = return_feature_list
    def __str__(self):
        _str = ''
        if self.vector_embedding:
            _str = _str + 'Dense layers: '
            for c in self.dense_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.dense_path[-1])
            _str += '\n'
        if len(self.up_path) > 1:
            _str += 'Patch expansion and mamba SSM layers: '
            for c in self.up_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.up_path[-1])
            _str += '\n'
        if len(self.final_path) > 1:
            _str += 'Final mamba SSM layers: '
            for c in self.final_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.final_path[-1])
            _str += '\n'
        return _str 
    def forward(self, x, t = None):
        if type(x) == tuple:
            x = x[0]
        if self.vector_embedding:
            x = self.dense(x)
            x = x.reshape(*self.view_shape)
        res = []
        for uid, (up, u_ssm) in enumerate(zip(self.up, self.u_ssm)):
            x = u_ssm(up(x), t)
            if hasattr(self, 'uca'):
                x = self.uca[uid](x)
            res = res + [x,]
        for fid, f_ssm in enumerate(self.final):
            x = f_ssm(x, t)
            if hasattr(self, 'fca'):
                x = self.fca[fid](x)
        x = self.patch_recov(x)
        if self.return_feature_list:
            return x, res
        return x
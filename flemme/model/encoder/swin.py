### swin-encoder: window and patch based vision transformer
import torch
import torch.nn.functional as F
from torch import nn
import math
from flemme.block import PatchConstructionBlock, PatchRecoveryBlock,\
    PatchExpansionBlock, PatchMergingBlock, \
    SequentialT, get_building_block, MultipleBuildingBlocks, DenseBlock
from flemme.logger import get_logger
logger = get_logger("model.encoder.swin")

class SwinEncoder(nn.Module):
    def __init__(self, image_size, image_channel = 3, 
                 window_size = 8, time_channel = 0,
                 patch_size = 2, patch_channel = 32,
                 building_block = 'swin', dense_channels = [256], 
                 mlp_hidden_ratio=[4., ], qkv_bias=True, qk_scale=None, 
                 down_channels = [128, 256], middle_channels = [256, 256], 
                 down_num_heads = [3, 3], middle_num_heads = [3, 3],
                 dropout=0., atten_dropout=0., drop_path=0.1, 
                 normalization = 'group', num_groups = 8, 
                 num_block = 2, activation = 'silu', 
                 abs_pos_embedding = False,
                 return_features = False,
                 z_count = 1, **kwargs):
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
        self.atten_dropout = atten_dropout
        self.num_block = num_block
        # stochastic depth decay rule
        self.drop_path = [x.item() for x in torch.linspace(0, drop_path,
                                                (self.d_depth + self.m_depth) * self.num_block)]
        self.window_size = window_size
        if not isinstance(window_size, tuple) and not isinstance(window_size, list):
            self.window_size = [self.window_size for _ in range(self.dim)]
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.patch_size = patch_size
        self.patch_channel = patch_channel
        self.patch_image_pyramid = [ [s // patch_size // (2 **i) for s in self.image_size] 
                                for i in range(self.d_depth + 1)]
        assert 'swin' in building_block, 'SwinEncoder only support swin-related building blocks.'
        ### building block: swin transformer block
        self.BuildingBlock = get_building_block(building_block, 
                                window_size = self.window_size, 
                                mlp_hidden_ratio = mlp_hidden_ratio,
                                time_channel = time_channel, 
                                qkv_bias = qkv_bias,
                                qk_scale = qk_scale,
                                dropout = dropout,
                                atten_dropout = atten_dropout,
                                norm = normalization, num_groups = num_groups,
                                activation = activation)
        ### construct patch
        self.patch_emb = PatchConstructionBlock(dim = self.dim, 
                                                patch_size = self.patch_size,
                                                in_channel = self.image_channel,
                                                out_channel = self.patch_channel,
                                                norm = None)
        self.absolute_pos_embed = None
        if abs_pos_embedding:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros([1,] + self.patch_image_pyramid[0] + [self.patch_channel,]))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        ### use patch merging block for down sampling
        down_channels = [self.patch_channel, ] + down_channels

        ### building block
        self.d_trans = nn.ModuleList([MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        num_heads = down_num_heads[i],
                                                        in_channel = down_channels[i],
                                                        out_channel = down_channels[i+1],
                                                        patch_image_size = self.patch_image_pyramid[i],
                                                        kwargs_list = {"shift_size":  [(ni % 2) * (min(self.window_size) // 2) for ni in range(self.num_block) ],
                                                                       "drop_path": [self.drop_path[i*self.num_block + ni] for ni in range(self.num_block) ] }
                                                        ) for i in range(self.d_depth) ])
        self.down = nn.ModuleList([PatchMergingBlock(dim = self.dim,
                                                     in_channel = down_channels[i+1],
                                                     out_channel = down_channels[i+1],
                                                     norm = normalization, 
                                                     num_groups = num_groups)  for i in range(self.d_depth)])
        self.down_path = [self.image_channel, ] + down_channels

        if not self.vector_embedding:
            middle_channels = [mc * self.z_count for mc in middle_channels]
        ### middle transformers
        middle_channels = [down_channels[-1], ] + middle_channels
        self.middle = SequentialT(*[MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        num_heads = middle_num_heads[i],
                                                        in_channel = middle_channels[i],
                                                        out_channel = middle_channels[i+1],
                                                        patch_image_size = self.patch_image_pyramid[-1],
                                                        kwargs_list = {"shift_size":  [(ni % 2) * (min(self.window_size) // 2)
                                                                                       for ni in range(self.num_block) ],
                                                                       "drop_path": [self.drop_path[(i + self.d_depth)*self.num_block + ni] 
                                                                                     for ni in range(self.num_block) ] }
                                                        ) for i in range(self.m_depth) ])
        self.middle_path = middle_channels

        ### fully connected layers
        dense_channels = [ int(middle_channels[-1] / self.z_count) if not self.vector_embedding else middle_channels[-1], ] + dense_channels
        
        self.dense_path = dense_channels.copy()
        if self.vector_embedding:
            dense_channels[0] = int( math.prod(self.image_size) / ((2**self.d_depth * self.patch_size)**self.dim ) *dense_channels[0])
            dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                                norm = normalization, num_groups=num_groups, 
                                                activation = self.activation) for i in range(len(dense_channels) - 2)]
            dense_sequence = dense_sequence + [DenseBlock(dense_channels[-2], dense_channels[-1], norm=None, activation = None), ]
            self.dense = nn.ModuleList([nn.Sequential(*(dense_sequence.copy()) ) for _ in range(z_count) ])

        ## set out_channel
        self.out_channel = dense_channels[-1]
        self.return_features = return_features
    def forward(self, x, t = None):
        x = self.patch_emb(x)
        res = []
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        for d_trans, down in zip(self.d_trans, self.down):
            x = d_trans(x, t)
            res = res + [x,]
            x = down(x)
        x, _ = self.middle(x, t)
        ### The last dimension is feature channel
        if self.vector_embedding:
            x = x.reshape(x.shape[0], -1)
            x = [ self.dense[i](x) for i in range(self.z_count) ]
        else:
            x = torch.split(x, self.out_channel, dim=-1)
        if self.z_count == 1:
            x = x[0]
        if self.return_features:
            return x, res
        return x
    def __str__(self):
        _str = ''
        # print down sampling
        _str += 'Patch merging and swin transformer layers: '
        for c in self.down_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.down_path[-1])
        _str += '\n'

        if len(self.middle_path) > 1:
            _str += 'Middle swin tranformer layers: '
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

class SwinDecoder(nn.Module):
    def __init__(self, image_size, image_channel = 3, in_channel = 64,
                 window_size = 8, patch_size = 2, dense_channels = [32], 
                 building_block = 'swin', time_channel = 0,
                 mlp_hidden_ratio=[4., ], qkv_bias=True, qk_scale=None, 
                 up_channels = [128, 64], final_channels = [64, 64], 
                 up_num_heads = [3, 3], final_num_heads = [3, 3],
                 dropout=0., atten_dropout=0., drop_path=0.1, 
                 normalization = 'group', num_groups = 8, 
                 num_block = 2, activation = 'silu', 
                 return_features = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        assert 'swin' in building_block, 'SwinDecoder only support swin-related building blocks.'
        self.image_size = image_size
        self.image_channel = image_channel
        self.dim = len(image_size)
        self.u_depth = len(up_channels)
        self.f_depth = len(final_channels)
        self.activation = activation
        self.dropout = dropout
        self.atten_dropout = atten_dropout
        self.vector_embedding = isinstance(dense_channels, list) and len(dense_channels) > 0
        self.num_block = num_block
        # stochastic depth decay rule
        self.drop_path = [x.item() for x in torch.linspace(0, drop_path,
                                                (self.u_depth + self.f_depth) * self.num_block)][::-1]
        self.window_size = window_size
        if not isinstance(window_size, tuple) and not isinstance(window_size, list):
            self.window_size = [self.window_size for _ in range(self.dim)]
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.patch_image_pyramid = [ [s //patch_size // (2 **i) for s in self.image_size] for i in range(self.u_depth + 1)][::-1]

        ### building block: swin transformer block
        self.BuildingBlock = get_building_block(building_block, 
                                window_size = self.window_size, 
                                time_channel = time_channel,
                                mlp_hidden_ratio = mlp_hidden_ratio,
                                qkv_bias = qkv_bias,
                                qk_scale = qk_scale,
                                dropout = dropout,
                                atten_dropout = atten_dropout,
                                norm = normalization, num_groups = num_groups,
                                activation = activation)
        ## fully connected layer
        dense_channels = [in_channel, ] + dense_channels 
        if not sum([im_size % (self.patch_size * (2** self.u_depth)) for im_size in self.image_size ]) == 0:
            logger.error('Please check your image size, patch size and downsample depth to make sure the image size can be divisible.')
            exit(1)
        if self.vector_embedding:
            # used for view (reshape)
            self.view_shape = [-1, ] + [int(im_size // (self.patch_size * (2** self.u_depth))) for im_size in self.image_size ] + [int( dense_channels[-1]),]
            module_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                                    norm = normalization, num_groups=num_groups, 
                                                    activation = self.activation) for i in range(len(dense_channels) - 2)]
            # to construct image shape
            # if there is not fc layer, then we also don't need this step
            module_sequence.append(DenseBlock(dense_channels[-2],  
                                                       int( dense_channels[-1] * math.prod(self.image_size) / ((2**self.u_depth * self.patch_size)**self.dim  )), 
                                                       norm = normalization, num_groups=num_groups,  
                                                       activation = self.activation))
            self.dense = nn.Sequential(*module_sequence)  
        self.dense_path = dense_channels
        ### use patch expansion block for up sampling
        up_channels = [dense_channels[-1], ] + up_channels
        self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                     in_channel = up_channels[i],
                                                     out_channel = up_channels[i],
                                                     norm = normalization, 
                                                     num_groups = num_groups) for i in range(self.u_depth)])
        ### building block
        self.u_trans = nn.ModuleList([MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        num_heads = up_num_heads[i],
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i+1],
                                                        patch_image_size = self.patch_image_pyramid[i+1],
                                                        kwargs_list = {"shift_size":  [(ni % 2) * (min(self.window_size) // 2) for ni in range(self.num_block) ],
                                                                       "drop_path": [self.drop_path[i*self.num_block + ni] for ni in range(self.num_block) ] }
                                                        ) for i in range(self.u_depth) ])
        self.up_path = up_channels

        final_channels = [up_channels[-1]] + final_channels
        self.final = SequentialT(*[MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        num_heads = final_num_heads[i],
                                                        in_channel = final_channels[i],
                                                        out_channel = final_channels[i+1],
                                                        patch_image_size = self.patch_image_pyramid[-1],
                                                        kwargs_list = {"shift_size":  [(ni % 2) * (min(self.window_size) // 2) 
                                                                                       for ni in range(self.num_block) ],
                                                                       "drop_path": [self.drop_path[(i + self._depth)*self.num_block + ni] 
                                                                                     for ni in range(self.num_block) ] }
                                                        ) for i in range(self.f_depth) ])
        ### from patch to image: up_sample
        self.patch_recov = PatchRecoveryBlock(dim = self.dim, 
                                                patch_size = self.patch_size,
                                                in_channel = final_channels[-1],
                                                out_channel = self.image_channel,
                                                norm = None)
        self.final_path = final_channels + [self.image_channel]
        self.return_features = return_features
    def __str__(self):
        _str = ''
        if self.vector_embedding:
            _str = _str + 'Dense layers: '
            for c in self.dense_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.dense_path[-1])
            _str += '\n'
        if len(self.up_path) > 1:
            _str += 'Patch expansion and swin transformer layers: '
            for c in self.up_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.up_path[-1])
            _str += '\n'
        if len(self.final_path) > 1:
            _str += 'Final swin transformer layers: '
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
        for up, u_trans in zip(self.up, self.u_trans):
            x = u_trans(up(x), t)
            res = res + [x,]
        x, _ = self.final(x, t)
        x = self.patch_recov(x)
        if self.return_features:
            return x, res
        return x
    


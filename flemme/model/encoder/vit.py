### vision transformer
import torch
import torch.nn.functional as F
from torch import nn
import math
from flemme.block import PatchConstructionBlock, PatchRecoveryBlock,\
    PatchExpansionBlock, PatchMergingBlock, \
    SequentialT, get_building_block, MultipleBuildingBlocks, DenseBlock
from flemme.logger import get_logger
logger = get_logger("model.encoder.vit")

class ViTEncoder(nn.Module):
    def __init__(self, image_size, image_channel = 3, 
                 patch_size = 4, patch_channel = 32,
                 building_block = 'vit', fc_channels = [256], 
                 time_channel = 0,
                 mlp_hidden_ratio=[4., ], qkv_bias=True, qk_scale=None, 
                 down_channels = [128, 256], middle_channels = [256, 256], 
                 down_num_heads = [3, 6], middle_num_heads = [12, 24],
                 dropout=0., atten_dropout=0., drop_path=0.1, 
                 normalization = 'layer', num_group = 8, 
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
        self.vector_embedding = isinstance(fc_channels, list) and len(fc_channels) > 0
        self.activation = activation
        self.dropout = dropout
        self.atten_dropout = atten_dropout
        self.num_block = num_block
        # stochastic depth decay rule
        self.drop_path = [x.item() for x in torch.linspace(0, drop_path,
                                                (self.d_depth + self.m_depth) * self.num_block)]
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.patch_size = patch_size
        self.patch_channel = patch_channel
        assert 'vit' in building_block, 'ViTEncoder only support transformer-related building blocks.'
        ### building block: vision transformer block
        self.BuildingBlock = get_building_block(building_block, 
                                dim = self.dim,
                                time_channel = time_channel,
                                mlp_hidden_ratio = mlp_hidden_ratio,
                                qkv_bias = qkv_bias,
                                qk_scale = qk_scale,
                                dropout = dropout,
                                atten_dropout = atten_dropout,
                                norm = normalization, num_group = num_group,
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
                torch.zeros([1,] + [s // self.patch_size for s in self.image_size] + [self.patch_channel,]))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        ### use patch merging block for down sampling
        down_channels = [self.patch_channel, ] + down_channels

        ### building block
        self.d_trans = nn.ModuleList([MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        num_heads = down_num_heads[i],
                                                        in_channel = down_channels[i],
                                                        out_channel = down_channels[i+1],
                                                        kwargs_list = {"drop_path": [self.drop_path[i*self.num_block + ni] for ni in range(self.num_block) ] }
                                                        ) for i in range(self.d_depth) ])
        self.down = nn.ModuleList([PatchMergingBlock(dim = self.dim,
                                                     in_channel = down_channels[i+1],
                                                     out_channel = down_channels[i+1],
                                                     norm = normalization, 
                                                     num_group = num_group)  for i in range(self.d_depth)])
        self.down_path = [self.image_channel, ] + down_channels

        if not self.vector_embedding:
            middle_channels = [mc * self.z_count for mc in middle_channels]
        ### middle transformers
        middle_channels = [down_channels[-1], ] + middle_channels
        self.middle = SequentialT(*[MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        num_heads = middle_num_heads[i],
                                                        in_channel = middle_channels[i],
                                                        out_channel = middle_channels[i+1],
                                                        kwargs_list = {"drop_path": [self.drop_path[(i + self.d_depth)*self.num_block + ni] 
                                                                                     for ni in range(self.num_block) ] }
                                                        ) for i in range(self.m_depth) ])
        self.middle_path = middle_channels

        ### fully connected layers
        fc_channels = [ int(middle_channels[-1] / self.z_count) if not self.vector_embedding else middle_channels[-1], ] + fc_channels
        
        self.fc_path = fc_channels.copy()
        if self.vector_embedding:
            fc_channels[0] = int( math.prod(self.image_size) / ((2**self.d_depth * self.patch_size)**self.dim ) *fc_channels[0])
            fc_sequence = [ DenseBlock(fc_channels[i], fc_channels[i+1], 
                                                norm = normalization, batch_dim=1, num_group=num_group, 
                                                    activation = self.activation) for i in range(len(fc_channels) - 2)]
            fc_sequence = fc_sequence + [DenseBlock(fc_channels[-2], fc_channels[-1], norm=None, activation = None), ]
            self.fc = nn.ModuleList([nn.Sequential(*(fc_sequence.copy()) ) for _ in range(z_count) ])

        ## set out_channel
        self.out_channel = fc_channels[-1]
        self.return_features = return_features
    def forward(self, x, t = None):
        x = self.patch_emb(x)
        res = []
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        for d_trans, down in zip(self.d_trans, self.down):
            x = d_trans(x, t)
            res = [x,] + res
            x = down(x)
        x, _ = self.middle(x, t)
        ### The last dimension is feature channel
        if self.vector_embedding:
            x = x.reshape(x.shape[0], -1)
            x = [ self.fc[i](x) for i in range(self.z_count) ]
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
        _str += 'Patch merging and vision transformer layers: '
        for c in self.down_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.down_path[-1])
        _str += '\n'

        if len(self.middle_path) > 1:
            _str += 'Middle vit tranformer layers: '
            for c in self.middle_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.middle_path[-1])
            _str += '\n'

        if self.vector_embedding:
            _str = _str + 'Fully-connected layers: '
            for c in self.fc_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.fc_path[-1])
            _str += '\n'
        return _str 

class ViTDecoder(nn.Module):
    def __init__(self, image_size, image_channel = 3, in_channel = 64,
                 patch_size = 4, fc_channels = [32], 
                 building_block = 'vit',
                 time_channel = 0,
                 mlp_hidden_ratio=[4., ], qkv_bias=True, qk_scale=None, 
                 up_channels = [128, 64], final_channels = [64, 64], 
                 up_num_heads = [24, 12], final_num_heads = [6, 3],
                 dropout=0., atten_dropout=0., drop_path=0.1, 
                 normalization = 'layer', num_group = 8, 
                 num_block = 2, activation = 'silu', 
                 return_features = False, **kwargs):
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
        self.atten_dropout = atten_dropout
        self.vector_embedding = isinstance(fc_channels, list) and len(fc_channels) > 0
        self.num_block = num_block
        # stochastic depth decay rule
        self.drop_path = [x.item() for x in torch.linspace(0, drop_path,
                                                (self.u_depth + self.f_depth) * self.num_block)][::-1]
        self.mlp_hidden_ratio = mlp_hidden_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.patch_size = patch_size
        self.in_channel = in_channel
        ### building block: vision transformer block
        assert 'vit' in building_block, 'ViTDecoder only support transformer-related building blocks.'
        self.BuildingBlock = get_building_block(building_block, 
                                mlp_hidden_ratio = mlp_hidden_ratio,
                                dim = self.dim,
                                time_channel = time_channel,
                                qkv_bias = qkv_bias,
                                qk_scale = qk_scale,
                                dropout = dropout,
                                atten_dropout = atten_dropout,
                                norm = normalization, num_group = num_group,
                                activation = activation)
        ## fully connected layer
        fc_channels = [in_channel, ] + fc_channels 
        if not sum([im_size % (self.patch_size * (2** self.u_depth)) for im_size in self.image_size ]) == 0:
            logger.error('Please check your image size, patch size and downsample depth to make sure the image size can be divisible.')
            exit(1)
        if self.vector_embedding:
            # used for view (reshape)
            self.view_shape = [-1, ] + [int(im_size // (self.patch_size * (2** self.u_depth))) for im_size in self.image_size ] + [int( fc_channels[-1]),]
            module_sequence = [ DenseBlock(fc_channels[i], fc_channels[i+1], 
                                                    norm = normalization, batch_dim=1, num_group=num_group, 
                                                activation = self.activation) for i in range(len(fc_channels) - 2)]
            # to construct image shape
            # if there is not fc layer, then we also don't need this step
            module_sequence.append(DenseBlock(fc_channels[-2],  
                                                       int( fc_channels[-1] * math.prod(self.image_size) / ((2**self.u_depth * self.patch_size)**self.dim  )), 
                                                       norm = normalization, batch_dim=1, num_group=num_group,  
                                                       activation = self.activation))
            self.fc = nn.Sequential(*module_sequence)  
        self.fc_path = fc_channels
        ### use patch expansion block for up sampling
        up_channels = [fc_channels[-1], ] + up_channels
        self.up = nn.ModuleList([PatchExpansionBlock(dim = self.dim,
                                                     in_channel = up_channels[i],
                                                     out_channel = up_channels[i],
                                                     norm = normalization, 
                                                     num_group = num_group) for i in range(self.u_depth)])
        ### building block
        self.u_trans = nn.ModuleList([MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        num_heads = up_num_heads[i],
                                                        in_channel = up_channels[i],
                                                        out_channel = up_channels[i+1],
                                                        kwargs_list = {"drop_path": [self.drop_path[i*self.num_block + ni] 
                                                                                     for ni in range(self.num_block) ] }
                                                        ) for i in range(self.u_depth) ])
        self.up_path = up_channels

        final_channels = [up_channels[-1]] + final_channels
        self.final = SequentialT(*[MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        num_heads = final_num_heads[i],
                                                        in_channel = final_channels[i],
                                                        out_channel = final_channels[i+1],
                                                        kwargs_list = {"drop_path": [self.drop_path[(i + self._depth)*self.num_block + ni] 
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
            _str = _str + 'Fully-connected layers: '
            for c in self.fc_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.fc_path[-1])
            _str += '\n'
        if len(self.up_path) > 1:
            _str += 'Patch expansion and vision transformer layers: '
            for c in self.up_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.up_path[-1])
            _str += '\n'
        if len(self.final_path) > 1:
            _str += 'Final vision transformer layers: '
            for c in self.final_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.final_path[-1])
            _str += '\n'
        return _str 
    def forward(self, x, t = None):
        if type(x) == tuple:
            x = x[0]
        if self.vector_embedding:
            x = self.fc(x)
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
    


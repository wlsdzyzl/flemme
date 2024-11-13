# point cloud encoder for 3D point cloud
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, SequentialT
from flemme.logger import get_logger
import copy
logger = get_logger("model.encoder.pointwise")
1### A very simple point-wise encoder
## Point-wise encoder only has linear and activation layers
## Point-wise encoder doesn't compress the data at all, 
## Therefore it's not suitable to be used to constuct a VAE by itself.
### Note that point-wise network can be used on any kinds of data, because it only operate the last channel.
class PointWiseEncoder(nn.Module):
    def __init__(self, point_dim=3, time_channel = 0, building_block = 'dense', dense_channels = [256], 
                normalization = 'group', num_groups = 8, 
                activation = 'lrelu', dropout = 0., z_count = 1, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_dim = point_dim
        self.activation = activation
        self.z_count = z_count
        self.pointwise = True
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                                activation=activation, 
                                                norm = normalization, 
                                                num_groups = num_groups, 
                                                dropout = dropout)
        dense_channels = [point_dim,] + dense_channels
        dense_sequence = [self.BuildingBlock(in_channel=dense_channels[i], out_channel=dense_channels[i+1]) 
                                        for i in range(len(dense_channels) - 1) ]
        self.dense = nn.ModuleList([SequentialT(*(copy.deepcopy(dense_sequence))) for _ in range(z_count)])
        self.dense_path = dense_channels
        self.out_channel = dense_channels[-1]

    def __str__(self):
        _str = ''
        # print convolution layers
        _str += 'Point-wise Dense layers:'
        for c in self.dense_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.dense_path[-1])
        _str += '\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None):
        # ## x is point cloud
        # print("?????", x.shape)
        x = [self.dense[i](x, t)[0] for i in range(self.z_count)]
        if self.z_count == 1:
            x = x[0]
        return x
        

# a very simple decoder
class PointWiseDecoder(nn.Module):
    def __init__(self, point_dim=3, in_channel = 256, time_channel = 0, 
                building_block = 'fc', dense_channels = [256], 
                normalization = 'group', num_groups = 8, 
                activation = 'lrelu', dropout = 0., **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_dim = point_dim
        self.activation = activation
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                                num_groups = num_groups)
        dense_channels = [in_channel,] + dense_channels
        dense_sequence = [self.BuildingBlock(in_channel=dense_channels[i], out_channel=dense_channels[i+1], 
                                              activation=activation, 
                                              norm = normalization, 
                                              dropout = dropout) 
                                        for i in range(len(dense_channels) - 1) ]
        dense_sequence = dense_sequence + [self.BuildingBlock(in_channel=dense_channels[-1], 
                                                                out_channel=point_dim, 
                                                                activation = None, 
                                                                norm = None,
                                                                dropout = None), ] 
        self.dense = SequentialT(*dense_sequence)
        self.dense_path = dense_channels + [self.point_dim, ]
    def __str__(self):
        _str = ''
        # print convolution layers
        _str += 'Point-wise Dense layers:'
        for c in self.dense_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.dense_path[-1])
        _str += '\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None):
        ## x is point cloud
        x, _ = self.dense(x, t)
        return x
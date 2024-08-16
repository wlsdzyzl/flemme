# point cloud encoder for 3D point cloud
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, SequentialT
from flemme.logger import get_logger
logger = get_logger("model.encoder.pointwise")
1### A very simple point-wise encoder
## Point-wise encoder only has linear and activation layers
## Point-wise encoder doesn't compress the data at all, 
## Therefore it's not suitable to be used to constuct a VAE by itself.
### Note that point-wise network can be used on any kinds of data, because it only operate the last channel.
class PointWiseEncoder(nn.Module):
    def __init__(self, point_dim=3, time_channel = 0, building_block = 'fc', fc_channels = [256], 
                normalization = 'group', num_group = 8, 
                activation = 'lrelu', dropout=0.1, z_count = 1, **kwargs):
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
                                                num_group = num_group, 
                                                batch_dim = 1,
                                                dropout = dropout)
        fc_channels = [point_dim,] + fc_channels
        fc_sequence = [self.BuildingBlock(in_channel=fc_channels[i], out_channel=fc_channels[i+1]) 
                                        for i in range(len(fc_channels) - 1) ]
        self.fc = nn.ModuleList([SequentialT(*(fc_sequence.copy())) for _ in range(z_count)])
        self.fc_path = fc_channels
        self.out_channel = fc_channels[-1]

    def __str__(self):
        _str = ''
        # print convolution layers
        _str += 'Point-wise Fully-connected layers:'
        for c in self.fc_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.fc_path[-1])
        _str += '\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None):
        # ## x is point cloud
        # print("?????", x.shape)
        x = [self.fc[i](x, t)[0] for i in range(self.z_count)]
        if self.z_count == 1:
            x = x[0]
        return x
        

# a very simple decoder
class PointWiseDecoder(nn.Module):
    def __init__(self, point_dim=3, in_channel = 256, time_channel = 0, 
                building_block = 'fc', fc_channels = [256], 
                normalization = 'group', num_group = 8, 
                activation = 'lrelu', dropout=0.1, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_dim = point_dim
        self.activation = activation
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                                num_group = num_group, 
                                                batch_dim = 1)
        fc_channels = [in_channel,] + fc_channels
        fc_sequence = [self.BuildingBlock(in_channel=fc_channels[i], out_channel=fc_channels[i+1], 
                                              activation=activation, 
                                              norm = normalization, 
                                              dropout = dropout) 
                                        for i in range(len(fc_channels) - 1) ]
        fc_sequence = fc_sequence + [self.BuildingBlock(in_channel=fc_channels[-1], 
                                                                out_channel=point_dim, 
                                                                activation = None, 
                                                                norm = None,
                                                                dropout = None), ] 
        self.fc = SequentialT(*fc_sequence)
        self.fc_path = fc_channels + [self.point_dim, ]
    def __str__(self):
        _str = ''
        # print convolution layers
        _str += 'Point-wise Fully-connected layers:'
        for c in self.fc_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.fc_path[-1])
        _str += '\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None):
        ## x is point cloud
        x, _ = self.fc(x, t)
        return x
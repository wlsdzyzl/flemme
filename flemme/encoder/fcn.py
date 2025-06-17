# point cloud encoder for 3D point cloud
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, SequentialT, MultipleBuildingBlocks, DenseBlock
from flemme.logger import get_logger
import copy
logger = get_logger("model.encoder.fcn")
### A very simple fully connected networks encoder to encoder vector
class FCNEncoder(nn.Module):
    def __init__(self, vec_dim=3, time_channel = 0, 
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'dense', seq_feature_channels = [256], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., z_count = 1,
                last_activation = True, 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.vec_dim = vec_dim
        self.activation = activation
        self.z_count = z_count
        self.vector_embedding = False
        self.num_blocks = num_blocks
        # self.time_channel = time_channel
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)

        seq_feature_channels = [vec_dim,] + seq_feature_channels
        sequence = [MultipleBuildingBlocks(n = self.num_blocks, 
                                           BuildingBlock=self.BuildingBlock,
                                           in_channel=seq_feature_channels[i], 
                                           out_channel=seq_feature_channels[i+1]) 
                                        for i in range(len(seq_feature_channels) - 1) ]
        if not last_activation:
            sequence.append(DenseBlock(seq_feature_channels[-1], seq_feature_channels[-1], norm = None, activation = None))
        self.seq = nn.ModuleList([SequentialT(*(copy.deepcopy(sequence))) for _ in range(z_count)])
        self.seq_path = seq_feature_channels
        self.out_channel = seq_feature_channels[-1]
    def __str__(self):
        _str = ''
        # print convolution layers
        _str += 'Fully Connected layers: '
        for c in self.seq_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.seq_path[-1])
        _str += '\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None, c = None):
        # ## x is point cloud
        x = [self.seq[i](x, t, c) for i in range(self.z_count)]
        if self.z_count == 1:
            x = x[0]
        return x
        

# a very simple decoder
class FCNDecoder(nn.Module):
    def __init__(self, vec_dim=3, in_channel = 256, time_channel = 0, 
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'dense', seq_feature_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.vec_dim = vec_dim
        self.activation = activation
        self.vector_embedding = False
        self.num_blocks = num_blocks

        # self.time_channel = time_channel
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)

        seq_feature_channels = [in_channel,] + seq_feature_channels
        sequence = [MultipleBuildingBlocks(n = self.num_blocks, 
                                           BuildingBlock=self.BuildingBlock,
                                           in_channel=seq_feature_channels[i], 
                                           out_channel=seq_feature_channels[i+1])  
                                        for i in range(len(seq_feature_channels) - 1) ]
        sequence.append(DenseBlock(seq_feature_channels[-1], vec_dim, norm = None, activation = None))
        self.seq = SequentialT(*(copy.deepcopy(sequence)))
        self.seq_path = seq_feature_channels


    def __str__(self):
        _str = ''
        # print convolution layers
        _str += 'Fully Connected layers: '
        for c in self.seq_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.seq_path[-1])
        _str += '\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None, c = None):
        ## x is point cloud
        x = self.seq(x, t, c)
        return x
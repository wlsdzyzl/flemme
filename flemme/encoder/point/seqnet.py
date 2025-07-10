# point cloud encoder for 3D point cloud
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, SequentialT, MultipleBuildingBlocks, DenseBlock, get_ca, VoxelLayer
from flemme.logger import get_logger
import copy
logger = get_logger("model.encoder.seqnet")
## only use dense layer to extract features.
## usually this leads to poor results.
class SeqEncoder(nn.Module):
    def __init__(self, point_dim, time_channel, 
                projection_channel,
                time_injection,
                num_blocks,
                seq_feature_channels, 
                voxel_resolutions,
                voxel_conv_kernel_size,
                with_se,
                coordinate_normalize,
                channel_attention, 
                num_norm_groups, 
                activation, 
                condition_channel,
                condition_injection,
                condition_first,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_dim = point_dim
        self.activation = activation
        self.vector_embedding = False
        self.point_proj = nn.Linear(point_dim, projection_channel)
        self.num_blocks = num_blocks

        self.seq_path = [projection_channel,] + seq_feature_channels
        self.out_channel = seq_feature_channels[-1]

        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = 1, channel = self.seq_path[i+1], channel_dim = -1, **channel_attention) 
                           for i in range(len(self.seq_path) - 1)]
            self.ca = nn.ModuleList(ca_sequence)
        
        # self.need_pos = len(self.voxel_resolutions) > 0

        # transformer voxel cnn
        if len(voxel_resolutions) > 0:
            logger.info('This model extracts voxel features.')
            VBuildingBlock = get_building_block('conv',
                                            dim = 3,
                                            time_channel = time_channel, 
                                            activation=activation, 
                                            norm = 'batch', 
                                            num_norm_groups = num_norm_groups, 
                                            kernel_size = voxel_conv_kernel_size,
                                            time_injection = time_injection,
                                            condition_channel = condition_channel,
                                            condition_injection = condition_injection,
                                            condition_first = condition_first)
            voxel_sequence = [VoxelLayer(resolution = voxel_resolutions[i], 
                                            in_channel = self.seq_path[i],
                                            out_channel = self.seq_path[i+1], 
                                            BuildingBlock = VBuildingBlock,
                                            num_blocks = self.num_blocks,
                                            with_se = with_se,
                                            coordinate_normalize=coordinate_normalize,
                                            ) for i in range(len(self.seq_path) - 1) ]
            self.vs = nn.ModuleList(voxel_sequence)
    def __str__(self):
        _str = f'Projection layer: {self.point_dim}->{self.seq_path[0]}\n'
        # print convolution layers
        _str += 'Sequential layers: '
        for c in self.seq_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.seq_path[-1])
        _str += '\n'
        return _str 

    # input: Nb * Np * d
    def forward(self, x, t = None, c = None):
        # ## x is point cloud
        pos = x[...,:3]
        x = self.point_proj(x)
        for sid in range(len(self.seq)):
            vx = x
            x = self.seq[sid](x, t, c)
            if hasattr(self, 'vs'):
                x = x + self.vs[sid](vx, pos, t, c)
            if hasattr(self, 'ca'):
                x = self.ca[sid](x)
        return x
class SeqNetEncoder(SeqEncoder):
    def __init__(self, point_dim=3, time_channel = 0, 
                projection_channel = 64,
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'dense', seq_feature_channels = [256], 
                voxel_resolutions = [],
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                channel_attention = None, 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(point_dim=point_dim, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                seq_feature_channels = seq_feature_channels, 
                num_blocks = num_blocks,
                num_norm_groups = num_norm_groups,
                activation = activation, 
                channel_attention = channel_attention,
                time_injection=time_injection,
                voxel_resolutions=voxel_resolutions,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = condition_first)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))

        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)

        sequence = [MultipleBuildingBlocks(n = self.num_blocks, 
                                           BuildingBlock=self.BuildingBlock,
                                           in_channel=self.seq_path[i], 
                                           out_channel=self.seq_path[i+1]) 
                                        for i in range(len(self.seq_path) - 1) ]
        self.seq = nn.ModuleList(sequence)
        

class SeqDecoder(nn.Module):
    def __init__(self, point_dim, in_channel,
                 num_blocks,
                seq_feature_channels, 
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_dim = point_dim
        self.vector_embedding = False
        self.seq = None
        self.num_blocks = num_blocks
        self.seq_path = [in_channel,] + seq_feature_channels
        self.latent_proj = nn.Linear(self.seq_path[-1], point_dim)

    def __str__(self):
        _str = ''
        # print convolution layers
        if len(self.seq_path) > 1:
            _str += 'Sequential layers: '
            for c in self.seq_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.seq_path[-1])
            _str += '\n'
        _str += f'Projection layer: {self.seq_path[-1]}->{self.point_dim}\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None, c = None):
        if self.seq:
            x = self.seq(x, t, c)
        return self.latent_proj(x)
    
class SeqNetDecoder(SeqDecoder):
    def __init__(self, point_dim=3, in_channel = 256, time_channel = 0, 
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'dense', seq_feature_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(point_dim=point_dim,
                         in_channel = in_channel,
                         num_blocks=num_blocks,
                         seq_feature_channels=seq_feature_channels)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))

        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)

        
        sequence = [MultipleBuildingBlocks(n = self.num_blocks, 
                                           BuildingBlock=self.BuildingBlock,
                                           in_channel=self.seq_path[i], 
                                           out_channel=self.seq_path[i+1])  
                                        for i in range(len(self.seq_path) - 1) ]
        if len(sequence):
            self.seq = SequentialT(*(copy.deepcopy(sequence)))



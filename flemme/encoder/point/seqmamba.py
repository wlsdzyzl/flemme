# point cloud encoder for 3D point cloud
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, SequentialT, MultipleBuildingBlocks, DenseBlock
from flemme.logger import get_logger
import copy
logger = get_logger("model.encoder.seqmamba")
class SeqMambaEncoder(nn.Module):
    def __init__(self, point_dim=3, time_channel = 0, 
                projection_channel = 64,
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'pmamba', seq_feature_channels = [256], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., z_count = 1,
                last_activation = True,
                state_channel = 64, 
                conv_kernel_size = 4, inner_factor = 2.0,  
                head_channel = 64,
                conv_bias=True, bias=False,
                learnable_init_states = True, chunk_size=256,
                dt_min=0.001, A_init_range=(1, 16),
                dt_max=0.1, dt_init_floor=1e-4, 
                dt_rank = None, dt_scale = 1.0,
                mlp_hidden_ratios=[4.0, 4.0], 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_dim = point_dim
        self.activation = activation
        self.z_count = z_count
        self.vector_embedding = False
        self.point_proj = nn.Linear(point_dim, projection_channel)
        self.num_blocks = num_blocks
        # self.time_channel = time_channel
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        state_channel = state_channel, 
                                        conv_kernel_size = conv_kernel_size, 
                                        inner_factor = inner_factor,  
                                        head_channel = head_channel,
                                        conv_bias=conv_bias, bias=bias,
                                        learnable_init_states = learnable_init_states, 
                                        chunk_size=chunk_size,
                                        dt_min=dt_min, A_init_range=A_init_range,
                                        dt_max=dt_max, dt_init_floor=dt_init_floor, 
                                        dt_rank = dt_rank, dt_scale = dt_scale,
                                        mlp_hidden_ratios = mlp_hidden_ratios,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)

        seq_feature_channels = [projection_channel,] + seq_feature_channels
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
        _str = f'Projection layer: {self.point_dim}->{self.seq_path[0]}\n'
        # print convolution layers
        _str += 'Sequential Mamba layers: '
        for c in self.seq_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.seq_path[-1])
        _str += '\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None, c = None):
        # ## x is point cloud
        x = self.point_proj(x)
        x = [self.seq[i](x, t, c) for i in range(self.z_count)]
        if self.z_count == 1:
            x = x[0]
        return x
        

# a very simple decoder
class SeqMambaDecoder(nn.Module):
    def __init__(self, point_dim=3, in_channel = 256, time_channel = 0, 
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'pmamba', seq_feature_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0.,
                state_channel = 64, 
                conv_kernel_size = 4, inner_factor = 2.0,  
                head_channel = 64,
                conv_bias=True, bias=False,
                learnable_init_states = True, chunk_size=256,
                dt_min=0.001, A_init_range=(1, 16),
                dt_max=0.1, dt_init_floor=1e-4, 
                dt_rank = None, dt_scale = 1.0,
                mlp_hidden_ratios=[4.0, 4.0], 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_dim = point_dim
        self.activation = activation
        self.vector_embedding = False
        self.num_blocks = num_blocks

        # self.time_channel = time_channel
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        state_channel = state_channel, 
                                        conv_kernel_size = conv_kernel_size, 
                                        inner_factor = inner_factor,  
                                        head_channel = head_channel,
                                        conv_bias=conv_bias, bias=bias,
                                        learnable_init_states = learnable_init_states, 
                                        chunk_size=chunk_size,
                                        dt_min=dt_min, A_init_range=A_init_range,
                                        dt_max=dt_max, dt_init_floor=dt_init_floor, 
                                        dt_rank = dt_rank, dt_scale = dt_scale,
                                        mlp_hidden_ratios = mlp_hidden_ratios,
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
        self.seq = None
        if len(sequence):
            self.seq = SequentialT(*(copy.deepcopy(sequence)))
        self.seq_path = seq_feature_channels
        self.latent_proj = nn.Linear(self.seq_path[-1], point_dim)
    def __str__(self):
        _str = ''
        if len(self.seq_path) > 1:
            _str += 'Sequential Mamba layers: '
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
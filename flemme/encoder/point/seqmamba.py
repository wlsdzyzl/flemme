# point cloud encoder for 3D point cloud
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, SequentialT, MultipleBuildingBlocks, DenseBlock
from flemme.logger import get_logger
import copy
from .seqnet import SeqEncoder, SeqDecoder
logger = get_logger("model.encoder.seqmamba")
class SeqMambaEncoder(SeqEncoder):
    def __init__(self, point_dim=3, time_channel = 0, 
                projection_channel = 64,
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'pmamba', seq_feature_channels = [256], 
                voxel_resolutions = [],
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                channel_attention = None, 
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


        sequence = [MultipleBuildingBlocks(n = self.num_blocks, 
                                           BuildingBlock=self.BuildingBlock,
                                           in_channel=self.seq_path[i], 
                                           out_channel=self.seq_path[i+1]) 
                                        for i in range(len(self.seq_path) - 1) ]

        self.seq = nn.ModuleList(sequence)
        

# a very simple decoder
class SeqMambaDecoder(SeqDecoder):
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

        sequence = [MultipleBuildingBlocks(n = self.num_blocks, 
                                           BuildingBlock=self.BuildingBlock,
                                           in_channel=self.seq_path[i], 
                                           out_channel=self.seq_path[i+1])  
                                        for i in range(len(self.seq_path) - 1) ]
        if len(sequence):
            self.seq = SequentialT(*(copy.deepcopy(sequence)))

import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, \
    LocalGraphLayer, MultipleBuildingBlocks
from .pointnet import PointEncoder
from flemme.logger import get_logger
logger = get_logger("encoder.point.pointmamba")
        
class PointMambaEncoder(PointEncoder):
    def __init__(self, point_dim=3, 
                projection_channel = 64,
                time_channel = 0,
                num_neighbors_k=0, 
                local_feature_channels = [64, 64, 128, 256], 
                num_blocks = 1,
                dense_channels = [256, 256],
                building_block = 'pmamba', 
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
                z_count = 1, vector_embedding = True, 
                last_activation = True,
                skip_connection = True,
                use_local = True,
                use_global = True,
                **kwargs):
        super().__init__(point_dim=point_dim, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                num_neighbors_k=num_neighbors_k, 
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                z_count = z_count, vector_embedding = vector_embedding,
                last_activation = last_activation)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))

        self.BuildingBlock = get_building_block(building_block, time_channel = self.time_channel, 
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
                                        skip_connection = skip_connection)

        ### convolution with kernel size = 1
        # compute point features
        ## local graph feature
        if self.num_neighbors_k > 0:
            mamba_sequence = [LocalGraphLayer(k = self.num_neighbors_k, 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            use_local = use_local,
                                            use_global = use_global,
                                            num_blocks = self.num_blocks) for i in range(len(self.lf_path) - 2) ]
        else:    
            mamba_sequence = [MultipleBuildingBlocks(in_channel=self.lf_path[i], 
                                            out_channel=self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks) for i in range(len(self.lf_path) - 2) ]
            
        mamba_sequence.append(self.BuildingBlock(in_channel=sum(self.lf_path[1:-1]), 
                                        out_channel=self.lf_path[-1]))

        self.lf = nn.ModuleList(mamba_sequence)
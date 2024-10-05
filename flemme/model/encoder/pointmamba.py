import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import SequentialT, get_building_block, FoldingLayer, LocalGraphLayer
from .pointnet import PointEncoder, PointDecoder
from flemme.logger import get_logger
logger = get_logger("model.encoder.pointmamba")
        
class PointMambaEncoder(PointEncoder):
    def __init__(self, point_dim=3, time_channel = 0, 
                local_graph_k=0, 
                local_feature_channels = [64, 64, 128, 256], 
                dense_channels = [256, 256],
                building_block = 'pmamba', 
                normalization = 'group', num_group = 8, 
                activation = 'lrelu', dropout = 0.,
                state_channel = 64, 
                conv_kernel_size = 4, inner_factor = 2.0,  
                head_channel = 64,
                conv_bias=True, bias=False,
                learnable_init_states = True, chunk_size=256,
                dt_min=0.001, A_init_range=(1, 16),
                dt_max=0.1, dt_init_floor=1e-4, 
                dt_rank = None, dt_scale = 1.0,
                z_count = 1, pointwise = False, 
                skip_connection = True,
                **kwargs):
        super().__init__(point_dim=point_dim, 
                local_graph_k=local_graph_k, 
                local_feature_channels = local_feature_channels, 
                dense_channels = dense_channels,
                normalization = normalization,
                num_group = num_group,
                activation = activation, dropout = dropout, 
                z_count = z_count, pointwise = pointwise)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))

        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_group = num_group, 
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
        if self.local_graph_k > 0:
            trans_sequence = [LocalGraphLayer(k = self.local_graph_k, 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            is_seq = True) for i in range(len(self.lf_path) - 2) ]
        else:    
            trans_sequence = [self.BuildingBlock(in_channel=self.lf_path[i], 
                                            out_channel=self.lf_path[i+1]) for i in range(len(self.lf_path) - 2) ]
            
        trans_sequence.append(self.BuildingBlock(in_channel=sum(self.lf_path[1:-1]), 
                                        out_channel=self.lf_path[-1]))

        self.lf = nn.ModuleList(trans_sequence)

class PointMambaDecoder(PointDecoder):
    def __init__(self, point_dim=3, point_num = 2048, 
                in_channel = 256, dense_channels = [256], 
                time_channel = 0, building_block = 'pmamba', 
                normalization = 'group', num_group = 8, 
                activation = 'lrelu', dropout = 0., 
                folding_times = 0, 
                base_shape_config = {},
                folding_hidden_channels = [512, 512],
                residual_attention = False,
                state_channel = 64, 
                conv_kernel_size = 4, inner_factor = 2.0,  
                head_channel = 64,
                conv_bias=True, bias=False,
                learnable_init_states = True, chunk_size=256,
                dt_min=0.001, A_init_range=(1, 16),
                dt_max=0.1, dt_init_floor=1e-4, 
                dt_rank = None, dt_scale = 1.0,
                skip_connection = True,
                pointwise = False, **kwargs):
        super().__init__(point_dim=point_dim, 
                point_num = point_num,
                in_channel = in_channel,
                dense_channels = dense_channels,
                normalization = normalization,
                num_group = num_group,
                activation = activation, dropout = dropout, 
                folding_times = folding_times,
                base_shape_config = base_shape_config,
                pointwise = pointwise)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        
        if self.folding_times > 0:
            self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                            activation=activation, 
                                            norm = normalization, 
                                            num_group = 1, 
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
            if self.folding_times < 1:
                logger.warning('Folding times should be larger than 1.')
            folding_channels = [dense_channels[-1] + 2, ] + [ dense_channels[-1] + point_dim] * (folding_times - 1)
            folding_sequence = [FoldingLayer(BuildingBlock = self.BuildingBlock,
                                in_channel = fc, out_channel = point_dim,
                                hidden_channels = folding_hidden_channels) for fc in folding_channels]
            self.fold = SequentialT(*folding_sequence)
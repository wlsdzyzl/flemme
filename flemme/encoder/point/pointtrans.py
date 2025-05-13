import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, LocalGraphLayer, MultipleBuildingBlocks
from .pointnet import PointEncoder
from flemme.logger import get_logger
logger = get_logger("encoder.point.pointtrans")
        
class PointTransEncoder(PointEncoder):
    def __init__(self, point_dim=3, 
                 projection_channel = 64,
                 time_channel = 0,
                 num_neighbors_k=0, 
                 local_feature_channels = [64, 64, 128, 256], 
                 num_blocks = 2,
                 dense_channels = [256, 256],
                 building_block = 'pct_sa', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., num_heads = 4, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 residual_attention = False, skip_connection = True,
                 z_count = 1, vector_embedding = True, 
                 last_activation = True,
                 channel_attention = None,
                 time_injection = 'gate_bias',
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
                last_activation = last_activation,
                channel_attention = channel_attention,
                time_injection=time_injection)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))

        self.BuildingBlock = get_building_block(building_block, time_channel = self.time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout, 
                                        residual_attention = residual_attention,
                                        skip_connection = skip_connection,
                                        time_injection = time_injection)

        # compute point features
        ## local graph feature
        if self.num_neighbors_k > 0:
            trans_sequence = [LocalGraphLayer(k = self.num_neighbors_k, 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,                                          
                                            num_blocks = self.num_blocks) for i in range(len(self.lf_path) - 2) ]
        else:    
            trans_sequence = [MultipleBuildingBlocks(in_channel=self.lf_path[i], 
                                            out_channel=self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks) for i in range(len(self.lf_path) - 2) ]
            
        trans_sequence.append(self.BuildingBlock(in_channel=sum(self.lf_path[1:-1]), 
                                        out_channel=self.lf_path[-1]))

        self.lf = nn.ModuleList(trans_sequence)
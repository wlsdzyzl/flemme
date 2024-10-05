import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import SequentialT, get_building_block, FoldingLayer, LocalGraphLayer
from .pointnet import PointEncoder, PointDecoder
from flemme.logger import get_logger
logger = get_logger("model.encoder.pointtrans")
        
class PointTransEncoder(PointEncoder):
    def __init__(self, point_dim=3, time_channel = 0, 
                 local_graph_k=0, 
                 local_feature_channels = [64, 64, 128, 256], 
                 dense_channels = [256, 256],
                 building_block = 'pct_sa', 
                 normalization = 'group', num_group = 8, 
                 activation = 'lrelu', dropout = 0., num_heads = 4, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 residual_attention = False, skip_connection = True,
                 z_count = 1, pointwise = False, **kwargs):
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
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout, 
                                        residual_attention = residual_attention,
                                        skip_connection = skip_connection)

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
        
class PointTransDecoder(PointDecoder):
    def __init__(self, point_dim=3, point_num = 2048, 
                in_channel = 256, dense_channels = [256], 
                time_channel = 0, building_block = 'pct_sa', 
                normalization = 'group', num_group = 8, 
                activation = 'lrelu', dropout = 0., 
                folding_times = 0, 
                base_shape_config = {},
                folding_hidden_channels = [512, 512],
                num_heads = 4, d_k = None, 
                qkv_bias = True, qk_scale = None, atten_dropout = None, 
                residual_attention = False, skip_connection = True,
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
                                            num_heads = num_heads, d_k = d_k, 
                                            qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                            atten_dropout = atten_dropout, 
                                            residual_attention = residual_attention,
                                            skip_connection = skip_connection)
            if self.folding_times < 1:
                logger.warning('Folding times should be larger than 1.')
            folding_channels = [dense_channels[-1] + 2, ] + [ dense_channels[-1] + point_dim] * (folding_times - 1)
            folding_sequence = [FoldingLayer(BuildingBlock = self.BuildingBlock,
                                in_channel = fc, out_channel = point_dim,
                                hidden_channels = folding_hidden_channels) for fc in folding_channels]
            self.fold = SequentialT(*folding_sequence)
# point cloud encoder for 3D point cloud
from torch import nn
from flemme.block import get_building_block, SequentialT, MultipleBuildingBlocks
from flemme.logger import get_logger
from .seqnet import SeqEncoder, SeqDecoder
import copy
logger = get_logger("model.encoder.seqtrans")
class SeqTransEncoder(SeqEncoder):
    def __init__(self, point_dim=3, time_channel = 0, 
                projection_channel = 64,
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'pct_sa', seq_feature_channels = [256], 
                voxel_resolutions = [],
                voxel_attens = [],
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                channel_attention = None, 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                num_heads = 4, d_k = None, 
                qkv_bias = True, qk_scale = None, atten_dropout = None, 
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
                normalization = normalization,
                voxel_resolutions=voxel_resolutions,
                voxel_attens = voxel_attens,
                voxel_conv_kernel_size = voxel_conv_kernel_size,
                with_se = with_se,
                coordinate_normalize = coordinate_normalize,
                condition_channel = condition_channel,
                condition_injection = condition_injection,
                condition_first = condition_first)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        # self.time_channel = time_channel
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout,
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
class SeqTransDecoder(SeqDecoder):
    def __init__(self, point_dim=3, latent_channel = 256, time_channel = 0, 
                time_injection = 'gate_bias',
                num_blocks = 2,
                building_block = 'pct_sa', seq_feature_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                num_heads = 4, d_k = None, 
                qkv_bias = True, qk_scale = None, atten_dropout = None, 
                mlp_hidden_ratios=[4.0, 4.0], 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(point_dim=point_dim,
                         latent_channel = latent_channel,
                         num_blocks=num_blocks,
                         seq_feature_channels=seq_feature_channels)
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))

        # self.time_channel = time_channel
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout,
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
            self.seq = nn.ModuleList(sequence)

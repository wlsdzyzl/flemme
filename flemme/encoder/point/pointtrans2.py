import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, FeaturePropogatingLayer as FPLayer,\
    SamplingAndGroupingLayer as MSGLayer, MultipleBuildingBlocks
from .pointnet2 import Point2Encoder, Point2Decoder
from flemme.logger import get_logger
logger = get_logger("encoder.point.pointtrans2")
        
class PointTrans2Encoder(Point2Encoder):
    def __init__(self, point_dim = 3,
                 projection_channel = 64,
                 time_channel = 0,
                 num_fps_points = [1024, 512, 256, 64],
                 num_neighbors_k = 32,
                 neighbor_radius = [0.1, 0.2, 0.4, 0.8], 
                 fps_feature_channels = [128, 256, 512, 1024], 
                 num_blocks = 2,
                 num_scales = 2,
                 use_xyz = True,
                 sorted_query = False,
                 knn_query = False,
                 dense_channels = [1024],
                 building_block = 'dense', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., 
                 num_heads = 4, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 mlp_hidden_ratios=[4.0, 4.0], 
                 vector_embedding = True, 
                 is_point2decoder = False,
                 long_range_modeling = False,
                 return_xyz = False,
                 final_concat = False,
                 pos_embedding = False,
                 channel_attention = None,
                 time_injection = 'gate_bias',
                 voxel_resolutions = [],
                 voxel_conv_kernel_size = 3,
                 with_se = False,
                 coordinate_normalize = True,
                 condition_channel = 0,
                 condition_injection = 'gate_bias',
                 condition_first = False,
                 **kwargs):
        super().__init__(point_dim=point_dim, 
                projection_channel = projection_channel,
                time_channel = time_channel,
                num_fps_points = num_fps_points,
                num_neighbors_k=num_neighbors_k,
                neighbor_radius = neighbor_radius,
                fps_feature_channels = fps_feature_channels,
                num_blocks = num_blocks,
                num_scales = num_scales,
                use_xyz = use_xyz,
                sorted_query = sorted_query,
                knn_query = knn_query,
                dense_channels = dense_channels,
                activation = activation, 
                dropout = dropout,
                normalization = normalization, 
                num_norm_groups = num_norm_groups,  
                vector_embedding = vector_embedding, 
                is_point2decoder = is_point2decoder,
                return_xyz = return_xyz,
                final_concat = final_concat,
                pos_embedding= pos_embedding,
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
            logger.debug("redundant parameters: {}".format(kwargs))
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout, 
                                        mlp_hidden_ratios = mlp_hidden_ratios,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)

        msg_sequence = [MSGLayer(in_channel = self.msg_path[fid], 
            out_channels = self.sub_out_channels[fid],
            num_fps_points = self.num_fps_points[fid],
            k = self.num_neighbors_k[fid],
            radius = self.neighbor_radius[fid],
            num_blocks = self.num_blocks,
            use_xyz = self.use_xyz,
            sorted_query = self.sorted_query,
            knn_query = self.knn_query,
            pos_embedding_channel = projection_channel if pos_embedding else point_dim,
            BuildingBlock = self.BuildingBlock) for fid in range(self.fps_depth)]
        self.msg = nn.ModuleList(msg_sequence)
        if long_range_modeling:
            lrm_sequence = [MultipleBuildingBlocks(in_channel = fps_feature_channels[fid], 
                out_channels = fps_feature_channels[fid],
                n = num_blocks,
                BuildingBlock = self.BuildingBlock) for fid in range(self.fps_depth)]
            self.lrm = nn.ModuleList(lrm_sequence)

class PointTrans2Decoder(Point2Decoder):
    def __init__(self, point_dim=3, point_num = 2048, 
                ### provide by encoder
                in_channels= [1024, 1024, 512, 256, 128], 
                time_channel = 0,
                building_block = 'pct_sa', 
                num_blocks = 2,
                fp_channels = [512, 512, 256, 128], 
                dense_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                num_heads = 4, d_k = None, 
                qkv_bias = True, qk_scale = None, atten_dropout = None, 
                mlp_hidden_ratios=[4.0, 4.0], 
                channel_attention = None,
                time_injection = 'gate_bias',
                voxel_resolutions = [],
                voxel_conv_kernel_size = 3,
                with_se = False,
                coordinate_normalize = True,
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(point_dim=point_dim, 
                point_num = point_num,
                in_channels = in_channels,
                time_channel = time_channel,
                num_blocks = num_blocks,
                fp_channels = fp_channels, 
                dense_channels = dense_channels, 
                normalization = normalization, 
                num_norm_groups = num_norm_groups, 
                activation = activation, 
                dropout = dropout,
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
            logger.debug("redundant parameters: {}".format(kwargs))
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups,
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout, 
                                        mlp_hidden_ratios = mlp_hidden_ratios,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)

        fp_sequence = [  FPLayer( in_channel_known = self.known_feature_channels[fid],
                                in_channel_unknown = self.unknow_feature_channels[fid],
                                out_channel = self.fp_path[fid + 1],
                                num_blocks = self.num_blocks,
                                BuildingBlock = self.BuildingBlock)
                            for fid in range(self.fp_depth)]
        self.fp = nn.ModuleList(fp_sequence)
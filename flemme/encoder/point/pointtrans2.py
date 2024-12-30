import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block, \
    SamplingAndGroupingBlock as MSGBlock, FeaturePropogatingBlock as FPBlock 
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
                 dense_channels = [1024],
                 building_block = 'dense', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., 
                 num_heads = 4, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 residual_attention = False, skip_connection = True,
                 vector_embedding = True, 
                 return_feature_list = True,
                 use_local = True,
                 use_global = True,
                 z_count = 1, 
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
                dense_channels = dense_channels,
                activation = activation, 
                dropout = dropout,
                normalization = normalization, 
                num_norm_groups = num_norm_groups,  
                z_count = z_count, 
                vector_embedding = vector_embedding, 
                return_feature_list = return_feature_list)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = self.time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout, 
                                        residual_attention = residual_attention,
                                        skip_connection = skip_connection)
        msg_sequence = [MSGBlock(in_channel = self.msg_path[fid], 
            out_channels = self.sub_out_channels[fid],
            num_fps_points = self.num_fps_points[fid],
            k = self.num_neighbors_k[fid],
            radius = self.neighbor_radius[fid],
            num_blocks = self.num_blocks,
            use_xyz = self.use_xyz,
            use_local = True,
            use_global = True,
            BuildingBlock = self.BuildingBlock) for fid in range(self.fps_depth)]
        self.msg = nn.ModuleList(msg_sequence)

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
                residual_attention = False, skip_connection = True,
                **kwargs):
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = self.time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups,
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout, 
                                        residual_attention = residual_attention,
                                        skip_connection = skip_connection)
        fp_sequence = [  FPBlock( in_channel_known = self.known_feature_channels[fid],
                                in_channel_unknown = self.unknow_feature_channels[fid],
                                out_channel = self.fp_path[fid + 1],
                                num_blocks = self.num_blocks,
                                BuildingBlock = self.BuildingBlock)
                            for fid in range(self.fp_depth)]
        self.fp = nn.ModuleList(fp_sequence)
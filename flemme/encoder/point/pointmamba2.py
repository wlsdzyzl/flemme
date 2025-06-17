import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import get_building_block, \
    SamplingAndGroupingLayer as MSGLayer, get_psmamba_block, get_scanners, \
    MultipleBuildingBlocks, FeaturePropogatingLayer as FPLayer
from .pointnet2 import Point2Encoder, Point2Decoder
from flemme.logger import get_logger
logger = get_logger("encoder.point.pointmamba2")
        
class PointMamba2Encoder(Point2Encoder):
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
            scan_strategies = None,
            flip_scan = False,
            normalization = 'group', num_norm_groups = 8, 
            activation = 'lrelu', dropout = 0., 
            vector_embedding = True, 
            state_channel = 64, 
            conv_kernel_size = 4, inner_factor = 2.0,  
            head_channel = 64,
            conv_bias=True, bias=False,
            learnable_init_states = True, chunk_size=256,
            dt_min=0.001, A_init_range=(1, 16),
            dt_max=0.1, dt_init_floor=1e-4, 
            dt_rank = None, dt_scale = 1.0,
            mlp_hidden_ratios=[4.0, 4.0], 
            is_point2decoder = False,
            long_range_modeling = False,
            z_count = 1, 
            return_xyz = False,
            last_activation = True,
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
                z_count = z_count, 
                vector_embedding = vector_embedding, 
                is_point2decoder = is_point2decoder,
                final_concat = final_concat,
                pos_embedding=pos_embedding,
                return_xyz = return_xyz,
                last_activation = last_activation,
                channel_attention = channel_attention,
                time_injection = time_injection,
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
                                        state_channel = state_channel, 
                                        conv_kernel_size = conv_kernel_size, 
                                        inner_factor = inner_factor,  
                                        head_channel = head_channel,
                                        conv_bias=conv_bias, bias=bias,
                                        chunk_size=chunk_size,
                                        dt_min=dt_min, A_init_range=A_init_range,
                                        dt_max=dt_max, dt_init_floor=dt_init_floor, 
                                        dt_rank = dt_rank, dt_scale = dt_scale,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)
        # print(self.msg_path)
        msg_sequence = [MSGLayer(in_channel = self.msg_path[fid], 
            out_channels = self.sub_out_channels[fid],
            num_fps_points = self.num_fps_points[fid],
            k = self.num_neighbors_k[fid],
            radius = self.neighbor_radius[fid],
            num_blocks = self.num_blocks,
            use_xyz = self.use_xyz,
            sorted_query = sorted_query,
            knn_query = self.knn_query,
            pos_embedding_channel = projection_channel if pos_embedding else point_dim,
            BuildingBlock = self.BuildingBlock) for fid in range(self.fps_depth)]
        self.msg = nn.ModuleList(msg_sequence)
        if long_range_modeling:
            ### scan mamba block
            self.scanners = get_scanners(scan_strategies)
            if len(self.scanners) > 0:
                self.flip_scan = flip_scan
                num_scan = len(self.scanners)
                if self.flip_scan: num_scan *= 2
                PSMambaBlock = get_psmamba_block(building_block, 
                                    time_channel = time_channel, 
                                    num_scan = num_scan,
                                    activation=activation, 
                                    norm = normalization, 
                                    num_norm_groups = num_norm_groups, 
                                    dropout = dropout,
                                    state_channel = state_channel, 
                                    conv_kernel_size = conv_kernel_size, 
                                    inner_factor = inner_factor,  
                                    head_channel = head_channel,
                                    conv_bias=conv_bias, bias=bias,
                                    chunk_size=chunk_size,
                                    learnable_init_states = learnable_init_states,
                                    dt_min=dt_min, A_init_range=A_init_range,
                                    dt_max=dt_max, dt_init_floor=dt_init_floor, 
                                    dt_rank = dt_rank, dt_scale = dt_scale,
                                    mlp_hidden_ratios = mlp_hidden_ratios, 
                                    time_injection = time_injection,
                                    condition_channel = condition_channel,
                                    condition_injection = condition_injection,
                                    condition_first = condition_first)
                lrm_sequence = [MultipleBuildingBlocks(in_channel = fps_feature_channels[fid], 
                    out_channels = fps_feature_channels[fid],
                    n = num_blocks,
                    BuildingBlock = PSMambaBlock) for fid in range(self.fps_depth)]
            else:
                lrm_sequence = [MultipleBuildingBlocks(in_channel = fps_feature_channels[fid], 
                    out_channels = fps_feature_channels[fid],
                    n = num_blocks,
                    BuildingBlock = self.BuildingBlock) for fid in range(self.fps_depth)]
            self.lrm = nn.ModuleList(lrm_sequence)
    def scan(self, xyz):
        if hasattr(self, 'scanners') and len(self.scanners) > 0:
            sorted_index_list = []
            for s in self.scanners:
                idx = s(xyz)
                sorted_index_list.append(idx)
            if self.flip_scan:
                sorted_index_list = sorted_index_list + [ torch.flip(idx, dims=[-1]) for idx in sorted_index_list]
            return sorted_index_list
        else:
            logger.error('No scanners.')
            exit(1)
class PointMamba2Decoder(Point2Decoder):
    def __init__(self, point_dim=3, point_num = 2048, 
                ### provide by encoder
                in_channels= [1024, 1024, 512, 256, 128], 
                time_channel = 0,
                building_block = 'pmamba', 
                num_blocks = 2,
                fp_channels = [512, 512, 256, 128], 
                dense_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                state_channel = 64, 
                conv_kernel_size = 4, inner_factor = 2.0,  
                head_channel = 64,
                conv_bias=True, bias=False,
                chunk_size=256,
                dt_min=0.001, A_init_range=(1, 16),
                dt_max=0.1, dt_init_floor=1e-4, 
                dt_rank = None, dt_scale = 1.0,
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
                time_injection = time_injection,
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
                                        state_channel = state_channel, 
                                        conv_kernel_size = conv_kernel_size, 
                                        inner_factor = inner_factor,  
                                        head_channel = head_channel,
                                        conv_bias=conv_bias, bias=bias,
                                        chunk_size=chunk_size,
                                        dt_min=dt_min, A_init_range=A_init_range,
                                        dt_max=dt_max, dt_init_floor=dt_init_floor, 
                                        dt_rank = dt_rank, dt_scale = dt_scale,
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
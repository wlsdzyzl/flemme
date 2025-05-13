### PointNet++: Use Group and Sampling for hierarchical feature learning.
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block, get_ca,\
    SamplingAndGroupingBlock as MSGBlock, FeaturePropogatingBlock as FPBlock, \
    SampledFeatureCatBlock as SFCBlock
from flemme.logger import get_logger
import copy
logger = get_logger("encoder.point.pointnet2")
## can be channel, radius, k.
def get_scale_channel(out_channel, num_scales):
    if num_scales == 1:
        return out_channel
    last_channel = out_channel
    res = []
    sum_channel = 0
    for i in range(num_scales - 1):
        c = int(last_channel // 2)
        res = res + [c,]
        last_channel = c
        sum_channel += c
    res = res + [out_channel - sum_channel, ]
    return res[::-1]
def get_scale_parameter(max_p, num_scales):
    t = type(max_p)
    if num_scales == 1: 
        return max_p
    res = [t(max_p / (2 ** i)) for i in range(num_scales)]
    return res[::-1]

class Point2Encoder(nn.Module):
    def __init__(self, point_dim,
                 projection_channel,
                 time_channel,
                 num_fps_points,
                 num_neighbors_k,
                 neighbor_radius, 
                 fps_feature_channels, 
                 num_blocks,
                 num_scales,
                 use_xyz,
                 sorted_query,
                 knn_query,
                 dense_channels,
                 activation, dropout,
                 normalization, num_norm_groups,  
                 z_count, vector_embedding, 
                 is_point2decoder,
                 return_xyz,
                 last_activation,
                 final_concat,
                 pos_embedding,
                 channel_attention,
                 time_injection,
                 **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.point_dim = point_dim
        self.z_count = z_count
        self.activation = activation
        self.num_neighbors_k = num_neighbors_k
        self.neighbor_radius = neighbor_radius
        self.num_scales = num_scales
        self.dropout = dropout
        self.normalization = normalization
        self.num_norm_groups = num_norm_groups
        self.point_proj = nn.Linear(point_dim, projection_channel)
        self.projection_channel = projection_channel
        self.time_channel = time_channel
        self.num_fps_points = num_fps_points
        self.num_blocks = num_blocks
        # fps_depth
        self.fps_depth = len(fps_feature_channels)
        self.is_point2decoder = is_point2decoder
        self.use_xyz = use_xyz
        self.sorted_query = sorted_query
        self.knn_query = knn_query
        if knn_query:
            assert knn_query in ['xyz', 'xyz_embed', 'feature'], "Unsupported KNN query space, shouled be one of ['xyz', 'xyz_embed', 'feature']."
            logger.info(f'Perform KNN query on {knn_query} space.')
        else:
            logger.info(f'Perform ball query on xyz space.')
        self.return_xyz = return_xyz
        if pos_embedding:
            self.pos_embed = nn.Linear(point_dim, projection_channel)
            logger.info("Using point cloud positional embedding.")
        if not type(self.num_fps_points) == list:
            self.num_fps_points = [self.num_fps_points,] * len(fps_feature_channels)

        if not type(self.num_neighbors_k) == list: 
            self.num_neighbors_k = [self.num_neighbors_k, ] * len(fps_feature_channels)

        if not type(self.neighbor_radius) == list:
            self.neighbor_radius = [self.neighbor_radius, ] * len(fps_feature_channels)
        
        if not type(self.num_scales) == list:
            self.num_scales = [self.num_scales, ] * len(fps_feature_channels)
        

        assert len(fps_feature_channels) == len(self.num_fps_points) and \
          len(fps_feature_channels) == len(self.num_neighbors_k) and \
          len(fps_feature_channels) == len(self.neighbor_radius) and \
          len(fps_feature_channels) == len(self.num_scales) 
        self.sub_out_channels = []
        for did in range(self.fps_depth):
            self.sub_out_channels.append(get_scale_channel(fps_feature_channels[did], self.num_scales[did]))
            if not type(self.num_neighbors_k[did]) == list:
                self.num_neighbors_k[did] = get_scale_parameter(self.num_neighbors_k[did], self.num_scales[did])
            if not type(self.neighbor_radius[did]) == list:
                self.neighbor_radius[did] = get_scale_parameter(self.neighbor_radius[did], self.num_scales[did])

            # assert self.sub_out_channels[did] == self.num_scales[did] and \
            #     self.num_neighbors_k[did] == self.num_scales[did] and \
            #     self.neighbor_radius[did] = self.num_scales[did], "Number of scales inferred from different parameters are not unmatched."

        self.msg_path = [projection_channel,] + fps_feature_channels
        self.msg = None

        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = 1, channel = self.msg_path[i+1], channel_dim = -1, **channel_attention) 
                           for i in range(len(self.msg_path) - 1)]
            self.ca = nn.ModuleList(ca_sequence)

        self.vector_embedding = vector_embedding
        assert len(dense_channels) > 0, "Point2 encoder need to have fully connected layers!"
        dense_channels = [fps_feature_channels[-1] * 2, ] + dense_channels
        if not self.vector_embedding:
            dense_channels[0] += fps_feature_channels[-1]
        if last_activation:
            dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1],  
                                                time_channel = self.time_channel,
                                                activation = self.activation, dropout=self.dropout, 
                                                norm = normalization, num_norm_groups=num_norm_groups,
                                                time_injection=time_injection) for i in range(len(dense_channels) - 1)]
        else:
            dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1],  
                                                time_channel = self.time_channel,
                                                activation = self.activation, dropout=self.dropout, 
                                                norm = normalization, num_norm_groups=num_norm_groups,
                                                time_injection=time_injection) for i in range(len(dense_channels) - 2)]
            # the last layer is a linear layer, without batch normalization
            dense_sequence = dense_sequence + [DenseBlock(dense_channels[-2], dense_channels[-1], 
                                        time_channel = self.time_channel,
                                        time_injection = time_injection,
                                        activation = None, norm = None), ]
        self.dense = nn.ModuleList([SequentialT(* (copy.deepcopy(dense_sequence)) ) for _ in range(z_count) ])
        self.out_channel = dense_channels[-1]
        self.dense_path = dense_channels
        self.out_channels = self.msg_path + [self.out_channel, ]
        if final_concat:
            logger.info('This encoder will process a final concatenation of the sampled features from all previous MSG layers')
            self.final_concat = SFCBlock(in_channels = self.msg_path, out_channel = fps_feature_channels[-1],
                                        num_blocks = self.num_blocks, time_channel = self.time_channel,
                                        activation = self.activation, dropout=self.dropout, 
                                        norm = normalization, num_norm_groups=num_norm_groups,
                                        time_injection=time_injection)
        if time_channel > 0:
            logger.info(f'Using time-step injection method: {time_injection}')
    def forward(self, xyz, t = None):
        if self.msg is None:
            raise NotImplementedError
        B, _, _ = xyz.shape
        ## N * Np * d
        features = self.point_proj(xyz)
        if hasattr(self, 'pos_embed'):
            xyz_embed = self.pos_embed(xyz)
        else:
            xyz_embed = xyz
        xyz = xyz[...,0:3]
        xyz_list, feature_list = [xyz], [features]
        sample_id_list = []
        for lid, msg in enumerate(self.msg):
            xyz, xyz_embed, features, sample_ids = msg(xyz, xyz_embed, features = features, t = t)
            if hasattr(self, 'lrm'):
                if hasattr(self, 'scanners') and len(self.scanners) > 0: 
                    sorted_index_list = self.scan(xyz)
                    features = self.lrm[lid](features, (sorted_index_list, t))
                else:
                    features = self.lrm[lid](features, t)
            if hasattr(self, 'ca'):
                self.ca[lid](features)

            xyz_list.append(xyz)
            feature_list.append(features)
            sample_id_list.append(sample_ids)
        if hasattr(self, 'final_concat'):
            features = self.final_concat(feature_list, sample_id_list, t)
        ## max and average pooling to get global feature
        f_max = F.adaptive_max_pool1d(features.transpose(1, 2), 1).transpose(1, 2)
        f_avg = F.adaptive_avg_pool1d(features.transpose(1, 2), 1).transpose(1, 2)
        global_features = torch.concat((f_max, f_avg), dim = -1)

        ## pointwise 
        if not self.vector_embedding:
            # B * D -> B * N * D
            # local feature plus global feature
            global_features = global_features.repeat(1, self.num_fps_points[-1], 1)
            features = torch.concat([global_features, features], dim=-1)
        else:
            xyz = None
            features = global_features.reshape(B, -1)
        
        ## compute latent embeddings
        features = [self.dense[i](features, t) for i in range(self.z_count)]
        if self.z_count == 1:
            features = features[0]

        if self.is_point2decoder:
            xyz_list.append(xyz)
            feature_list.append(features)
            return feature_list, xyz_list
        if self.return_xyz:
            return features, xyz
        return features

    def __str__(self):
        _str = f'projection layer: {self.point_dim}->{self.projection_channel}\n'
        _str += 'Furthest Point Sampling and Grouping layers:'
        for c in self.msg_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.msg_path[-1])
        _str += '\n'
        ## print fc layer
        _str = _str + 'Dense layers: '
        for c in self.dense_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.dense_path[-1])
        _str += '\n'
        return _str 

class Point2Decoder(nn.Module):
    def __init__(self, point_dim, point_num, 
                ### provide by encoder
                in_channels, time_channel,
                num_blocks,
                fp_channels, 
                dense_channels, normalization, 
                num_norm_groups, activation, dropout, 
                channel_attention,
                time_injection,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_num = point_num
        self.point_dim = point_dim
        self.activation = activation
        self.time_channel = time_channel
        # self.vector_embedding = False
        fp_channels = [in_channels[-1], ] + fp_channels
        self.fp_depth = len(fp_channels)
        self.fp_path = [in_channels[-1], ] + fp_channels
        self.num_blocks = num_blocks

        ## fully connected layer
        dense_channels = [fp_channels[-1],] + dense_channels 
        dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                        time_channel = self.time_channel,
                                        norm = normalization, num_norm_groups=num_norm_groups, 
                                        activation = activation, dropout=dropout,
                                        time_injection=time_injection) for i in range(len(dense_channels) - 1)]
        self.dense = SequentialT(*dense_sequence) 
        self.dense_path = dense_channels
        self.final = nn.Linear(dense_channels[-1], point_dim)
        self.fp = None
        assert len(in_channels) == len(fp_channels) + 1, 'The number of feature propagation layers is ambiguous .'
        self.unknow_feature_channels = in_channels[-2:-len(in_channels) - 1:-1]
        self.known_feature_channels = [in_channels[-1],] + fp_channels[:-1]

        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = 1, channel = self.fp_path[fid + 1], channel_dim = -1, **channel_attention) 
                           for fid in range(self.fp_depth)]
            self.ca = nn.ModuleList(ca_sequence)
        if time_channel > 0:
            logger.info(f'Using time-step injection method: {time_injection}')
    def forward(self, features_xyz, t = None):
        feature_list, xyz_list = features_xyz
        if self.fp is None:
            raise NotImplementedError
        assert self.point_num == xyz_list[0].shape[1], 'Unmatched point cloud size.'
        feature_list = feature_list[::-1]
        xyz_list = xyz_list[::-1]
        for i in range(1, len(feature_list)):
            feature_list[i] = self.fp[i-1]( xyz_list[i], xyz_list[i-1], feature_list[i], feature_list[i-1], t = t)
        feature = self.dense(feature_list[-1], t)
        return self.final(feature)
    def __str__(self):
        _str = ''
        _str += 'Feature Propagating layers:'
        for c in self.fp_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.fp_path[-1])
        _str += '\n'
        _str = _str + 'Dense layers: '
        for c in self.dense_path:
            _str += '{}->'.format(c)  
        _str += f'{self.point_dim}'
        return _str 

class PointNet2Encoder(Point2Encoder):
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
                 vector_embedding = True, 
                 is_point2decoder = False,
                 z_count = 1, 
                 return_xyz = False,
                 last_activation = True,
                 final_concat = False,
                 pos_embedding = False,
                 channel_attention = None,
                 time_injection = 'gate_bias',
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
                return_xyz = return_xyz, 
                last_activation = last_activation,
                final_concat = final_concat,
                pos_embedding=pos_embedding,
                channel_attention = channel_attention,
                time_injection=time_injection)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = self.time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        time_injection = time_injection)
        msg_sequence = [MSGBlock(in_channel = self.msg_path[fid], 
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
        # if final_concat:
            
        #     self.final_concat = SFCBlock(in_channels = self.msg_path, out_channel = fps_feature_channels[-1],
        #     num_blocks = self.num_blocks, BuildingBlock = self.BuildingBlock)

class PointNet2Decoder(Point2Decoder):
    def __init__(self, point_dim=3, point_num = 2048, 
                ### provide by encoder
                in_channels= [1024, 1024, 512, 256, 128], 
                time_channel = 0,
                num_blocks = 2,
                fp_channels = [512, 512, 256, 128], 
                dense_channels = [],
                building_block = 'dense',  
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                channel_attention = None,
                time_injection = 'gate_bias',
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
                time_injection=time_injection)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = self.time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups,
                                        dropout = dropout,
                                        time_injection = time_injection)
        fp_sequence = [  FPBlock( in_channel_known = self.known_feature_channels[fid],
                                in_channel_unknown = self.unknow_feature_channels[fid],
                                out_channel = self.fp_path[fid + 1],
                                num_blocks = self.num_blocks,
                                BuildingBlock = self.BuildingBlock)
                            for fid in range(self.fp_depth)]
        self.fp = nn.ModuleList(fp_sequence)

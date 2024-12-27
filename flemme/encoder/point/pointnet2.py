### PointNet++: Use Group and Sampling for hierarchical feature learning.
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block, FoldingLayer, LocalGraphLayer
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
        c = last_channel / 2
        res = res + [c,]
        last_channel = c
        sum_channel += c
    res = res + [out_channel - sum_channel, ]
    return res[::-1]
def get_scale_parameter(max_p, num_scales):
    if num_scales == 1: 
        return max_p
    res = [ max_p / (2 ** i) for i in range(num_scales)]
    return res[::-1]

class Point2Encoder(nn.Module):
    def __init__(self, point_dim,
                 projection_channel,
                 time_channel,
                 num_fps_points,
                 num_neighbors_k,
                 neighbor_radius, 
                 fps_feature_channels, 
                 num_block,
                 num_scales,
                 use_xyz,
                 hidden_channels,
                 dense_channels,
                 activation, dropout,
                 normalization, num_groups,  
                 z_count, vector_embedding, 
                 return_feature_list,
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
        self.num_groups = num_groups
        self.point_proj = nn.Linear(point_dim, projection_channel)
        self.projection_channel = projection_channel
        self.time_channel = time_channel
        self.num_fps_points = num_fps_points
        # fps_depth
        self.fps_depth = len(fps_feature_channels)
        self.return_feature_list = return_feature_list
        if not type(self.num_fps_points, list):
            self.num_fps_points = [self.num_fps_points,] * len(fps_feature_channels)

        if not type(self.num_neighbors_k, list): 
            self.num_neighbors_k = [self.num_neighbors_k, ] * len(fps_feature_channels)

        if not type(self.neighbor_radius, list):
            self.neighbor_radius = [self.neighbor_radius, ] * len(fps_feature_channels)
        
        if not type(self.num_scales, list):
            self.num_scales = [self.num_scales, ] * len(fps_feature_channels)
        
        if not type(num_block) == list:
            num_block = [num_block, ] * len(fps_feature_channels)

        ### using hidden_channels is not reconmmended
        if not (isinstance(hidden_channels, list) and \
            len(hidden_channels) > 0 and  \
            isinstance(hidden_channels[0], list)):
            hidden_channels = [hidden_channels, ] * len(fps_feature_channels)

        assert len(fps_feature_channels) == len(self.num_fps_points) and \
          len(fps_feature_channels) == len(self.num_neighbors_k) and \
          len(fps_feature_channels) == len(self.neighbor_radius) and \
          len(fps_feature_channels) == len(self.num_scales) and \
          len(fps_feature_channels) == len(num_block) and \
          len(fps_feature_channels) == len(hidden_channels), 'The numbers of layers inferred from different parameters are not identical.'
        self.sub_out_channels = []
        for did in range(self.fps_depth):
            self.sub_out_channels.append(get_scale_channel(fps_feature_channels[did]), self.num_scales[did])
            if not type(self.num_neighbors_k[did]) == list:
                self.num_neighbors_k[did] = get_scale_parameter(self.num_neighbors_k[did], self.num_scales[did])
            if not type(self.neighbor_radius[did]) == list:
                self.neighbor_radius[did] = get_scale_parameter(self.neighbor_radius[did], self.num_scales[did])

            # assert self.sub_out_channels[did] == self.num_scales[did] and \
            #     self.num_neighbors_k[did] == self.num_scales[did] and \
            #     self.neighbor_radius[did] = self.num_scales[did], "Number of scales inferred from different parameters are not unmatched."

        self.msg_path = [projection_channel,] + fps_feature_channels
        self.msg = None
        self.vector_embedding = vector_embedding
        assert len(dense_channels) > 0, "Point2 encoder need to have fully connected layers!"
        dense_channels = [fps_feature_channels[-1] * 2, ] + dense_channels
        if not self.vector_embedding:
            dense_channels[0] += fps_feature_channels[-1]
        dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1],  
                                            time_channel = self.time_channel,
                                            activation = self.activation, dropout=self.dropout, 
                                            norm = normalization, num_groups=num_groups) for i in range(len(dense_channels) - 2)]
        # the last layer is a linear layer, without batch normalization
        dense_sequence = dense_sequence + [DenseBlock(dense_channels[-2], dense_channels[-1], 
                                    time_channel = self.time_channel,
                                    activation = None, norm = None), ]
        self.dense = nn.ModuleList([SequentialT(* (copy.deepcopy(dense_sequence)) ) for _ in range(z_count) ])
        self.out_channel = dense_channels[-1]
        self.dense_path = dense_channels
    def forward(self, xyz, t = None):
        if self.msg is None:
            raise NotImplementedError
        B, N, _ = xyz.shape
        ## N * Np * d
        res = []
        feature = self.point_proj(xyz)
        xyz = xyz[...,0:3]
        xyz_list, feature_list = [xyz], [feature]
        for msg in self.msg:
            xyz, feature = msg(xyz, feature = feature, t = t)
            xyz_list.append(xyz)
            feature_list.append(feature)

        ## max and average pooling to get global feature
        f_max = F.adaptive_max_pool1d(feature.transpose(1, 2), 1).transpose(1, 2)
        f_avg = F.adaptive_avg_pool1d(feature.transpose(1, 2), 1).transpose(1, 2)
        feature_global = torch.concat((x1, x2), dim = -1)

        ## pointwise 
        if not self.vector_embedding:
            # B * D -> B * N * D
            # local feature plus global feature
            feature_global = feature_global.repeat(1, N, 1)
            feature = torch.concat([feature_global, feature], dim=-1)
        else:
            xyz = None
            feature = feature_global.reshape(B, -1)
        
        ## compute latent embeddings
        feature = [self.dense[i](feature, t)[0] for i in range(self.z_count)]
        if self.z_count == 1:
            feature = feature[0]

        ### for pointnet2 decoder
        if self.return_feature_list:
            xyz_list.append(xyz)
            feature_list.append(feature)
            return feature_list, xyz_list

        return feature, xyz

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
                num_block,
                hidden_channels,
                fp_channels, 
                dense_channels, normalization, 
                num_groups, activation, dropout, 
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_num = point_num
        self.point_dim = point_dim
        self.activation = activation
        self.time_channel = time_channel
        self.vector_embedding = False
        self.fp_channels = fp_channels
        ## fully connected layer
        dense_channels = [fp_channels[-1],] + dense_channels 
        dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                        time_channel = self.time_channel,
                                        norm = normalization, num_groups=num_groups, 
                                        activation = activation, dropout=dropout) for i in range(len(dense_channels) - 1)]
        self.dense = SequentialT(*dense_sequence) 
        self.dense_path = dense_channels
        self.final = nn.Linear(dense_channels[-1], point_dim)
        self.fp = None
        assert len(in_channels) == len(fp_channels) + 1, 'The number of feature propagation layers is ambiguous .'
        self.unknow_feature_channels = in_channels[1:]
        self.known_feature_channels = [in_channels[0],] + fp_channels[:-1]
    def forward(self, feature_list, xyz_list, t = None):
        if self.fp is None:
            raise NotImplementedError
        assert self.point_num == xyz_list[0].shape[1], 'Unmatched point cloud size.'
        
        for i in range(len(feature_list) - 2, -1, -1):
            feature_list[i] = self.fp[i]( xyz_list[i], xyz_list[i+1], feature_list[i], feature_list[i+1], t = t)
        feature, _ = self.dense(feature_list[0], t)
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

def PointNet2Encoder():

def PointNet2Decoder():
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block
from flemme.logger import get_logger
from flemme.logger import 
from torch_geometric.utils import scatter
import copy
logger = get_logger("model.encoder.graph_neural_network")
class GraphEncoder(nn.Module):
    def __init__(self, node_dim=0, pos_dim=3, 
                 # concatenate position and node features
                 message_components=['pos', 'feature'],
                 ### wait to be implemented
                 node_projection_channel = 64,
                 message_passing_channels = [64, 64, 128, 256], 
                 dense_channels = [],
                 activation = 'lrelu', dropout = 0.,
                 normalization = 'group', num_groups = 8,  
                 z_count = 1, nodewise = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.node_dim = node_dim 
        self.pos_dim = pos_dim
        self.node_projection_channel = node_projection_channel
        self.activation = activation
        self.dropout = dropout
        self.nodewise = nodewise
        self.normalization = normalization
        self.num_groups = num_groups
        self.message_components = message_components
        node_feature_dim = 0
        if 'pos' in message_components:
            node_feature_dim += pos_dim
        if 'feature' in message_components:
            node_feature_dim += node_dim
        assert len(node_feature_dim) > 0, 'message waited to pass is empty, please check if the graph has used the non-empty features.'

        self.node_proj = nn.Linear(node_feature_dim, node_projection_channel)
        assert len(message_passing_channels) > 1, 'Graph encoder needs more than one message pasing channels.'
        dense_channels = [message_passing_channels[-1], ] + dense_channels
        if len(dense_channels) > 0:
            if self.nodewise:
                dense_channels[0] += message_passing_channels[-1]
            dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1],  
                                                activation = self.activation, dropout=self.dropout, 
                                                norm = normalization, num_groups=num_groups) for i in range(len(dense_channels) - 2)]
            # the last layer is a linear layer, without batch normalization
            dense_sequence = dense_sequence + [DenseBlock(dense_channels[-2], dense_channels[-1], activation = None, norm = None), ]
            self.dense = nn.ModuleList([nn.Sequential(* (copy.deepcopy(dense_sequence)) ) for _ in range(z_count) ])
        else:
            message_passing_channels = [m * self.z_count for m in message_passing_channels]
        self.out_channel = dense_channels[-1]
        self.dense_path = dense_channels
        self.mp_path = [node_projection_channel,] + message_passing_channels
        self.mp = None
    ### similar to point cloud
    def forward(self, data, t = None):
        if self.mp is None:
            raise NotImplementedError

        feature, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = None
        if 'pos' in self.message_components:
            x = pos
        if 'feature' in self.message_components and feature is not None:
            x = torch.concat((x, feature), dim = -1) if x is not None else feature
        # transfer to Nb * d * Np
        res = []
        x = self.node_proj(x)
        for mp in self.mp[:-1]:
            x = mp(x, edge_index, t)
            res.append(x)
        x = torch.concat(res, dim=-1)
        nf = self.mp[-1](x, edge_index)
        if self.dense is not None:
            ## max and average pooling to get global feature
            x1 = scatter(nf, batch, dim=0, reduce='max')
            x2 = scatter(nf, batch, dim=0, reduce='mean')
            # global feature
            x = torch.concat((x1, x2), dim = -1)

            if self.nodewise:
                # B * D -> BN * D
                x = x[batch]
                x = torch.concat([x, nf], dim=-1)

            ## compute embedding vectors
            x = [self.dense[i](x) for i in range(self.z_count)]
        else:
            ## split nf to 
            x = torch.chunk(x, self.z_count, dim = -1)
        if self.z_count == 1:
            x = x[0]
        return x

    def __str__(self):
        _str = f'node projection layer: {self.node_dim}->{self.projection_channel}\n'
        _str += 'Message passing layers:'
        for c in self.mp_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.mp_path[-1])
        _str += '\n'
        ## print fc layer
        if self.dense is not None:
            _str = _str + 'Dense layers: '
            for c in self.dense_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.dense_path[-1])
            _str += '\n'
        return _str 



class GraphDecoder(nn.Module):
    def __init__(self, node_dim=0, pos_dim=3, node_num = 2048, 
                in_channel = 256, dense_channels = [], 
                normalization = 'group', num_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                ## should be subset of ['feature', 'pos', 'edge']
                recon_components = ['pos'],
                nodewise = False, 
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.node_dim = node_dim
        self.pos_dim = pos_dim
        self.node_num = node_num

        self.activation = activation
        self.nodewise = nodewise
        self.de_pos = None
        self.de_feature = None
        self.de_edge = None
        dense_channels = [in_channel,] + dense_channels 
        dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                        norm = normalization, num_groups=num_groups, 
                                        activation = activation, dropout=dropout) for i in range(len(dense_channels) - 1)]

        ## fully connected layer
        if 'pos' in recon_components and pos_dim > 0:
            final_channel = pos_dim * node_num if not self.nodewise else self.pos_dim
            de_pos_sequence = dense_sequence + [DenseBlock(dense_channels[-1], final_channel, 
                            activation = None, norm = None)]
            self.de_pos = nn.Sequential(*de_pos_sequence) 
            self.de_pos_path = dense_channels + [final_channel, ]
        if 'feature' in recon_components and node_dim > 0:
            final_channel = node_dim * node_num if not self.nodewise else self.node_dim
            de_feature_sequence = dense_sequence + [DenseBlock(dense_channels[-1], final_channel, 
                            activation = None, norm = None)]
            self.de_feature = nn.Sequential(*de_feature_sequence) 
            self.de_feature_path = dense_channels + [final_channel, ]
        if 'edge' in recon_components:
            if not nodewise:
                logger.error('Inner product decoder needs nodewise latent features.')
                exit(1)
            de_edge_sequence = dense_sequence + [DenseBlock(dense_channels[-1], dense_channels[-1], 
                            activation = None, norm = None)]
            de_edge_sequence.append(InnerProductDecoder())
            self.de_edge_path = dense_channels + [dense_channels[-1] ]
            self.de_edge = nn.Sequential(*de_edge_sequence) 
            ## inner product decoder
    def __str__(self):
        _str = ''
        if self.de_pos is not None:
            _str = _str + 'Dense layers for pos: '
            for c in self.de_pos_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.de_pos_path[-1])
            if self.dense_path[-1] != self.pos_dim:
                _str += f' ({self.node_num} * {self.pos_dim})'
            _str += '\n'
        if self.de_feature is not None:
            _str = _str + 'Dense layers for feature: '
            for c in self.de_feature_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.de_feature_path[-1])
            if self.dense_path[-1] != self.node_dim:
                _str += f' ({self.node_num} * {self.node_dim})'
            _str += '\n'
        if self.de_edge is not None:
            _str = _str + 'Dense layers for edge: '
            for c in self.de_edge_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.de_edge_path[-1])
        return _str 
    def forward(self, x, t = None):
        
        if self.de_pos is not None:
            pos = self.de_pos(x)
            if not self.nodewise:
                pos = pos.reshape(-1, self.pos_dim)
        if self.de_feature is not None:
            feature = self.de_feature(x)
            if not self.nodewise:
                feature = feature.reshape(-1, self.node_dim)
        if self.de_edge is not None:
            edge = self.de_edge(x)

        return pos, feature, 




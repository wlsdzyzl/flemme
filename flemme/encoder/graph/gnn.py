import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block, InnerProductBlock
from flemme.logger import get_logger
from torch_geometric.utils import scatter
import copy
logger = get_logger("model.encoder.graph_neural_network")
class GraphEncoder(nn.Module):
    def __init__(self, pos_dim=3, node_dim=0, 
                    time_channel = 0,
                    # concatenate position and node features
                    pass_pos = True,
                    pass_feature = True,
                    ### wait to be implemented
                    projection_channel = 64,
                    message_passing_channels = [64, 64, 128, 256], 
                    dense_channels = [],
                    activation = 'lrelu', dropout = 0.,
                    normalization = 'group', num_norm_groups = 8,  
                    nodewise = False, 
                    time_injection = 'gate_bias', 
                    condition_channel = 0,
                    condition_injection = 'gate_bias',
                    condition_first = False,
                    **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.node_dim = node_dim 
        self.pos_dim = pos_dim
        self.projection_channel = projection_channel
        self.activation = activation
        self.dropout = dropout
        self.nodewise = nodewise
        self.normalization = normalization
        self.num_norm_groups = num_norm_groups
        self.pass_pos = pass_pos
        self.pass_feature = pass_feature
        # self.time_channel = time_channel
        # self.condition_channel = condition_channel
        node_feature_dim = 0
        if self.pass_pos:
            node_feature_dim += pos_dim
        if self.pass_feature:
            node_feature_dim += node_dim
        assert len(node_feature_dim) > 0, 'message waited to pass is empty, please check if the graph has used the non-empty features.'

        self.node_proj = nn.Linear(node_feature_dim, projection_channel)
        assert len(message_passing_channels) > 1, 'Graph encoder needs more than one message pasing channels.'
        dense_channels = [message_passing_channels[-1], ] + dense_channels
        self.mp_path = [projection_channel,] + message_passing_channels
        if len(dense_channels) > 1:
            if self.nodewise:
                dense_channels[0] += message_passing_channels[-1]
            dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1],  
                                                time_channel = time_channel,
                                                time_injection = time_injection,
                                                activation = self.activation, dropout=self.dropout, 
                                                norm = normalization, num_norm_groups=num_norm_groups,
                                                condition_channel = condition_channel,
                                                condition_injection = condition_injection,
                                                condition_first = condition_first) 
                                                for i in range(len(dense_channels) - 2)]
            # the last layer is a linear layer, without batch normalization
            dense_sequence = dense_sequence + [DenseBlock(dense_channels[-2], dense_channels[-1], activation = None, norm = None), ]
            self.dense = SequentialT(* (copy.deepcopy(dense_sequence)) ) 

        self.out_channel = dense_channels[-1]
        self.dense_path = dense_channels
        self.mp = None
    ### similar to point cloud
    def forward(self, data, t = None):
        if self.mp is None:
            raise NotImplementedError

        feature, pos, edge_index, batch = data.x, data.pos, data.edge_index, data.batch
        x = None
        if self.pass_pos:
            x = pos
        if self.pass_feature and feature is not None:
            x = torch.concat((x, feature), dim = -1) if x is not None else feature
        # transfer to Nb * d * Np
        res = []
        x = self.node_proj(x)
        for mp in self.mp[:-1]:
            x = mp(x, edge_index, t, c)
            res.append(x)
        x = torch.concat(res, dim=-1)
        nf = self.mp[-1](x, edge_index)
        if hasattr(self, 'dense'):
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
            x = self.dense(x, t, c) 

        return x

    def __str__(self):
        _str = f'node projection layer: {self.node_dim}->{self.projection_channel}\n'
        _str += 'Message passing layers: '
        for c in self.mp_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.mp_path[-1])
        _str += '\n'
        if hasattr(self, 'dense'):
            _str = _str + 'Dense layers: '
            for c in self.dense_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.dense_path[-1])
            _str += '\n'
        return _str 

class GCNEncoder(GraphEncoder):
    def __init__(self, node_dim=0, pos_dim=3, 
                # GCN parameters
                graph_normalize = True,
                improved = False,
                cached = False,  
                bias = True,
                # concatenate position and node features
                building_block = 'gcn',
                pass_pos = True,
                pass_feature = True,
                projection_channel = 64,
                message_passing_channels = [64, 64, 128, 256], 
                dense_channels = [],
                activation = 'lrelu', dropout = 0.,
                normalization = 'group', num_norm_groups = 8,  
                nodewise = False, 
                time_channel = 0, 
                time_injection = 'gate_bias', 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(pos_dim=pos_dim, node_dim=node_dim, 
                    time_channel = time_channel,
                    # concatenate position and node features
                    pass_pos = pass_pos,
                    pass_feature = pass_feature,
                    ### wait to be implemented
                    projection_channel = projection_channel,
                    message_passing_channels = message_passing_channels, 
                    dense_channels = dense_channels,
                    activation = activation, dropout = dropout,
                    normalization = normalization, num_norm_groups = num_norm_groups,  
                    nodewise = nodewise, 
                    time_injection = time_injection, 
                    condition_channel = condition_channel,
                    condition_injection = condition_injection,
                    condition_first = condition_first,)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups, 
                                        graph_normalize = graph_normalize,
                                        improved = improved,
                                        cached = cached,
                                        bias = bias,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)
        mp_sequence = [self.BuildingBlock(in_channel=self.mp_path[i], 
                                        out_channel=self.mp_path[i+1]) for i in range(len(self.mp_path) - 2) ]
            
        mp_sequence.append(self.BuildingBlock(in_channel=sum(self.mp_path[1:-1]), 
                                        out_channel=self.mp_path[-1]))

        self.mp = nn.ModuleList(mp_sequence)

class ChebEncoder(GraphEncoder):
    def __init__(self, node_dim=0, pos_dim=3, 
                # GCN parameters
                filter_size = 5,
                graph_normalization = 'sym', 
                bias = True,
                building_block = 'cheb',
                # concatenate position and node features
                pass_pos = True,
                pass_feature = True,
                projection_channel = 64,
                message_passing_channels = [64, 64, 128, 256], 
                dense_channels = [],
                activation = 'lrelu', dropout = 0.,
                normalization = 'group', num_norm_groups = 8,  
                nodewise = False, 
                time_channel = 0, 
                time_injection = 'gate_bias', 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(pos_dim=pos_dim, node_dim=node_dim, 
                    time_channel = time_channel,
                    # concatenate position and node features
                    pass_pos = pass_pos,
                    pass_feature = pass_feature,
                    ### wait to be implemented
                    projection_channel = projection_channel,
                    message_passing_channels = message_passing_channels, 
                    dense_channels = dense_channels,
                    activation = activation, dropout = dropout,
                    normalization = normalization, num_norm_groups = num_norm_groups,  
                    nodewise = nodewise, 
                    time_injection = time_injection, 
                    condition_channel = condition_channel,
                    condition_injection = condition_injection,
                    condition_first = condition_first,)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_norm_groups = num_norm_groups, 
                                        filter_size = filter_size,
                                        graph_normalization = graph_normalization,
                                        bias = bias,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)
        mp_sequence = [self.BuildingBlock(in_channel=self.mp_path[i], 
                                        out_channel=self.mp_path[i+1]) for i in range(len(self.mp_path) - 2) ]
            
        mp_sequence.append(self.BuildingBlock(in_channel=sum(self.mp_path[1:-1]), 
                                        out_channel=self.mp_path[-1]))

        self.mp = nn.ModuleList(mp_sequence)

class TransConvEncoder(GraphEncoder):
    def __init__(self, node_dim=0, pos_dim=3, 
                # GCN parameters
                num_heads = 1,
                concat = True,
                beta = False,
                bias = True,
                # concatenate position and node features
                building_block = 'gtrans',
                pass_pos = True,
                pass_feature = True,
                projection_channel = 64,
                message_passing_channels = [64, 64, 128, 256], 
                dense_channels = [],
                activation = 'lrelu', dropout = 0.,
                normalization = 'group', num_norm_groups = 8,  
                nodewise = False, 
                time_channel = 0, 
                time_injection = 'gate_bias', 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__(pos_dim=pos_dim, node_dim=node_dim, 
                    time_channel = time_channel,
                    # concatenate position and node features
                    pass_pos = pass_pos,
                    pass_feature = pass_feature,
                    ### wait to be implemented
                    projection_channel = projection_channel,
                    message_passing_channels = message_passing_channels, 
                    dense_channels = dense_channels,
                    activation = activation, dropout = dropout,
                    normalization = normalization, num_norm_groups = num_norm_groups,  
                    nodewise = nodewise, 
                    time_injection = time_injection, 
                    condition_channel = condition_channel,
                    condition_injection = condition_injection,
                    condition_first = condition_first,)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_heads = num_heads,
                                        concat = concat,
                                        beta = beta,
                                        dropout = self.dropout,
                                        bias = bias,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)
        mp_sequence = [self.BuildingBlock(in_channel=self.mp_path[i], 
                                        out_channel=self.mp_path[i+1]) for i in range(len(self.mp_path) - 2) ]
            
        mp_sequence.append(self.BuildingBlock(in_channel=sum(self.mp_path[1:-1]), 
                                        out_channel=self.mp_path[-1]))

        self.mp = nn.ModuleList(mp_sequence)


class GraphDecoder(nn.Module):
    def __init__(self, pos_dim=3, node_dim=0, node_num = 2048, 
                in_channel = 256, time_channel = 0, 
                dense_channels = [], 
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                recon_pos = True,
                recon_feature = False,
                recon_edge = False,
                nodewise = False, 
                time_injection = 'gate_bias', 
                condition_channel = 0,
                condition_injection = 'gate_bias',
                condition_first = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.node_dim = node_dim
        self.pos_dim = pos_dim
        self.node_num = node_num
        self.activation = activation
        self.nodewise = nodewise
        dense_channels = [in_channel,] + dense_channels 
        dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                        time_channel = time_channel,
                                        norm = normalization, num_norm_groups=num_norm_groups, 
                                        activation = activation, dropout=dropout,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first=condition_first) for i in range(len(dense_channels) - 1)]
        ## fully connected layer
        if recon_pos and pos_dim > 0:
            final_channel = pos_dim * node_num if not self.nodewise else self.pos_dim
            de_pos_sequence = dense_sequence + [DenseBlock(dense_channels[-1], final_channel, 
                            activation = None, norm = None)]
            self.de_pos = SequentialT(*de_pos_sequence) 
            self.de_pos_path = dense_channels + [final_channel, ]
        if recon_feature and node_dim > 0:
            final_channel = node_dim * node_num if not self.nodewise else self.node_dim
            de_feature_sequence = dense_sequence + [DenseBlock(dense_channels[-1], final_channel, 
                            activation = None, norm = None)]
            self.de_feature = SequentialT(*de_feature_sequence) 
            self.de_feature_path = dense_channels + [final_channel, ]
        if recon_edge:
            ## inner product decoder
            if not nodewise:
                logger.error('Inner product decoder needs nodewise latent features to reconstruct edges.')
                exit(1)
            de_edge_sequence = dense_sequence + [DenseBlock(dense_channels[-1], dense_channels[-1], 
                            activation = None, norm = None)]
            self.de_edge_path = dense_channels + [dense_channels[-1] ]
            self.de_edge = SequentialT(*de_edge_sequence) 
            self.inner_p = InnerProductBlock()
            ### edge indices for batch graph
            x, y = torch.meshgrid(torch.arange(self.node_num), torch.arange(self.node_num), indexing = ij)
            # 2, (node_num^2)
            self.edge_index = torch.stack((x.flatten(), y.flatten()))

    def __str__(self):
        _str = ''
        if hasattr(self, 'de_pos'):
            _str = _str + 'Dense layers for pos: '
            for c in self.de_pos_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.de_pos_path[-1])
            if self.dense_path[-1] != self.pos_dim:
                _str += f' ({self.node_num} * {self.pos_dim})'
            _str += '\n'
        if hasattr(self, 'de_feature'):
            _str = _str + 'Dense layers for feature: '
            for c in self.de_feature_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.de_feature_path[-1])
            if self.dense_path[-1] != self.node_dim:
                _str += f' ({self.node_num} * {self.node_dim})'
            _str += '\n'
        if hasattr(self, 'de_edge'):
            _str = _str + 'Dense layers for edge: '
            for c in self.de_edge_path[:-1]:
                _str += '{}->'.format(c)  
            _str += str(self.de_edge_path[-1])
        return _str 
    def forward(self, x, t = None, c = None):
        pos, feature, edge = None, None, None
        if hasattr(self, 'de_pos'):
            pos = self.de_pos(x, t, c)
            if not self.nodewise:
                pos = pos.reshape(-1, self.pos_dim)

        if hasattr(self, 'de_feature'):
            feature = self.de_feature(x, t, c)
            if not self.nodewise:
                feature = feature.reshape(-1, self.node_dim)

        if hasattr(self, 'de_edge'):
            z = self.de_edge(x, t, c)
            batch_size = x.shape[0] // self.node_num
            ## 2, B * (node_num^2)
            edge_index = torch.concat([self.edge_index + bid * self.node_num for bid in batch_size], dim = -1).to(x.device)
            ## edge is a sparse tensor
            edge = torch.sparse_coo_tensor(indices = edge_index, values = self.inner_p(z, edge_index = edge_index)).coalesce()

        return pos, feature, edge

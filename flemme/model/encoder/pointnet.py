import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block, FoldingLayer, LocalGraphLayer
from flemme.logger import get_logger
logger = get_logger("model.encoder.pointnet")
class PointEncoder(nn.Module):
    def __init__(self, point_dim=3, 
                 local_graph_k=0, 
                 local_feature_channels = [64, 64, 128, 256], 
                 dense_channels = [256, 256],
                 activation = 'lrelu', dropout = 0.,
                 normalization = 'group', num_group = 8,  
                 z_count = 1, pointwise = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.point_dim = point_dim
        self.z_count = z_count
        self.activation = activation
        self.local_graph_k = local_graph_k
        self.dropout = dropout
        self.pointwise = pointwise
        self.normalization = normalization
        self.num_group = num_group
        ## fully connected layers
        # z_count = 2 usually means we compute mean and variance.
        # compute embedding from global feature
        assert len(local_feature_channels) > 1, "Point encoder need more than one local feature channel to extract local feature!"
        assert len(dense_channels) > 0, "Point encoder need to have fully connected layers!"
        dense_channels = [local_feature_channels[-1] * 2, ] + dense_channels
        if self.pointwise:
            dense_channels[0] += local_feature_channels[-1]
        dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1],  
                                            activation = self.activation, dropout=self.dropout, 
                                            norm = normalization, num_group=num_group) for i in range(len(dense_channels) - 2)]
        # the last layer is a linear layer, without batch normalization
        dense_sequence = dense_sequence + [DenseBlock(dense_channels[-2], dense_channels[-1], activation = None, norm = None), ]
        self.dense = nn.ModuleList([nn.Sequential(* (dense_sequence.copy()) ) for _ in range(z_count) ])
        self.out_channel = dense_channels[-1]
        self.dense_path = dense_channels
        self.lf_path = [point_dim,] + local_feature_channels
        self.lf = None

    def forward(self, x, t = None):
        if self.lf is None:
            raise NotImplementedError
        B = x.shape[0]
        # transfer to Nb * d * Np
        res = []
        for lf in self.lf[:-1]:
            x = lf(x, t)
            res.append(x)
        x = torch.concat(res, dim=-1)
        
        pf = self.lf[-1](x)
        ## max and average pooling to get global feature
        x1 = F.adaptive_max_pool1d(pf.transpose(1, 2), 1)
        x2 = F.adaptive_avg_pool1d(pf.transpose(1, 2), 1)
        
        x = torch.concat((x1, x2), dim = -1)
        if self.pointwise:
            # B * D -> B * N * D
            x = x.unsqueeze(1).repeat(1, N, 1)
            x = torch.concat([x, pf], dim=1)
        else:
            x = x.reshape(B, -1)
        ## compute embedding vectors
        x = [self.dense[i](x) for i in range(self.z_count)]
        if self.z_count == 1:
            x = x[0]
        return x

    def __str__(self):
        _str = ''
        _str += 'Local feature extraction layers:'
        for c in self.lf_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.lf_path[-1])
        _str += '\n'
        ## print fc layer
        _str = _str + 'Dense layers: '
        for c in self.dense_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.dense_path[-1])
        _str += '\n'
        return _str 

class PointNetEncoder(PointEncoder):
    def __init__(self, point_dim=3, time_channel = 0, 
                 local_graph_k=0, 
                 local_feature_channels = [64, 64, 128, 256], 
                 dense_channels = [256, 256],
                 building_block = 'dense', 
                 normalization = 'group', num_group = 8, 
                 activation = 'lrelu', dropout = 0., 
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

        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, 
                                        num_group = num_group, 
                                        dropout = dropout)

        # compute point features
        ## local graph feature
        if self.local_graph_k > 0:
            lf_sequence = [LocalGraphLayer(k = self.local_graph_k, 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            is_seq = False) for i in range(len(self.lf_path) - 2) ]
        ## lf, similar to pointnet
        else:    
            lf_sequence = [self.BuildingBlock(in_channel=self.lf_path[i], 
                                            out_channel=self.lf_path[i+1]) for i in range(len(self.lf_path) - 2) ]
            
        lf_sequence.append(self.BuildingBlock(in_channel=sum(self.lf_path[1:-1]), 
                                        out_channel=self.lf_path[-1]))

        self.lf = nn.ModuleList(lf_sequence)

class PointDecoder(nn.Module):
    def __init__(self, point_dim=3, point_num = 2048, 
                in_channel = 256, dense_channels = [], 
                normalization = 'group', num_group = 8, 
                activation = 'lrelu', dropout = 0., 
                folding_times = 0, 
                base_shape_config = {}, 
                pointwise = False, 
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_num = point_num
        self.point_dim = point_dim
        self.activation = activation
        self.pointwise = pointwise
        self.folding_times = folding_times
        ## fully connected layer
        dense_channels = [in_channel,] + dense_channels 
        dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                        norm = normalization, num_group=num_group, 
                                        activation = activation, dropout=dropout) for i in range(len(dense_channels) - 1)]

        if self.folding_times > 0:
            assert not self.pointwise, \
                'point cloud decoder with folding operations shouldn\'t be pointwise.'
            self.fold = None
            base_shape = base_shape_config.get('type', 'grid')
            logger.info(f'using {base_shape} as base shape for folding.')
            
            if base_shape == 'grid':
                grid_len = int(point_num**0.5)
                self.base_shape_dim = 2
                self.actual_point_num = grid_len * grid_len
                # Sample the grids in 2D space
                width = base_shape_config.get('width', 1.0)
                height = base_shape_config.get('height', 1.0)
                xx = np.linspace(-width / 2, width / 2, grid_len, dtype=np.float32)
                yy = np.linspace(-height / 2, height / 2, grid_len, dtype=np.float32)
                np_grid = np.stack(np.meshgrid(xx, yy), axis=-1)   # (45, 45, 2)
                self.base_shape = torch.Tensor(np_grid).view(-1, 2)

            else base_shape == 'cylinder':
                self.base_shape_dim = 3
                c_len = int(point_num**0.5) * 2
                c_num = int(point_num // c_len)
                self.actual_point_num = c_num * c_len
                height = base_shape_config.get('height', 1.6)
                radius = base_shape_config.get('radius', 0.15)
                # c_len 
                z = np.linspace(-height / 2, height / 2, c_len, dtype=np.float32)
                # c_num
                angle = np.linspace(-np.pi, np.pi, c_num, dtype=np.float32)
                x, y = radius * np.cos(angle), radius * np.sin(angle)
                # c_len * c_num
                xx = np.tile(x, c_len)
                yy = np.tile(y, c_len)
                zz = z.repeat(c_num)
                np_cylinder = np.stack([xx, yy, zz], axis=-1)
                self.base_shape = torch.Tensor(np_cylinder)
            else:
                raise NotImplementedError
        else:
            final_channel = point_dim * point_num if not self.pointwise else self.point_dim
            dense_sequence.append(DenseBlock(dense_channels[-1], final_channel, 
                            activation = None, norm = None))
            dense_channels = dense_channels + [final_channel, ]
        self.dense = nn.Sequential(*dense_sequence) 
        self.dense_path = dense_channels
    def __str__(self):
        _str = ''
        _str = _str + 'Dense layers: '
        for c in self.dense_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.dense_path[-1])
        _str += '\n'
        if self.folding_times > 0:
            _str+= f'Fold for {self.folding_times}, 2 ->{self.point_dim}'
        return _str 
    def forward(self, x, t = None):
        x = self.dense(x)
        if self.folding_times > 0:
            if self.fold is None: raise NotImplementedError
            # repeat grid for batch operation
            shape = self.base_shape.to(x.device)
            # (grid_len * grid_len, 2) -> (B, grid_len * grid_len, 2) 
            shape = shape.unsqueeze(0).repeat(x.shape[0], 1, 1)
            # (B, D) -> (B, grid_len * grid_len, D)
            x = x.unsqueeze(1).repeat(1, self.actual_point_num, 1, ) 
            x = self.fold(shape, x, t)[0]
            # x = self.final(x)
        elif not self.pointwise:
            x = x.reshape(-1, self.point_num, self.point_dim)
        return x


class PointNetDecoder(PointDecoder):
    def __init__(self, point_dim=3, point_num = 2048, 
                in_channel = 256, dense_channels = [256], 
                time_channel = 0, building_block = 'dense', 
                normalization = 'group', num_group = 8, 
                activation = 'lrelu', dropout = 0., 
                folding_times = 0, 
                base_shape_config = {},
                folding_hidden_channels = [512, 512],
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
            self.BuildingBlock = get_building_block(building_block, 
                                            time_channel = time_channel, 
                                            activation=activation, 
                                            norm = normalization, 
                                            num_group = 1)
            if self.folding_times < 1:
                logger.warning('Folding times should be larger than 1.')
            folding_channels = [dense_channels[-1] + self.base_shape_dim, ] + [ dense_channels[-1] + point_dim] * (folding_times - 1)
            folding_sequence = [FoldingLayer(BuildingBlock = self.BuildingBlock,
                                    in_channel = fc, out_channel = point_dim,
                                    hidden_channels = folding_hidden_channels) for fc in folding_channels]
            self.fold = SequentialT(*folding_sequence)


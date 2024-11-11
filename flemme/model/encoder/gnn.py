import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block
from flemme.logger import get_logger
logger = get_logger("model.encoder.pointnet")
class GraphEncoder(nn.Module):
    def __init__(self, node_dim=0, pos_dim=3, 
                 node_num=2048,
                 edge_num=None,
                 ### wait to be implemented
                 projection_channel = 64,
                 local_graph_k=0, 
                 local_feature_channels = [64, 64, 128, 256], 
                 dense_channels = [256, 256],
                 activation = 'lrelu', dropout = 0.,
                 normalization = 'group', num_groups = 8,  
                 z_count = 1, pointwise = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.node_dim = node_dim 
        self.pos_dim = pos_dim

    def forward(self, data, t = None):
        if self.lf is None:
            raise NotImplementedError
        B = x.shape[0]
        # transfer to Nb * d * Np
        res = []
        x = self.point_proj(x)
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
        _str = f'projection layer: {self.point_dim}->{self.projection_channel}\n'
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



class PointDecoder(nn.Module):
    def __init__(self, point_dim=3, point_num = 2048, 
                in_channel = 256, dense_channels = [], 
                normalization = 'group', num_groups = 8, 
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
                                        norm = normalization, num_groups=num_groups, 
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
                # Sample the grids in 2D space
                width = base_shape_config.get('width', 1.0)
                height = base_shape_config.get('height', 1.0)
                xx = np.linspace(-width / 2, width / 2, grid_len, dtype=np.float32)
                yy = np.linspace(-height / 2, height / 2, grid_len, dtype=np.float32)
                np_grid = np.stack(np.meshgrid(xx, yy), axis=-1)   # (45, 45, 2)
                self.base_shape = torch.Tensor(np_grid).view(-1, 2)

            elif base_shape == 'cylinder':
                self.base_shape_dim = 3
                c_len = int(point_num**0.5) * 2
                c_num = int(point_num // c_len)
                height = base_shape_config.get('height', 1.6)
                height_axis = base_shape_config.get('height_axis', 'z')
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
                if height_axis == 'z':
                    np_cylinder = np.stack([xx, yy, zz], axis=-1)
                elif height_axis == 'x':
                    np_cylinder = np.stack([zz, xx, yy], axis=-1)
                elif height_axis == 'y':
                    np_cylinder = np.stack([xx, zz, yy], axis=-1)
                else:
                    logger.error('Height axis should be one of [x, y, z].')
                    exit(1)
                self.base_shape = torch.Tensor(np_cylinder)
            else:
                logger.error('Unsupported base shape.')
                exit(1)
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
        if self.folding_times > 0:
            _str+= f'\nFold for {self.folding_times}, 2 ->{self.point_dim}'
        elif self.dense_path[-1] != self.point_dim:
            _str += f' ({self.point_num} * {self.point_dim})'
        _str += '\n'
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
            x = x.unsqueeze(1).repeat(1, shape.shape[1], 1) 
            x = self.fold(shape, x, t)[0]
            # x = self.final(x)
        elif not self.pointwise:
            x = x.reshape(-1, self.point_num, self.point_dim)
        return x




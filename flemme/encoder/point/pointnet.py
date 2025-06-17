import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block, get_ca, \
    FoldingLayer, LocalGraphLayer, MultipleBuildingBlocks, MultiLayerPerceptionBlock,\
    VoxelLayer
from flemme.logger import get_logger
from .sphere3d import icosphere, uvsphere
import copy
logger = get_logger("encoder.point.pointnet")
class PointEncoder(nn.Module):
    def __init__(self, point_dim,
                 projection_channel,
                 time_channel,
                 num_neighbors_k, 
                 local_feature_channels,
                 voxel_resolutions,
                 voxel_conv_kernel_size,
                 num_blocks,
                 dense_channels,
                 activation, dropout,
                 normalization, num_norm_groups,  
                 z_count, vector_embedding, 
                 last_activation,
                 channel_attention,
                 time_injection,
                 with_se,
                 coordinate_normalize,
                 condition_channel,
                 condition_injection,
                 condition_first,
                 **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.point_dim = point_dim
        self.z_count = z_count
        self.activation = activation
        self.num_neighbors_k = num_neighbors_k
        if self.num_neighbors_k > 0:
            logger.info('This encoder will construct local neighbor graph for dynamic feature extraction.')
        self.dropout = dropout
        self.vector_embedding = vector_embedding
        self.normalization = normalization
        self.num_norm_groups = num_norm_groups
        self.point_proj = nn.Linear(point_dim, projection_channel)
        self.projection_channel = projection_channel
        # self.time_channel = time_channel
        # self.condition_channel = condition_channel
        self.num_blocks = num_blocks
        self.voxel_resolutions = voxel_resolutions

        assert len(self.voxel_resolutions) == 0 or len(self.voxel_resolutions) == len(local_feature_channels),\
            "Voxel resolutions should have a same size with local feature channels."
        ## fully connected layers
        # z_count = 2 usually means we compute mean and variance.
        # compute embedding from global feature
        assert len(local_feature_channels) > 1, "Point encoder need more than one local feature channel to extract local feature!"
        assert len(dense_channels) > 0, "Point encoder need to have fully connected layers!"
        global_feature_channel = local_feature_channels[-1] * 2
        dense_channels = [global_feature_channel, ] + dense_channels
        if not self.vector_embedding:
            dense_channels[0] += local_feature_channels[-1]
        if last_activation:
            dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1],  
                                                time_channel = time_channel,
                                                activation = self.activation, dropout=self.dropout, 
                                                norm = normalization, num_norm_groups=num_norm_groups,
                                                time_injection=time_injection,
                                                condition_channel = condition_channel,
                                                condition_injection = condition_injection,
                                                condition_first = condition_first) for i in range(len(dense_channels) - 1)]
        else:
            dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1],  
                                                time_channel = time_channel,
                                                activation = self.activation, dropout=self.dropout, 
                                                norm = normalization, num_norm_groups=num_norm_groups,
                                                time_injection=time_injection,
                                                condition_channel = condition_channel,
                                                condition_injection = condition_injection,
                                                condition_first = condition_first) for i in range(len(dense_channels) - 2)]
            # the last layer is a linear layer, without batch normalization
            dense_sequence = dense_sequence + [DenseBlock(dense_channels[-2], dense_channels[-1], 
                                        activation = None, norm = None), ]
        self.dense = nn.ModuleList([SequentialT(* (copy.deepcopy(dense_sequence)) ) for _ in range(z_count) ])
        self.out_channel = dense_channels[-1]
        self.dense_path = dense_channels
        self.lf_path = [projection_channel,] + local_feature_channels
        self.lf = None

        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = 1, channel = self.lf_path[i+1], channel_dim = -1, **channel_attention) 
                           for i in range(len(self.lf_path) - 1)]
            self.ca = nn.ModuleList(ca_sequence)
        
        # self.need_pos = len(self.voxel_resolutions) > 0

        # point-voxel cnn
        if len(self.voxel_resolutions) > 0:
            logger.info('This model extracts voxel features.')
            VBuildingBlock = get_building_block('conv',
                                            dim = 3,
                                            time_channel = time_channel, 
                                            activation=activation, 
                                            norm = 'batch', 
                                            num_norm_groups = num_norm_groups, 
                                            kernel_size = voxel_conv_kernel_size,
                                            time_injection = time_injection,
                                            condition_channel = condition_channel,
                                            condition_injection = condition_injection,
                                            condition_first = condition_first)
            vlf_sequence = [VoxelLayer(resolution = self.voxel_resolutions[i], 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = VBuildingBlock,
                                            num_blocks = self.num_blocks,
                                            with_se = with_se,
                                            coordinate_normalize=coordinate_normalize,
                                            ) for i in range(len(self.lf_path) - 2) ]
            vlf_sequence.append(VoxelLayer(resolution = self.voxel_resolutions[-1], 
                                            in_channel = sum(self.lf_path[1:-1]),
                                            out_channel = self.lf_path[-1], 
                                            BuildingBlock = VBuildingBlock,
                                            num_blocks = self.num_blocks,
                                            with_se = with_se,
                                            coordinate_normalize=coordinate_normalize,
                                            ))
            self.vlf = nn.ModuleList(vlf_sequence)
    def forward(self, x, t = None, c = None):
        if self.lf is None:
            raise NotImplementedError
        B, N, _ = x.shape
        ## N * Np * d
        res = []
        pos = x[...,:3]
        x = self.point_proj(x)
        for lid, lf in enumerate(self.lf[:-1]):
            ## fuse voxel features and point features
            vx = x
            x = lf(x, t, c)
            if hasattr(self, 'vlf'):
                x = x + self.vlf[lid](vx, pos, t, c)
            if hasattr(self, 'ca'):
                self.ca[lid](x)
            res.append(x)

        x = torch.concat(res, dim=-1)
        pf = self.lf[-1](x, t, c)
        if hasattr(self, 'vlf'):
            pf = pf + self.vlf[-1](x, pos, t, c)
        if hasattr(self, 'ca'):
            pf = self.ca[-1](pf)
        ## max and average pooling to get global feature
        x1 = F.adaptive_max_pool1d(pf.transpose(1, 2), 1).transpose(1, 2)
        x2 = F.adaptive_avg_pool1d(pf.transpose(1, 2), 1).transpose(1, 2)
        x = torch.concat((x1, x2), dim = -1)
        if not self.vector_embedding:
            # B * D -> B * N * D
            # local feature plus global feature
            x = x.repeat(1, N, 1)
            x = torch.concat([x, pf], dim=-1)
        else:
            x = x.reshape(B, -1)
        ## compute embedding vectors
        x = [self.dense[i](x, t, c) for i in range(self.z_count)]
        if self.z_count == 1:
            x = x[0]
        return x

    def __str__(self):
        _str = f'Projection layer: {self.point_dim}->{self.projection_channel}\n'
        _str += 'Local feature extraction layers: '
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
    def __init__(self, point_dim=3, 
                 projection_channel = 64,
                 time_channel = 0,
                 num_neighbors_k=0, 
                 ### point-voxel cnn
                 local_feature_channels = [64, 64, 128, 256], 
                 voxel_resolutions = [],
                 num_blocks = 2,
                 dense_channels = [256, 256],
                 building_block = 'dense', 
                 normalization = 'group', num_norm_groups = 8, 
                 activation = 'lrelu', dropout = 0., 
                 z_count = 1, vector_embedding = True, 
                 last_activation = True, 
                 channel_attention = None, 
                 time_injection = 'gate_bias', 
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
                num_neighbors_k=num_neighbors_k, 
                local_feature_channels = local_feature_channels, 
                num_blocks = num_blocks,
                dense_channels = dense_channels,
                normalization = normalization,
                num_norm_groups = num_norm_groups,
                activation = activation, dropout = dropout, 
                z_count = z_count, vector_embedding = vector_embedding, 
                last_activation = last_activation,
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
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)
        
        # compute point features
        ## local graph feature
        if self.num_neighbors_k > 0:
            lf_sequence = [LocalGraphLayer(k = self.num_neighbors_k, 
                                            in_channel = self.lf_path[i],
                                            out_channel = self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks,
                                            ) for i in range(len(self.lf_path) - 2) ]
        ## local feature, similar to pointnet
        else:    
            lf_sequence = [MultipleBuildingBlocks(in_channel=self.lf_path[i], 
                                            out_channel=self.lf_path[i+1], 
                                            BuildingBlock = self.BuildingBlock,
                                            num_blocks = self.num_blocks,
                                            channel_dim=-1) for i in range(len(self.lf_path) - 2) ]
        
        lf_sequence.append(self.BuildingBlock(in_channel=sum(self.lf_path[1:-1]), 
                                        out_channel=self.lf_path[-1]))
        self.lf = nn.ModuleList(lf_sequence)

class PointNetDecoder(nn.Module):
    def __init__(self, point_dim=3, point_num = 2048, 
                in_channel = 256, dense_channels = [256], 
                time_channel = 0,
                normalization = 'group', num_norm_groups = 8, 
                activation = 'lrelu', dropout = 0., 
                folding_times = 0, 
                base_shape_config = {},
                num_blocks = 2,
                final_channels = [512, 512],
                folding_hidden_channels = [512, 512],
                vector_embedding = True, 
                time_injection = 'gate_bias', 
                condition_channel = 0, 
                condition_injection = 'gate_bias', 
                condition_first = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_num = point_num
        self.point_dim = point_dim
        self.activation = activation
        self.vector_embedding = vector_embedding
        self.folding_times = folding_times
        ## fully connected layer
        dense_channels = [in_channel,] + dense_channels 
        dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                        time_channel = time_channel,
                                        norm = normalization, num_norm_groups=num_norm_groups, 
                                        activation = activation, dropout=dropout,
                                        time_injection=time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first) for i in range(len(dense_channels) - 1)]
        self.dense = SequentialT(*dense_sequence) 
        self.dense_path = dense_channels
        if self.folding_times > 0:
            assert self.vector_embedding, \
                'point cloud decoder with folding operations should have vector embeddings.'
            assert folding_times > 1, 'You need fold twice at least.'

            base_shape = base_shape_config.get('type', 'grid2d')
            logger.info(f'using {base_shape} as base shape for folding.')
            
            if base_shape == 'grid2d':
                grid_len = int(point_num**0.5)
                self.base_shape_dim = 2
                # Sample the grids in 2D space
                width = base_shape_config.get('width', 1.0)
                height = base_shape_config.get('height', 1.0)
                xx = np.linspace(-width / 2, width / 2, grid_len, dtype=np.float32)
                yy = np.linspace(-height / 2, height / 2, grid_len, dtype=np.float32)
                np_grid = np.stack(np.meshgrid(xx, yy), axis=-1)   # (grid_len, grid_len, 2)
                self.base_shape = torch.Tensor(np_grid).view(-1, 2)
            elif base_shape == 'grid3d':
                grid_len = int(point_num**(1 / 3))
                self.base_shape_dim = 2
                # Sample the grids in 2D space
                width = base_shape_config.get('width', 1.0)
                height = base_shape_config.get('height', 1.0)
                depth = base_shape_config.get('depth', 1.0)
                xx = np.linspace(-width / 2, width / 2, grid_len, dtype=np.float32)
                yy = np.linspace(-height / 2, height / 2, grid_len, dtype=np.float32)
                zz = np.linspace(-depth / 2, depth / 2, grid_len, dtype=np.float32)
                np_grid = np.stack(np.meshgrid(xx, yy, zz), axis=-1)   # (grid_len, grid_len, grid_len, 3)
                self.base_shape = torch.Tensor(np_grid).view(-1, 3)
            elif base_shape == 'icosphere':
                radius = base_shape_config.get('radius', 1.0)
                self.base_shape = torch.from_numpy(icosphere(point_num))
                self.base_shape = self.base_shape * radius
            elif base_shape == 'uvsphere':
                radius = base_shape_config.get('radius', 1.0)
                self.base_shape = torch.from_numpy(uvsphere(point_num))
                self.base_shape = self.base_shape * radius
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
            folding_channels = [self.dense_path[-1] + self.base_shape_dim, ] + [ self.dense_path[-1] + point_dim] * (folding_times - 1)
            folding_sequence = [FoldingLayer(in_channel = fc, out_channel = point_dim,
                                    hidden_channels = folding_hidden_channels,
                                    n = num_blocks,
                                    time_channel = time_channel,
                                    norm = normalization, num_norm_groups=num_norm_groups, 
                                    activation = activation, dropout=dropout,
                                    time_injection=time_injection,
                                    condition_channel = condition_channel,
                                    condition_injection = condition_injection,
                                    condition_first = condition_first) for fc in folding_channels]
            self.fold = SequentialT(*folding_sequence)
        else:
            final_out_channel = point_dim * point_num if self.vector_embedding else self.point_dim
            # nn.Linear(dense_channels[-1], final_out_channel)
            self.final = MultiLayerPerceptionBlock(in_channel=dense_channels[-1], 
                                            out_channel=final_out_channel,
                                            n = num_blocks, 
                                            hidden_channels = final_channels,
                                            time_channel = time_channel,
                                            norm = normalization, num_norm_groups=num_norm_groups, 
                                            activation = activation, dropout=dropout,
                                            final_activation = False,
                                            time_injection=time_injection,
                                            condition_channel = condition_channel,
                                            condition_injection = condition_injection,
                                            condition_first = condition_first)
        

    def __str__(self):
        _str = ''
        _str = _str + 'Dense layers: '
        for c in self.dense_path:
            _str += '{}->'.format(c)  
        _str += f'{self.point_dim}'
        # if self.folding_times > 0:
        #     _str+= f'\nFold for {self.folding_times} times to {self.point_dim}'
        # else:
        #     _str += f'-> {self.point_dim}'
        _str += '\n'
        return _str 
    def forward(self, x, t = None, c = None):
        if type(x) == tuple:
            x = x[0]
        x = self.dense(x, t, c)
        if self.folding_times > 0:
            # repeat grid for batch operation
            shape = self.base_shape.to(x.device)
            # (grid_len * grid_len, 2) -> (B, grid_len * grid_len, 2) 
            shape = shape.unsqueeze(0).repeat(x.shape[0], 1, 1)
            # (B, D) -> (B, grid_len * grid_len, D)
            x = x.unsqueeze(1).repeat(1, shape.shape[1], 1) 
            x = self.fold(shape, x, t, c)
            # x = self.final(x)
        else:
            x = self.final(x, t, c)
            if self.vector_embedding:
                x = x.reshape(-1, self.point_num, self.point_dim)
        return x

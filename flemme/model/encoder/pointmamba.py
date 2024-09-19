# point cloud encoder for 3D point cloud
import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, SequentialT, get_building_block
from flemme.logger import get_logger
logger = get_logger("model.encoder.pointnet")
class PointTransformerEncoder(nn.Module):
    def __init__(self, point_dim=3, time_channel = 0, 
                conv_channels = [64, 128, 256], 
                conv_attens = [None, None, None],
                fc_channels = [256], 
                building_block = 'single', 
                normalization = 'group', num_group = 8, cn_order = 'cn',
                activation = 'lrelu', dropout=0.1, 
                num_heads = 1, d_k = None, 
                qkv_bias = True, qk_scale = None, atten_dropout = None,
                z_count = 1, pointwise = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.point_dim = point_dim
        self.conv_channels = conv_channels
        self.z_count = z_count
        self.activation = activation
        self.pointwise = pointwise
        ### use a 'AaaBbb' style for class name
        self.BuildingBlock = get_building_block(building_block, 
                                        time_channel = time_channel, 
                                        activation=activation, 
                                        norm = normalization, num_group = num_group, 
                                        order = cn_order, dropout = dropout, 
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout)
        assert len(fc_channels) > 0, "PointNet encoder need to have fully connected layers!"
        ### convolution with kernel size = 1
        # compute point features
        conv_channels = [point_dim,] + conv_channels
        conv_sequence = [self.BuildingBlock(dim=1, in_channel=conv_channels[i], out_channel=conv_channels[i+1], 
                                        kernel_size=1, padding=0, atten = conv_attens[i]) for i in range(len(conv_channels) - 1) ]
        self.conv = SequentialT(*conv_sequence)
        self.conv_path = conv_channels
        # z_count = 2 usually means we compute mean and variance.
        # compute embedding from global feature
        
        ## fully connected layers
        fc_channels = [conv_channels[-1], ] + fc_channels
        ### concat
        if self.pointwise:
            fc_channels[0] += conv_channels[-1]
        fc_sequence = [ DenseBlock(fc_channels[i], fc_channels[i+1],  
                                            norm = normalization, batch_dim=1, num_group=num_group, 
                                            activation = self.activation) for i in range(len(fc_channels) - 2)]
        # the last layer is a linear layer, without batch normalization
        fc_sequence = fc_sequence + [DenseBlock(fc_channels[-2], fc_channels[-1], 0, activation = None), ]
        self.fc = nn.ModuleList([nn.Sequential(* (fc_sequence.copy()) ) for _ in range(z_count) ])
        self.out_channel = fc_channels[-1]
        self.fc_path = fc_channels

    def __str__(self):
        _str = ''
        # print convolution layers
        _str += 'convolution layers:'
        for c in self.conv_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.conv_path[-1])
        _str += '\n'
        ## print fc layer
        _str = _str + 'Fully-connected layers: '
        for c in self.fc_path[:-1]:
            _str += '{}->'.format(c)  
        _str += str(self.fc_path[-1])
        _str += '\n'
        return _str 
    # input: Nb * Np * d
    def forward(self, x, t = None):
        # transfer to Nb * d * Np
        x = x.transpose(1, 2)
        B, _, N = x.shape
        ## point feature
        pf, _ = self.conv(x, t)

        ## global feature
        x = F.adaptive_max_pool1d(pf, 1)
        

        if self.pointwise:
            x = x.repeat(1, 1, N)
            x = torch.concat([x, pf], dim=1)
            x = x.transpose(1, 2)
        else:
            x = x.reshape(B, -1)

        ## fully connected layer
        x = [self.fc[i](x) for i in range(self.z_count)]
        if self.z_count == 1:
            x = x[0]
        return x
# a very simple decoder
class PointMambaDecoder(PointNetDecoder):
    def __init__(self, point_dim=3, point_num = 2048, in_channel = 128, fc_channels = [256], 
                 activation = 'lrelu', dropout = 0.1, pointwise = False, **kwargs):
        super().__init__(point_dim, point_num, in_channel, fc_channels, 
                         activation = activation, dropout = dropout, pointwise=pointwise)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
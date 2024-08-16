import torch
import torch.nn.functional as F
from torch import nn
from flemme.block import DenseBlock, get_building_block
from .pointnet import PointNetDecoder
from flemme.logger import get_logger
logger = get_logger("model.encoder.dgcnn")
## part of this code is adopted from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
def knn(x, k):
    # x: B*D*N, inner: B*N*N
    inner = 2*torch.matmul(x.transpose(1, 2), x)
    # x^2: B*1*N
    xx = torch.sum(x**2, dim=1, keepdim=True)
    # negative distance
    pairwise_distance = -(xx - inner + xx.transpose(1, 2))
    # return the closest k indices, idx: B*N*K
    idx = pairwise_distance.topk(k=k, dim=-1)[1] 
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size, num_dims, num_points = x.shape
    # why do we need this line?
    # x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    x = x.transpose(1, 2).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    # feature: B*N*K*D 
    feature = feature.view(batch_size, num_points, k, num_dims) 
    # x: B*N*K*D
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    # feature: B*D*N*2K
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature
    
class DGCNNEncoder(nn.Module):
    def __init__(self, point_dim=3, k = 20, time_channel = 0, conv_channels = [64, 64, 128, 256], 
                 conv_attens = [None, None, None, None],
                 fc_channels = [512, 512, 256],
                 building_block = 'single', 
                 normalization = 'group', num_group = 8, cn_order = 'cn',
                 activation = 'lrelu', dropout = 0.1, num_heads = 1, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 z_count = 1, pointwise = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.point_dim = point_dim
        self.conv_channels = conv_channels
        self.z_count = z_count
        self.activation = activation
        self.k = k
        self.dropout = dropout
        self.pointwise = pointwise
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, activation=activation, 
                                        norm = normalization, num_group = num_group, 
                                        order = cn_order, dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout)
        assert len(fc_channels) > 0, "DGCNN encoder need to have fully connected layers!"
        ### convolution with kernel size = 1
        # compute point features
        conv_channels = [point_dim,] + conv_channels
        conv_sequence = [self.BuildingBlock(dim=2, in_channel=conv_channels[i] * 2, out_channel=conv_channels[i+1], 
                                        kernel_size=1, padding=0, atten = conv_attens[i]) for i in range(len(conv_channels) - 2) ]
        
        conv_sequence.append(self.BuildingBlock(dim=1, in_channel=sum(self.conv_channels[:-1]), out_channel=conv_channels[-1], 
                                            kernel_size=1, padding=0))
        self.conv = nn.ModuleList(conv_sequence)
        self.conv_path = conv_channels
        # z_count = 2 usually means we compute mean and variance.
        # compute embedding from global feature
        
        ## fully connected layers
        fc_channels = [conv_channels[-1] * 2, ] + fc_channels
        if self.pointwise:
            fc_channels[0] += conv_channels[-1]
        fc_sequence = [ DenseBlock(fc_channels[i], fc_channels[i+1], 1, 
                                            activation = self.activation, dropout=self.dropout) for i in range(len(fc_channels) - 2)]
        # the last layer is a linear layer, without batch normalization
        fc_sequence = fc_sequence + [DenseBlock(fc_channels[-2], fc_channels[-1], 0, activation = None), ]
        self.fc = nn.ModuleList([nn.Sequential(* (fc_sequence.copy()) ) for _ in range(z_count) ])
        self.out_channel = fc_channels[-1]
        self.fc_path = fc_channels

    def __str__(self):
        _str = ''
        # print convolution layers
        _str += 'Edge convolution layers:'
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
        res = []
        # edge convolution
        for conv in self.conv[:-1]:
            x = get_graph_feature(x, k = self.k)
            x = conv(x, t)
            x = x.max(dim=-1, keepdim=False)[0]
            res.append(x)

        x = torch.concat(res, dim=1)
        pf = self.conv[-1](x)
        ## max and average pooling
        x1 = F.adaptive_max_pool1d(pf, 1)
        x2 = F.adaptive_avg_pool1d(pf, 1)
        
        x = torch.concat((x1, x2), dim = 1)
        if self.pointwise:
            x = x.repeat(1, 1, N)
            x = torch.concat([x, pf], dim=1)
            x = x.transpose(1, 2)
        else:
            x = x.reshape(B, -1)
        ## compute embedding vectors
        x = [self.fc[i](x) for i in range(self.z_count)]
        if self.z_count == 1:
            x = x[0]
        return x

# Here we directly use pointnet decoder as the DGCNN decoder.
class DGCNNDecoder(PointNetDecoder):
    def __init__(self, point_dim=3, point_num = 2048, in_channel = 128, fc_channels = [256], 
                 activation = 'lrelu', dropout = 0.1, pointwise = False, **kwargs):
        super().__init__(point_dim, point_num, in_channel, fc_channels, 
                         activation = activation, dropout = dropout, pointwise=pointwise)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
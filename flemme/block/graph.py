
from .common import *
# import torch
# import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, TransformerConv, norm as gnorm, conv as gconv
from functools import partial
def get_graph_norm(norm_name, norm_channel, num_norm_groups = 0):
        ### normalization layer
        if norm_name == 'batch':
            return gnorm.BatchNorm(in_channels = norm_channel), Norm.BATCH
        elif norm_name == 'group' and num_norm_groups > 0:
            return gnorm.DiffGroupNorm(in_channels = norm_channel, groups = num_norm_groups), Norm.GROUP
        elif norm_name == 'layer':
            return gnorm.LayerNorm(in_channels = norm_channel), Norm.LAYER
        elif norm_name == 'instance':
            return gnorm.InstanceNorm(in_channels = norm_channel), Norm.INSTANCE
        else:
            return nn.Identity(), Norm.NONE 

# def get_graph_conv(graph_conv_config):
#   gconfig = graph_conv_config.copy()
#   name = gconfig.pop('name', 'GCNConv')
#   graph_class = get_class(name, module = gconv)
#   return partial(graph_class, **gconfig)

### graph convolution
class GraphConvBlock(nn.Module):
    def __init__(self, in_channel, 
                out_channel, 
                time_channel = 0,
                activation = 'relu', 
                norm = 'batch', 
                num_norm_groups = 0, 
                order = 'cn',
                time_injection = 'gate_bias',
                graph_normalize = True,
                improved = False,
                cached = False,  
                bias = True):
        super().__init__()
        # convolution layer
        self.conv = GCNConv(in_channels = in_channel, 
          out_channels = out_channel,
          normalize = graph_normalize,
          improved = improved,
          cached = cached,
          bias = bias)
        norm_channel = out_channel
        if order.index('n') < order.index('c'):
            norm_channel = in_channel
        self.norm, self.norm_type = get_graph_norm(norm, 
                        norm_channel = norm_channel, num_norm_groups = num_norm_groups)

        # activation function
        self.act = get_act(activation)
        self.order = order
        self.time_channel = time_channel
        if self.time_channel > 0:
            self.time = get_context_injection(time_injection, self.time_channel, out_channel, channel_dim=-1)
            

    def forward(self, x, edge_index, t = None):
        for m in self.order:
            if m == 'n':
                x = self.norm(x)
            elif m == 'c':
                x = self.conv(x, edge_index)
        x = self.act(x)

        if t is not None:
            assert self.time_channel == t.shape[-1], \
                f'time channel mismatched: want {self.time_channel} but got {t.shape[-1]}.'  
            x = self.time(x, t)
        return x

class ChebConvBlock(GraphConvBlock):
    def __init__(self, in_channel, 
                out_channel, 
                time_channel = 0,
                activation = 'relu', 
                norm = 'batch', 
                num_norm_groups = 0, 
                order = 'cn',
                filter_size = 5,
                graph_normalization = 'sym', 
                bias = True):
        super().__init__(time_channel = time_channel,
          activation = activation, norm = norm,
          num_norm_groups = num_norm_groups, order = order)
        # convolution layer
        self.conv = ChebConv(in_channels = in_channel, 
          out_channels = out_channel,
          K = filter_size,
          normalization = graph_normalization,
          bias = bias)

class TransConvBlock(GraphConvBlock):
    def __init__(self, in_channel, 
                out_channel, 
                graph_conv,
                time_channel = 0,
                activation = 'relu', 
                norm = 'batch', 
                num_norm_groups = 0, 
                order = 'cn',
                num_heads = 1,
                concat = True,
                beta = False,
                dropout = 0.0,
                bias = True):
        super().__init__(time_channel = time_channel,
          activation = activation, norm = norm,
          num_norm_groups = num_norm_groups, order = order)
        # convolution layer
        self.conv = TransformerConv(in_channels = in_channel, 
          out_channels = out_channel,
          heads = num_heads, concat = concat, 
          beta = beta, dropout = dropout,
          bias = bias)

class InnerProductBlock(nn.Module):
    def forward(self, z, edge_index = None):
        if edge_index is not None:
            return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        else: 
            return torch.matmul(z, z.t())
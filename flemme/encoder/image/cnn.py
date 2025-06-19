# encode image to an vector embedding
import torch
import torch.nn.functional as F
from torch import nn
import math
from flemme.block import DenseBlock, DownSamplingBlock, UpSamplingBlock, \
    get_building_block, MultipleBuildingBlocks, get_ca
from flemme.logger import get_logger
import copy
logger = get_logger("encoder.image.cnn")

# add fc layer to embed the image into a latent vector
class CNNEncoder(nn.Module):
    '''
    image size: a list such as [64, 64], the length of image size determines the dimension of image.
    image_channel: number of image channels
    down_channels: a list of feature_channels during the downsampling stage. The length of conv_channels determines the downsampling times.
    middle_channels: a list of feature_channels during the middle convolution stage.
    dsample_function: function for downsampling, can be one of ['conv', 'max_pooling', 'avg_pooling']
    building_block: convolution block, can be one of ['single', 'double']
    dense_channels: int or list. dimensions of embedding vectors during the 'fully connected layer' stage. The length of dense_channels determines the number of fc layers. The last
    value would be the dimension of output embedding vectors.
    zcount: int, zcount is the number of output embedding vectors
    '''
    def __init__(self, image_size, image_channel = 3, time_channel = 0, patch_channel = 32, patch_size = 2,
                 down_channels = [64, 128], down_attens = [None, None], 
                 shape_scaling = [2, 2],  middle_channels = [256, 256], 
                 middle_attens = [None, None], depthwise = False, kernel_size = 3, 
                 dense_channels = [256], dsample_function = 'conv', building_block='single', 
                 normalization = 'group', num_norm_groups = 8, num_blocks = 2,
                 activation = 'relu', dropout = 0., num_heads = 1, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 abs_pos_embedding = False, return_feature_list = False,
                 channel_attention = None,
                 time_injection = 'gate_bias', 
                 condition_channel = 0,
                 condition_injection = 'gate_bias',
                 condition_first = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.image_channel = image_channel
        self.dim = len(image_size)
        self.dsample_function = dsample_function
        self.image_size = image_size
        self.image_patch_channel = patch_channel
        self.vector_embedding = isinstance(dense_channels, list) and len(dense_channels) > 0
        ## down sample times
        self.d_depth = 0 if not isinstance(down_channels, list) else len(down_channels)
        self.activation = activation
        self.shape_scaling = shape_scaling
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        if not sum([im_size % (patch_size * math.prod(shape_scaling)) for im_size in self.image_size ]) == 0:
            logger.error('Please check your image size, patch size and downsample depth to make sure the image size can be divisible.')
            exit(1)
        ### use a 'AaaBbb' style for class name
        self.BuildingBlock = get_building_block(building_block, dim=self.dim, 
                                        time_channel = time_channel, 
                                        activation = activation, depthwise = depthwise, 
                                        kernel_size = kernel_size, padding = (kernel_size - 1) // 2,
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,  
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)
        ## down-sampling and convolution layers
        self.image_proj = DownSamplingBlock(dim=self.dim, scale_factor=patch_size, in_channel=image_channel, 
                                               out_channel=self.image_patch_channel, func=dsample_function)
        self.absolute_pos_embed = None
        if abs_pos_embedding:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros([1, self.image_patch_channel] + [s // self.patch_size for s in self.image_size] ))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)

        down_channels = [self.image_patch_channel, ] + down_channels
        self.down = nn.ModuleList([DownSamplingBlock(dim=self.dim, in_channel=down_channels[i], 
                                                        func=dsample_function, scale_factor=shape_scaling[i-1]) for i in range(1, len(down_channels))])    
        self.d_conv = nn.ModuleList( [MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        in_channel=down_channels[i], 
                                                        out_channel=down_channels[i+1], 
                                                        atten=down_attens[i]) for i in range(len(down_channels) - 1) ])
        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = self.dim, channel = down_channels[i+1], channel_dim = 1, **channel_attention) 
                           for i in range(len(down_channels) - 1)]
            self.dca = nn.ModuleList(ca_sequence)

        self.down_path = [self.image_channel, ] + down_channels


        dense_channels = [ middle_channels[-1], ] + dense_channels
        self.dense_path = dense_channels.copy()
        ## middle convolution layer
        self.middle_path = [down_channels[-1], ] + middle_channels 

        middle_channels = [down_channels[-1], ] + middle_channels 
        module_sequence = [MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                          dim=self.dim, in_channel=middle_channels[i], 
                                          out_channel=middle_channels[i+1], 
                                          atten = middle_attens[i]) for i in range(len(middle_channels) - 1)]
        
        self.middle = nn.ModuleList(module_sequence)
        
        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = self.dim, channel = middle_channels[i+1], channel_dim = 1, **channel_attention) 
                           for i in range(len(middle_channels) - 1)]
            self.mca = nn.ModuleList(ca_sequence)
        ### fully connected layers
        
        if self.vector_embedding:
            dense_channels[0] = int( math.prod(self.image_size) / ((self.patch_size *  math.prod(self.shape_scaling)) ** self.dim)  *dense_channels[0])
            dense_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                            norm = normalization, num_norm_groups=num_norm_groups, 
                                            activation = self.activation) for i in range(len(dense_channels) - 1)]
            self.dense = nn.Sequential(*(copy.deepcopy(dense_sequence)))

        ## set out_channel
        self.out_channel = dense_channels[-1]
        self.return_feature_list = return_feature_list
    def __str__(self):
        _str = ''
        if len(self.down_path) > 1:
            _str += 'Down-sampling and convolution layers: '
            for c in self.down_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.down_path[-1])
            _str += '\n'

        if len(self.middle_path) > 1:
            _str += 'Middle convolution layers: '
            for c in self.middle_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.middle_path[-1])
            _str += '\n'
    
        if self.vector_embedding:
            _str = _str + 'Dense layers: '
            for c in self.dense_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.dense_path[-1])
            _str += '\n'
        return _str 
    
    def forward(self, x, t = None, c = None):
        x = self.image_proj(x)
        res = []
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        if self.d_depth > 0:
            for did, (d_conv, down) in enumerate(zip(self.d_conv, self.down)):
                x = d_conv(x, t, c)
                if hasattr(self, 'dca'):
                    x = self.dca[did](x)
                res = res + [x,]
                x = down(x)
        for mid, m_conv in enumerate(self.middle):
            x = m_conv(x, t, c)
            if hasattr(self, 'mca'):
                x = self.mca[mid](x)

        if self.vector_embedding:
            x = x.reshape(x.shape[0], -1)
            x = self.dense(x)
        if self.return_feature_list:
            return x, res
        return x

    
# recover image from embedding vector
class CNNDecoder(nn.Module):
    '''
    image size: a list such as [64, 64], the length of image size determines the dimension of output image.
    out_channel: number of output image channels
    conv_channels: a list of feature_channels during the upsampling stage. The length of conv_channels determines the upsampling times.
    dsample_function: function for downsampling, can be one of ['conv', 'inter']
    building_block: convolution block, can be one of ['single', 'double']
    dense_channels: list. dimensions of embedding vectors during the 'fully connected layer' stage. The length of dense_channels determines the number of fc layers. The last
    value would be the dimension of output embedding vectors.
    '''
    def __init__(self, image_size, image_channel = 3, patch_size = 2, in_channel = 256,  time_channel = 0, 
                 dense_channels = [256], up_channels = [128, 64], up_attens = [None, None], 
                 shape_scaling = [2, 2], final_channels = [], 
                 final_attens = [], depthwise = False, kernel_size = 3, 
                 usample_function = 'conv', building_block='single', 
                 normalization = 'group', num_norm_groups = 8, 
                 num_blocks = 2, activation = 'relu', dropout = 0., num_heads = 1, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 return_feature_list = False, 
                 channel_attention = None, 
                 time_injection = 'gate_bias', 
                 condition_channel = 0,
                 condition_injection = 'gate_bias',
                 condition_first = False,
                 **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.image_channel = image_channel
        self.dim = len(image_size)
        self.final_channels = final_channels
        self.usample_function = usample_function
        self.image_size = image_size
        self.vector_embedding = isinstance(dense_channels, list) and len(dense_channels) > 0
        ## up-sampling times
        self.u_depth = 0 if not isinstance(up_channels, list) else len(up_channels)
        self.activation = activation
        self.patch_size = patch_size
        self.shape_scaling = shape_scaling
        self.num_blocks = num_blocks
        ### use a 'AaaBbb' style for class name
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        depthwise = depthwise, activation=activation, 
                                        kernel_size = kernel_size, padding = (kernel_size - 1) // 2,
                                        norm = normalization, num_norm_groups = num_norm_groups, 
                                        dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout,
                                        time_injection = time_injection,
                                        condition_channel = condition_channel,
                                        condition_injection = condition_injection,
                                        condition_first = condition_first)
        ## fully connected layer
        dense_channels = [in_channel, ] + dense_channels 
        if not sum([im_size % (patch_size * math.prod(shape_scaling)) for im_size in self.image_size ]) == 0:
            logger.error('Please check your image size, patch size and downsample depth to make sure the image size can be divisible.')
            exit(1)
        if self.vector_embedding:
            # used for view (reshape)
            self.view_shape = [-1, int( dense_channels[-1]),] +[int(im_size // (self.patch_size * math.prod(shape_scaling) )) for im_size in self.image_size ]
            module_sequence = [ DenseBlock(dense_channels[i], dense_channels[i+1], 
                                                    norm = normalization, num_norm_groups=num_norm_groups, 
                                                    activation = self.activation) for i in range(len(dense_channels) - 2)]
            # to construct image shape
            # if there is not fc layer, then we also don't need this step
            module_sequence.append(DenseBlock(dense_channels[-2],  
                                                       int( dense_channels[-1] * math.prod(self.image_size) / ((self.patch_size *  math.prod(self.shape_scaling)) ** self.dim)), 
                                                       norm = normalization, num_norm_groups=num_norm_groups,  
                                                       activation = self.activation))
            self.dense = nn.Sequential(*module_sequence)  
        self.dense_path = dense_channels

        ## up-sampling layers and convolution layer
        up_channels = [dense_channels[-1], ] + up_channels
        self.up = nn.ModuleList([UpSamplingBlock(dim=self.dim, in_channel=up_channels[i], 
                                                 func=usample_function, scale_factor=shape_scaling[i]) for i in range(len(up_channels) - 1)])
        self.u_conv = nn.ModuleList([MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                                        dim=self.dim, in_channel=up_channels[i], 
                                                        out_channel=up_channels[i+1], 
                                                        atten=up_attens[i]) for i in range(len(up_channels) - 1) ])
      
        self.up_path = up_channels
        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = self.dim, channel = up_channels[i+1], channel_dim = 1, **channel_attention) 
                           for i in range(len(up_channels) - 1)]
            self.uca = nn.ModuleList(ca_sequence)
        ## final convolution layer
        final_channels = [up_channels[-1],] + final_channels
        module_sequence = [MultipleBuildingBlocks(n = self.num_blocks, BuildingBlock=self.BuildingBlock, 
                                          dim=self.dim, in_channel=final_channels[i], 
                                          out_channel=final_channels[i+1], 
                                          atten = final_attens[i]) for i in range(len(final_channels) - 1) ]
        self.final = nn.ModuleList(module_sequence)
        if channel_attention is not None:
            if isinstance(channel_attention, str):
                channel_attention = {'method': channel_attention}
            else:
                assert type(channel_attention) == dict and len(channel_attention) > 0, "Channel attention should be a str or non-empty dict."
            logger.info(f'Using channel attention: {channel_attention}')
            ca_sequence = [get_ca(dim = self.dim, channel = final_channels[i+1], channel_dim = 1, **channel_attention) 
                           for i in range(len(final_channels) - 1)]
            self.fca = nn.ModuleList(ca_sequence)
        self.image_back_proj = UpSamplingBlock(dim = self.dim, scale_factor=patch_size, in_channel=final_channels[-1], 
                                                       out_channel=self.image_channel, func=usample_function)
        self.final_path = final_channels + [self.image_channel]
        self.return_feature_list = return_feature_list
    def __str__(self):
        _str = ''
        if self.vector_embedding:
            _str = _str + 'Dense layers: '
            for c in self.dense_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.dense_path[-1])
            _str += '\n'

        if len(self.up_path) > 1:
            _str += 'Up-sampling and convolution layers: '
            for c in self.up_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.up_path[-1])
            _str += '\n'
        if len(self.final_path) > 1:
            _str += 'Final convolution layers: '
            for c in self.final_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.final_path[-1])
            _str += '\n'
        return _str 
    
    def forward(self, x, t = None, c = None):
        ### ignore returned features
        if type(x) == tuple:
            x = x[0]
        if self.vector_embedding:
            x = self.dense(x)
            x = x.reshape(*self.view_shape)
        
        res = []
        if self.u_depth > 0:
            for uid, (up, u_conv) in enumerate(zip(self.up, self.u_conv)):
                x = u_conv(up(x), t, c)
                if hasattr(self, 'uca'):
                    x = self.uca[uid](x)
                res = res + [x,]
        for fid, f_conv in enumerate(self.final):
            x = f_conv(x, t, c)
            if hasattr(self, 'fca'):
                x = self.fca[fid](x)
        x = self.image_back_proj(x)
        if self.return_feature_list:
            return x, res
        return x
    
## test code
if __name__ == '__main__':
    pass
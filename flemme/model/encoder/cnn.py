# encode image to an vector embedding
import torch
import torch.nn.functional as F
from torch import nn
import math
from flemme.block import DenseBlock, DownSamplingBlock, UpSamplingBlock, SequentialT,\
    get_building_block, MultipleBuildingBlocks
from flemme.logger import get_logger
logger = get_logger("model.encoder.cnn")

# add fc layer to embed the image into a latent vector
class CNNEncoder(nn.Module):
    '''
    image size: a list such as [64, 64], the length of image size determines the dimension of image.
    image_channel: number of image channels
    down_channels: a list of feature_channels during the downsampling stage. The length of conv_channels determines the downsampling times.
    middle_channels: a list of feature_channels during the middle convolution stage.
    dsample_function: function for downsampling, can be one of ['conv', 'max_pooling', 'avg_pooling']
    building_block: convolution block, can be one of ['single', 'double']
    fc_channels: int or list. dimensions of embedding vectors during the 'fully connected layer' stage. The length of fc_channels determines the number of fc layers. The last
    value would be the dimension of output embedding vectors.
    zcount: int, zcount is the number of output embedding vectors
    '''
    def __init__(self, image_size, image_channel = 3, time_channel = 0, patch_channel = 32, patch_size = 2,
                 down_channels = [64, 128], down_attens = [None, None], 
                 shape_scaling = [2, 2],  middle_channels = [256, 256], 
                 middle_attens = [None, None], depthwise = False, kernel_size = 3, 
                 fc_channels = [256], dsample_function = 'conv', building_block='single', 
                 normalization = 'group', num_group = 8, cn_order = 'cn', num_block = 2,
                 activation = 'relu', z_count = 1, dropout = 0.1, num_heads = 1, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 abs_pos_embedding = False, return_features = False,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.image_channel = image_channel
        self.dim = len(image_size)
        self.dsample_function = dsample_function
        self.z_count = z_count
        self.image_size = image_size
        self.image_patch_channel = patch_channel
        self.vector_embedding = isinstance(fc_channels, list) and len(fc_channels) > 0
        ## down sample times
        self.d_depth = 0 if not isinstance(down_channels, list) else len(down_channels)
        self.activation = activation
        self.shape_scaling = shape_scaling
        self.patch_size = patch_size
        if not sum([im_size % (patch_size * math.prod(shape_scaling)) for im_size in self.image_size ]) == 0:
            logger.error('Please check your image size, patch size and downsample depth to make sure the image size can be divisible.')
            exit(1)
        ### use a 'AaaBbb' style for class name
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        activation = activation, depthwise = depthwise,
                                        kernel_size = kernel_size, padding = (kernel_size - 1) // 2,
                                        norm = normalization, num_group = num_group, 
                                        order = cn_order, dropout = dropout,  
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout)
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
        self.d_conv = nn.ModuleList( [MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        dim=self.dim, in_channel=down_channels[i], 
                                                        out_channel=down_channels[i+1], 
                                                        atten=down_attens[i]) for i in range(len(down_channels) - 1) ])
        self.down_path = [self.image_channel, ] + down_channels

        ## middle convolution layer
        if not self.vector_embedding:
            middle_channels = [mc * self.z_count for mc in middle_channels]

        middle_channels = [down_channels[-1], ] + middle_channels 
        module_sequence = [MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                          dim=self.dim, in_channel=middle_channels[i], 
                                          out_channel=middle_channels[i+1], 
                                          atten = middle_attens[i]) for i in range(len(middle_channels) - 1)]
        
        self.middle = SequentialT(*module_sequence)
        self.middle_path = middle_channels
        
        ### fully connected layers
        fc_channels = [ int(middle_channels[-1] / self.z_count) if not self.vector_embedding else middle_channels[-1], ] + fc_channels
        
        self.fc_path = fc_channels.copy()
        if self.vector_embedding:
            fc_channels[0] = int( math.prod(self.image_size) / ((self.patch_size *  math.prod(self.shape_scaling)) ** self.dim)  *fc_channels[0])
            fc_sequence = [ DenseBlock(fc_channels[i], fc_channels[i+1], 
                                                    norm = normalization, batch_dim=1, num_group=num_group, 
                                                    activation = self.activation) for i in range(len(fc_channels) - 2)]
            fc_sequence = fc_sequence + [DenseBlock(fc_channels[-2], fc_channels[-1], norm=None, activation = None), ]
            self.fc = nn.ModuleList([nn.Sequential(*(fc_sequence.copy()) ) for _ in range(z_count) ])

        ## set out_channel
        self.out_channel = fc_channels[-1]
        self.return_features = return_features
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
            _str = _str + 'Fully-connected layers: '
            for c in self.fc_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.fc_path[-1])
            _str += '\n'
        return _str 
    
    def forward(self, x, t = None):
        x = self.image_proj(x)
        res = []
        if self.absolute_pos_embed is not None:
            x = x + self.absolute_pos_embed
        if self.d_depth > 0:
            for d_conv, down in zip(self.d_conv, self.down):
                x = d_conv(x, t)
                res = [x,] + res
                x = down(x)
        
        x, _ = self.middle(x, t)

        if self.vector_embedding:
            x = x.reshape(x.shape[0], -1)
            x = [ self.fc[i](x) for i in range(self.z_count) ]
        else:
            x = torch.split(x, self.out_channel, dim=1)
        if self.z_count == 1:
            x = x[0]
        if self.return_features:
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
    fc_channels: list. dimensions of embedding vectors during the 'fully connected layer' stage. The length of fc_channels determines the number of fc layers. The last
    value would be the dimension of output embedding vectors.
    '''
    def __init__(self, image_size, image_channel = 3, patch_size = 2, in_channel = 256,  time_channel = 0, 
                 fc_channels = [32], up_channels = [128, 64], up_attens = [None, None], 
                 shape_scaling = [2, 2], final_channels = [], 
                 final_attens = [], depthwise = False, kernel_size = 3, 
                 usample_function = 'conv', building_block='single', 
                 normalization = 'group', num_group = 8, cn_order = 'cn', 
                 num_block = 2, activation = 'relu', dropout=0.1, num_heads = 1, d_k = None, 
                 qkv_bias = True, qk_scale = None, atten_dropout = None, 
                 return_features = False, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
           logger.debug("redundant parameters:{}".format(kwargs))
        self.image_channel = image_channel
        self.dim = len(image_size)
        self.final_channels = final_channels
        self.usample_function = usample_function
        self.image_size = image_size
        self.vector_embedding = isinstance(fc_channels, list) and len(fc_channels) > 0
        ## up-sampling times
        self.u_depth = 0 if not isinstance(up_channels, list) else len(up_channels)
        self.activation = activation
        self.patch_size = patch_size
        self.shape_scaling = shape_scaling
        ### use a 'AaaBbb' style for class name
        self.BuildingBlock = get_building_block(building_block, time_channel = time_channel, 
                                        depthwise = depthwise, activation=activation, 
                                        kernel_size = kernel_size, padding = (kernel_size - 1) // 2,
                                        norm = normalization, num_group = num_group, 
                                        order = cn_order, dropout = dropout,
                                        num_heads = num_heads, d_k = d_k, 
                                        qkv_bias = qkv_bias, qk_scale = qk_scale, 
                                        atten_dropout = atten_dropout)
        ## fully connected layer
        fc_channels = [in_channel, ] + fc_channels 
        if not sum([im_size % (patch_size * math.prod(shape_scaling)) for im_size in self.image_size ]) == 0:
            logger.error('Please check your image size, patch size and downsample depth to make sure the image size can be divisible.')
            exit(1)
        if self.vector_embedding:
            # used for view (reshape)
            self.view_shape = [-1, int( fc_channels[-1]),] +[int(im_size // (self.patch_size * math.prod(shape_scaling) )) for im_size in self.image_size ]
            module_sequence = [ DenseBlock(fc_channels[i], fc_channels[i+1], 
                                                    norm = normalization, batch_dim=1, num_group=num_group, 
                                                activation = self.activation) for i in range(len(fc_channels) - 2)]
            # to construct image shape
            # if there is not fc layer, then we also don't need this step
            module_sequence.append(DenseBlock(fc_channels[-2],  
                                                       int( fc_channels[-1] * math.prod(self.image_size) / ((self.patch_size *  math.prod(self.shape_scaling)) ** self.dim)), 
                                                       norm = normalization, batch_dim=1, num_group=num_group,  
                                                       activation = self.activation))
            self.fc = nn.Sequential(*module_sequence)  
        self.fc_path = fc_channels

        ## up-sampling layers and convolution layer
        up_channels = [fc_channels[-1], ] + up_channels
        self.up = nn.ModuleList([UpSamplingBlock(dim=self.dim, in_channel=up_channels[i], 
                                                 func=usample_function, scale_factor=shape_scaling[i]) for i in range(len(up_channels) - 1)])
        self.u_conv = nn.ModuleList([MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                                        dim=self.dim, in_channel=up_channels[i], 
                                                        out_channel=up_channels[i+1], 
                                                        atten=up_attens[i]) for i in range(len(up_channels) - 1) ])
      
        self.up_path = up_channels

        ## final convolution layer
        final_channels = [up_channels[-1],] + final_channels
        module_sequence = [MultipleBuildingBlocks(n = num_block, BlockClass=self.BuildingBlock, 
                                          dim=self.dim, in_channel=final_channels[i], 
                                          out_channel=final_channels[i+1], 
                                          atten = final_attens[i]) for i in range(len(final_channels) - 1) ]
        self.final = SequentialT(*module_sequence)
        self.image_back_proj = UpSamplingBlock(dim = self.dim, scale_factor=patch_size, in_channel=final_channels[-1], 
                                                       out_channel=self.image_channel, func=usample_function)
        self.final_path = final_channels + [self.image_channel]
        self.return_features = return_features
    def __str__(self):
        _str = ''
        if self.vector_embedding:
            _str = _str + 'Fully-connected layers: '
            for c in self.fc_path[:-1]:
               _str += '{}->'.format(c)  
            _str += str(self.fc_path[-1])
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
    
    def forward(self, x, t = None):
        ### ignore returned features
        if type(x) == tuple:
            x = x[0]
        if self.vector_embedding:
            x = self.fc(x)
            x = x.reshape(*self.view_shape)
        
        res = []
        if self.u_depth > 0:
            for up, u_conv in zip(self.up, self.u_conv):
                x = u_conv(up(x), t)
                res = res + [x,]
        x, t = self.final(x, t)
        x = self.image_back_proj(x)
        if self.return_features:
            return x, res
        return x
    
## test code
if __name__ == '__main__':
    pass
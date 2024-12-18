# auto-encoder for 2D image and 3D point cloud
import torch
import torch.nn as nn
from flemme.utils import DataForm
from flemme.logger import get_logger
from flemme.model.embedding import get_embedding, add_embedding, concat_embedding
from flemme.block import CombineLayer, ResConvBlock, ConvBlock, \
    channel_recover, TimeEmbeddingBlock, VMambaBlock
from flemme.utils import DataForm
from .encoder import create_encoder
logger = get_logger('model.base')

class BaseModel(nn.Module):
    ''' 
    Base model for ae and vae. It follows a encoder-decoder structure, the output can be reconstruction or predicted mask.
    Loss is not specified.
    '''
    def __init__(self, model_config):
        super().__init__()
        encoder_config = model_config.get('encoder', None)
        decoder_config = encoder_config.copy()
        specified_decoder_config = model_config.get('decoder', None)
        if specified_decoder_config is not None:
            for k in specified_decoder_config:
                decoder_config[k] = specified_decoder_config[k]

        assert encoder_config is not None, 'There is no encoder configuration.'
        self.encoder_name = encoder_config.get('name')
        self.decoder_name = decoder_config.get('name')

        self.in_channel = encoder_config.get('in_channel')
        self.out_channel = encoder_config.get('out_channel', self.in_channel)
        ### time_embedding
        self.time_channel = model_config.get('time_channel', 0)
        self.with_time_embedding = (self.time_channel > 0)
        self.time_act = model_config.get('time_activation', 'silu')
        ### time step embeddin
        if self.with_time_embedding:
            logger.info("Construct base model with time embedding.")
            self.t_embed = TimeEmbeddingBlock(out_channel=self.time_channel, activation=self.time_act)
        self.final_time_channel = model_config.get('concat_time_channel', 0) + self.time_channel

        ### condition_embedding
        en_emb_config, de_emb_config = None, None
        cemb_config = model_config.get('condition_embedding', None)
        if cemb_config is not None:
            en_emb_config = cemb_config.get('encoder', None)
            de_emb_config = cemb_config.get('decoder', None)
            self.combine_condition = cemb_config.get('combine_condition', 'add')
            self.merge_ct = cemb_config.get('merge_timestep_and_condition', False)

        if en_emb_config is not None:
            logger.info("Create conditional embedding for encoder.")
            self.en_cemb = get_embedding(en_emb_config)
            if self.combine_condition == 'cat':
                if not self.merge_ct:
                    ### should we use a convolution to project the catted feature to the encoder input space ?
                    encoder_config['encoder_additional_in_channel'] = \
                        encoder_config.get('encoder_additional_in_channel', 0) + self.en_cemb.out_channel
                else:
                    self.final_time_channel += self.en_cemb.out_channel
            else:
                if self.merge_ct:
                    assert self.final_time_channel == self.en_cemb.out_channel,\
                        "condition embedding of encoder and time step embedding should have the same shape for addition."
                else:
                    assert self.in_channel == self.en_cemb.out_channel, \
                        "condition embedding of encoder and input data should have the same shape for addition."
        ### create encoder
        encoder_config['time_channel'] = self.final_time_channel
        self.encoder = create_encoder(encoder_config=encoder_config, return_decoder=False)[0]

        if de_emb_config is not None:
            logger.info("Create conditional embedding for decoder.")
            ### like an auto-regressive model
            if de_emb_config == 'same_as_encoder':
                logger.info('Using encoder\'s condition embedding to compute condition embedding for decoder.')
                if not hasattr(self, 'en_cemb'):
                    logger.error('This is no condition embedding for encoder.')
                    exit(1)
                self.de_cemb = self.en_cemb
            else:
                self.de_cemb = get_embedding(de_emb_config)

            if self.merge_ct:
                logger.error('Merging condition with time-step embedding and decoder condition embedding are specified at the same time, which can lead to conflicts. ')
                exit(1)
            elif self.combine_condition == 'cat':
                decoder_config['decoder_additional_in_channel'] = \
                     decoder_config.get('decoder_additional_in_channel', 0) + self.de_cemb.out_channel
            else:
                assert self.encoder.out_channel == self.de_cemb.out_channel, \
                    "condition embedding of decoder and the output of encoder should have the same shape for addition."
        ### create decoder
        decoder_config['time_channel'] = self.final_time_channel
        decoder_config['decoder_in_channel'] = self.encoder.out_channel
        self.decoder = create_encoder(encoder_config=decoder_config, return_encoder=False)[1]

        if self.final_time_channel > 0:
            logger.info(f'original time channel / concated time channel: {self.time_channel} / {self.final_time_channel - self.time_channel}.')
        

        self.is_generative = False
        self.is_conditional = hasattr(self, 'en_cemb') or hasattr(self, 'de_cemb')
        self.is_supervised = False
        self.loss_reduction = model_config.get('loss_reduction', 'mean')
        self.data_form = self.encoder.data_form
        self.channel_dim = -1 if self.data_form == DataForm.PCD else 1

    def __str__(self):
        _str = '********************* BaseModel ({} - {}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.decoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def encode(self, x, t = None, c = None):
        if hasattr(self, 'en_cemb'):
            if c is not None:
                c = self.en_cemb(c)
                if self.merge_ct:
                    if self.combine_condition == 'add':
                        t = add_embedding(t, c, self.channel_dim)
                    elif self.combine_condition == 'cat':
                        t = concat_embedding(t, c, self.channel_dim)
                else:
                    if self.combine_condition == 'add':
                        x = add_embedding(x, c, self.channel_dim)
                    elif self.combine_condition == 'cat':
                        x = concat_embedding(x, c, self.channel_dim)
            elif self.combine_condition == 'cat':
                logger.error('Condition is necessary for concatenation.')
                exit(1)
        res = self.encoder(x, t)
        return res
    ### usually t-embedding should be the same for encoder and decoder
    def decode(self, z, t = None, c = None):
        if hasattr(self, 'de_cemb'):
            if c is not None:
                c = self.de_cemb(c)
                if type(c) == list:
                    c = c[0]
                if self.combine_condition == 'add':
                    z = add_embedding(z, c, self.channel_dim)
                elif self.combine_condition == 'cat':
                    z = concat_embedding(z, c, self.channel_dim)
            elif self.combine_condition == 'cat':
                logger.error('Condition is necessary for concatenation.')
                exit(1)
        return self.decoder(z, t)

    def forward(self, x, t = None, c = None, return_z = False):
        if t is not None:
            if self.with_time_embedding:
                t = self.t_embed(t)
            else:
                logger.warning("There is no time embedding for this model, the input time step will be ignored.")
                t = None
        if (not self.is_conditional) and (c is not None):
            logger.warning("There is no condition embedding for this model, the input condition will be ignored.")
            c = None
        if not self.with_time_embedding:
            z = self.encode(x, c = c)
        else:
            z = self.encode(x, t = t, c = c)
        if not self.with_time_embedding:
            res = self.decode(z, c = c)
        else:
            res = self.decode(z, t = t, c = c)
        if return_z:
            return res, z
        return res
    def get_latent_shape(self):
        if self.data_form == DataForm.IMG and not self.encoder.vector_embedding:
            return [self.encoder.out_channel, ] + \
                        [int(im_size / (2** len(self.encoder.down_channels))) for im_size in self.encoder.image_size ]
        if self.data_form == DataForm.PCD and self.encoder.pointwise == True:
                return [self.encoder.point_num, ] + [self.encoder.out_channel,]        
        ### latent embedding is a vector
        return [self.encoder.out_channel]

    def get_input_shape(self):
        if self.data_form == DataForm.PCD:
            return [self.encoder.point_num, ] + [self.in_channel,]      
        if self.data_form == DataForm.IMG:
            return [self.in_channel, ] + self.encoder.image_size
        ### vector
        return [self.in_channel]
        
    def get_output_shape(self):
        if self.data_form == DataForm.PCD:
            return [self.decoder.point_num, ] + [self.out_channel,]      
        if self.data_form == DataForm.IMG:
            return [self.out_channel, ] + self.decoder.image_size
        ### vector
        return [self.out_channel]
    @property
    def device(self):
        return next(self.parameters()).device
    

class HBaseModel(BaseModel):
    def __init__(self, model_config):
        encoder_config = model_config['encoder']
        encoder_config['return_features'] = True
        super().__init__(model_config)
        assert self.data_form == DataForm.IMG, "Currently, HSeM only support image data."

        self.inter_mode = model_config.get('interpolation_mode', 'nearest')
        combine = model_config.get('combine', 'cat')
        apply_fft = model_config.get('apply_fft', False)
        ### additional layers beyond normal segmentation model
        self.combine_layer = CombineLayer(in_channels=[self.decoder.image_channel,] * (len(self.decoder.up_path) + 1),
                                            out_channel=self.decoder.image_channel, 
                                            target_size=self.decoder.image_size,
                                            mode = self.inter_mode,
                                            combine = combine, 
                                            apply_fft=apply_fft)
        self.channel_recover = False
        if self.decoder_name in ['Swin', 'SwinU', 'SwinD', 'VMamba', 'VMambaU', 'VMambaD']:
            self.channel_recover = True
            self.abbs = nn.ModuleList([VMambaBlock(dim = self.encoder.dim,
                                    in_channel=upc, 
                                    out_channel=upc,
                                    normalization='layer',
                                    activation='silu',
                                    time_channel=self.final_time_channel) 
                                    for upc in self.decoder.up_path])
        else:
            self.abbs = nn.ModuleList([ResConvBlock(dim = self.encoder.dim, 
                                    in_channel=upc, 
                                    out_channel=upc,
                                    normalization='group',
                                    num_groups=16,
                                    activation='relu',
                                    time_channel=self.final_time_channel) 
                                    for upc in self.decoder.up_path])
                            
            
        self.final_projs = nn.ModuleList([ConvBlock(dim = self.encoder.dim, 
                                in_channel=upc, 
                                out_channel=self.decoder.image_channel, 
                                kernel_size=1,
                                activation=None, 
                                norm=None) for upc in self.decoder.up_path])
        
        
    def __str__(self):
        _str = '********************* HBaseModel ({} - {}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.decoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def forward(self, x, t = None, c = None, return_z = False):
        if t is not None:
            if self.with_time_embedding:
                t = self.t_embed(t)
            else:
                logger.warning("There is no time embedding for this model, the input time step will be ignored.")
                t = None
        if (not self.is_conditional) and (c is not None):
            logger.warning("There is no condition embedding for this model, the input condition will be ignored.")
            c = None

        if not self.with_time_embedding:
            z = self.encode(x, c = c)
        else:
            z = self.encode(x, t = t, c = c)
        en_feature = z
        if type(en_feature) == tuple:
            en_feature = en_feature[0]
        #### t is None
        ## de_features is None
        if not self.with_time_embedding:
            x, de_features = self.decode(z, c = c)
        else:
            x, de_features = self.decode(z, t = t, c = c)
        features = [en_feature,] + de_features
        h_x = []
        for i in range(len(features)):
            fi = self.abbs[i] ( features[i], t)
            if self.channel_recover:
                fi = channel_recover(fi)
            h_x.append(self.final_projs[i](fi))
        x = self.combine_layer(h_x + [x,])
        if return_z:
            return (x, h_x), z
        return x, h_x
    
##### old implementation, saved only for loading the old checkpoints
# class HBaseModel(BaseModel):
#     def __init__(self, model_config):
#         encoder_config = model_config['encoder']
#         encoder_config['return_features'] = True
#         activation = encoder_config.get('activation', 'relu')
#         normalization = encoder_config.get('normalization', 'group')
#         num_groups = encoder_config.get('num_groups', 8)
#         super().__init__(model_config)
#         assert self.data_form == DataForm.IMG, "Currently, HSeM only support image data."

#         self.inter_mode = model_config.get('interpolation_mode', 'nearest')
#         combine = model_config.get('combine', 'cat')
#         ### additional layers beyond normal SeM
#         self.combine_layer = CombineLayer(in_channels=[self.decoder.image_channel,] * (len(self.decoder.up_path) + 1),
#                                             out_channel=self.decoder.image_channel, 
#                                             target_size=self.decoder.image_size,
#                                             mode = self.inter_mode,
#                                             combine = combine)
#         self.channel_recover = False
#         if self.decoder_name in ['Swin', 'SwinU', 'VMamba', 'VMambaU']:
#             self.channel_recover = True
#         if not self.with_time_embedding:
#             self.convs = nn.ModuleList([ResConvBlock(dim = self.encoder.dim, 
#                                     in_channel=upc, 
#                                     out_channel=upc, 
#                                     activation=activation, 
#                                     norm=normalization,
#                                     num_groups=num_groups) for upc in self.decoder.up_path])
#             self.final_convs = nn.ModuleList([ConvBlock(dim = self.encoder.dim, 
#                                     in_channel=upc, 
#                                     out_channel=self.decoder.image_channel, 
#                                     activation=None, 
#                                     norm=None) for upc in self.decoder.up_path])
#         else:
#             self.convs = nn.ModuleList([ResConvTBlock(dim = self.encoder.dim, 
#                                     in_channel=upc, 
#                                     out_channel=upc, 
#                                     time_channel=self.final_time_channel,
#                                     activation=activation, 
#                                     norm=normalization,
#                                     num_groups=num_groups) for upc in self.decoder.up_path])
#             self.final_convs = nn.ModuleList([ConvTBlock(dim = self.encoder.dim, 
#                                     in_channel=upc, 
#                                     out_channel=self.decoder.image_channel, 
#                                     time_channel=self.final_time_channel,
#                                     activation=None, 
#                                     norm=None) for upc in self.decoder.up_path])
        
        
#     def __str__(self):
#         _str = '********************* BaseModel ({}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.encoder.__str__(), self.decoder.__str__())
#         return _str
#     def forward(self, x, t = None, c = None):
#         if t is not None:
#             if self.with_time_embedding:
#                 t = self.t_embed(t)
#             else:
#                 logger.warning("There is no time embedding for this model, the input time step will be ignored.")

#         if not self.with_time_embedding:
#             z = self.encode(x, c = c)
#         else:
#             z = self.encode(x, t = t, c = c)
#         en_feature = z
#         if type(en_feature) == tuple:
#             en_feature = en_feature[0]
#         if not self.with_time_embedding:
#             x, de_features = self.decode(z, c = c)
#         else:
#             x, de_features = self.decode(z, t = t, c = c)
#         features = [en_feature,] + de_features
#         h_x = []
#         for i in range(len(features)):
#             if self.channel_recover:
#                 features[i] = channel_recover(features[i])
#             h_x.append(self.final_convs[i]( self.convs[i] ( features[i], t = t), t = t))
#         x = self.combine_layer(h_x + [x,])
#         return x, h_x

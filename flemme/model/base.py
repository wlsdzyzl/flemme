# auto-encoder for 2D image and 3D point cloud
import torch
import torch.nn as nn
from flemme.utils import DataForm
from flemme.logger import get_logger
from flemme.model.embedding import get_embedding, add_embedding, concat_embedding
from flemme.block import CombineLayer, ResConvBlock, ConvBlock, \
    channel_recover, PositionEmbeddingBlock
from flemme.utils import DataForm
from flemme.config import module_config
logger = get_logger('model.base')

class BaseModel(nn.Module):
    ''' 
    Base model for ae and vae. It follows a encoder-decoder structure, the output can be reconstruction or predicted mask.
    Loss is not specified.
    '''
    def __init__(self, model_config, create_encoder_func):
        super().__init__()
        encoder_config = model_config.get('encoder', None)
        assert encoder_config is not None, 'There is no encoder configuration.'

        self.in_channel = encoder_config.get('in_channel')
        if 'out_channel' not in encoder_config:
            encoder_config['out_channel'] = self.in_channel
        self.out_channel = encoder_config.get('out_channel')
        ### time_embedding
        self.time_channel = model_config.get('time_channel', 0)
        self.time_injection = model_config.get('time_injection', 'gate_bias')
        
        encoder_config['time_channel'] = self.time_channel
        encoder_config['time_injection'] = self.time_injection
        
        self.with_time_embedding = (self.time_channel > 0)
        self.time_act = model_config.get('time_activation', 'silu')
        ### time step embeddin
        if self.with_time_embedding:
            logger.info("Construct base model with time embedding.")
            self.t_embed = PositionEmbeddingBlock(out_channel=self.time_channel, activation=self.time_act)

        ### condition_embedding
        en_emb_config, de_emb_config = None, None
        self.condition_for_encoder = False 
        self.condition_for_decoder = False
        cemb_config = model_config.get('condition_embedding', None)
        
        if cemb_config is not None:
            en_emb_config = cemb_config.get('encoder', None)
            de_emb_config = cemb_config.get('decoder', None)
            self.combine_condition = cemb_config.get('combine_condition', 'add')
            self.condition_injection = cemb_config.get('condition_injection', 'gate_bias')
            self.condition_first = cemb_config.get('condition_first', False)
            encoder_config['condition_injection'] = self.condition_injection
            encoder_config['condition_first'] = self.condition_first
            assert self.combine_condition in ['add', 'cat', 'injection'],\
                "combine_condition should be one of ['add', 'cat', 'injection']"
        if en_emb_config is not None:
            logger.info("Create conditional embedding for encoder.")
            self.en_cemb = get_embedding(en_emb_config)
            self.condition_for_encoder = True
            if self.combine_condition == 'cat':
                ### should we use a convolution to project the catted feature to the encoder input space ?
                encoder_config['in_channel'] += self.en_cemb.out_channel
            elif self.combine_condition == 'injection':
                encoder_config['condition_channel'] = self.en_cemb.out_channel

        if de_emb_config is not None:
            logger.info("Create conditional embedding for decoder.")
            ### like an auto-regressive model
            if de_emb_config == 'same_as_encoder':
                logger.info('Using encoder\'s condition embedding to compute condition embedding for decoder.')
                if not self.condition_for_encoder:
                    logger.error('This is no condition embedding for encoder.')
                    exit(1)
                self.de_cemb = self.en_cemb
            else:
                self.de_cemb = get_embedding(de_emb_config)
            self.condition_for_decoder = True

            if self.combine_condition == 'cat':
                encoder_config['latent_channel'] += self.de_cemb.out_channel
            elif self.combine_condition == 'injection':
                encoder_config['decoder_condition_channel'] = self.de_cemb.out_channel
        
        #### check if there is a specified decoder configuration
        decoder_config = None
        specified_decoder_config = model_config.get('decoder', None)
        if specified_decoder_config is not None:
            decoder_config = encoder_config.copy()
            for k in specified_decoder_config:
                decoder_config[k] = specified_decoder_config[k]

        self.encoder_name = encoder_config.get('name')
        self.decoder_name = decoder_config.get('name') if decoder_config is not None else self.encoder_name

        ### point2encoder
        if self.encoder_name in ['PointNet2', 'PointTrans2', 'PointMamba2']:
            if self.decoder_name in ['PointNet2', 'PointTrans2', 'PointMamba2']:
                encoder_config['is_point2decoder'] = True
            else:
                encoder_config['is_point2decoder'] = False
        
        self.encoder, self.decoder = create_encoder_func(encoder_config=encoder_config, return_decoder=(decoder_config is None))

        ### create decoder
        if decoder_config is not None:
            decoder_config['latent_channel'] = self.encoder.out_channel
            if hasattr(self.encoder, 'out_channels'):
                decoder_config['latent_channels'] = self.encoder.out_channels
            self.decoder = create_encoder_func(encoder_config=decoder_config, return_encoder=False)[1]
        
        if self.condition_for_encoder and self.combine_condition == 'add':
            assert self.in_channel == self.en_cemb.out_channel, \
                "condition embedding of encoder and input data should have the same shape for addition."
        if self.condition_for_decoder and self.combine_condition == 'add':
                assert self.encoder.out_channel == self.de_cemb.out_channel, \
                    "condition embedding of decoder and the output of encoder should have the same shape for addition."

        self.is_generative = False
        self.is_conditional = self.condition_for_encoder or self.condition_for_decoder
        self.is_supervised = False

        self.loss_reduction = model_config.get('loss_reduction', 'mean')
        self.data_form = self.encoder.data_form
        self.channel_dim = self.encoder.channel_dim 
        self.feature_channel_dim = self.encoder.feature_channel_dim
    def __str__(self):
        _str = '********************* BaseModel ({} - {}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.decoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def encode(self, x, t = None, c = None):
        if self.condition_for_encoder:
            if c is not None:
                c = self.en_cemb(c)
                if self.combine_condition == 'add':
                    x = add_embedding(x, c, self.channel_dim)
                    c = None
                elif self.combine_condition == 'cat':
                    x = concat_embedding(x, c, self.channel_dim)
                    c = None
            elif self.combine_condition == 'cat':
                logger.error('Condition is necessary for concatenation.')
                exit(1)
        else:
            logger.debug('Model\'s encoder cannot compute condition embedding. Input condition will be ignored.')
            c = None
        res = self.encoder(x, t = t, c = c)
        return res
    ### usually t-embedding should be the same for encoder and decoder
    def decode(self, z, t = None, c = None):
        if self.condition_for_decoder:
            if c is not None:
                c = self.de_cemb(c)
                ## when happened?
                # if type(c) == list:
                #     c = c[0]
                if self.combine_condition == 'add':
                    z = add_embedding(z, c, self.feature_channel_dim)
                    c = None
                elif self.combine_condition == 'cat':
                    z = concat_embedding(z, c, self.feature_channel_dim)
                    c = None
            elif self.combine_condition == 'cat':
                logger.error('Condition is necessary for concatenation.')
                exit(1)
        else:
            logger.debug('Model\'s decoder cannot compute condition embedding. Input condition will be ignored.')
            c = None
        return self.decoder(z, t = t, c = c)

    def forward(self, x, t = None, c = None, return_z = False):
        if t is not None:
            if self.with_time_embedding:
                t = self.t_embed(t)
            else:
                logger.warning("There is no time embedding for this model, the input time step will be ignored.")
                t = None
        if self.with_time_embedding:
            z = self.encode(x, t = t, c = c)
            res = self.decode(z, t = t, c = c)
        else:
            z = self.encode(x, c = c)
            res = self.decode(z, c = c)
        if return_z:
            return res, z
        return res
    def get_latent_shape(self):
        if self.data_form == DataForm.IMG and not self.encoder.vector_embedding:
            return [self.encoder.out_channel, ] + \
                        [int(im_size / (2** self.encoder.d_depth)) for im_size in self.encoder.image_size ]
        if self.data_form == DataForm.PCD and not self.encoder.vector_embedding:
                return [self.encoder.point_num, self.encoder.out_channel,]        
        ### latent embedding is a vector
        return [self.encoder.out_channel]

    def get_input_shape(self):
        if self.data_form == DataForm.PCD:
            return [self.encoder.point_num, self.in_channel,]      
        if self.data_form == DataForm.IMG:
            return [self.in_channel, ] + self.encoder.image_size
        ### vector
        return [self.in_channel]
        
    def get_output_shape(self):
        if self.data_form == DataForm.PCD:
            return [self.decoder.point_num, self.out_channel,]      
        if self.data_form == DataForm.IMG:
            return [self.out_channel, ] + self.decoder.image_size
        ### vector
        return [self.out_channel]
    @property
    def device(self):
        return next(self.parameters()).device
    

class HBaseModel(BaseModel):
    def __init__(self, model_config, create_encoder_func):
        encoder_config = model_config['encoder']
        encoder_config['return_feature_list'] = True
        super().__init__(model_config, create_encoder_func)
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
        self.final_feature_extraction = model_config.get('final_feature_extraction', True)
        if self.final_feature_extraction:
            if self.decoder_name in ['ViT', 'ViTU', 'ViTD', 'Swin', 'SwinU', 'SwinD', 'VMamba', 'VMambaU', 'VMambaD']:
                self.channel_recover = True
                if module_config['mamba']:
                    from flemme.block import VMambaBlock
                    SeqModelingBlock = VMambaBlock
                else:
                    from flemme.block import VisionTransformerBlock
                    SeqModelingBlock = VisionTransformerBlock
                self.abbs = nn.ModuleList([SeqModelingBlock(dim = self.encoder.dim,
                                        in_channel=upc, 
                                        out_channel=upc,
                                        normalization='layer',
                                        activation='silu',
                                        time_channel=self.time_channel) 
                                        for upc in self.decoder.up_path])
            else:
                self.abbs = nn.ModuleList([ResConvBlock(dim = self.encoder.dim, 
                                        in_channel=upc, 
                                        out_channel=upc,
                                        normalization='group',
                                        num_norm_groups=16,
                                        activation='relu',
                                        time_channel=self.time_channel) 
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
        if self.with_time_embedding:
            z = self.encode(x, t = t, c = c)
        else:
            z = self.encode(x, c = c)
        en_feature = z
        if type(en_feature) == tuple:
            en_feature = en_feature[0]
        #### t is None
        ## de_features is None
        if self.with_time_embedding:
            x, de_features = self.decode(z, t = t, c = c)
        else:
            x, de_features = self.decode(z, c = c)
        features = [en_feature,] + de_features
        h_x = []
        for i in range(len(features)):
            if self.final_feature_extraction:
                fi = self.abbs[i] ( features[i], t)
            if self.channel_recover:
                fi = channel_recover(fi)
            h_x.append(self.final_projs[i](fi))
        x = self.combine_layer(h_x + [x,])
        if return_z:
            return (x, h_x), z
        return x, h_x
# auto-encoder for 2D image and 3D point cloud
import torch.nn as nn
from flemme.utils import DataForm
from flemme.logger import get_logger
from flemme.model.embedding import get_embedding, add_embedding, concat_embedding
from flemme.utils import DataForm
from flemme.encoder import create_encoder
logger = get_logger('model.half')
### some times we may want to construct a model with only encoder or decoder with conditional embedding.
### this might be useful, for a classification model
class OnlyEncoder(nn.Module):
    ''' 
    Base model for ae and vae. It follows a encoder-decoder structure, the output can be reconstruction or predicted mask.
    Loss is not specified.
    '''
    def __init__(self, model_config):
        super().__init__()
        encoder_config = model_config.get('encoder', None)

        assert encoder_config is not None, 'There is no encoder configuration.'
        self.encoder_name = encoder_config.get('name')

        self.in_channel = encoder_config.get('in_channel')
        ### condition_embedding
        cemb_config = model_config.get('condition_embedding', None)
        if cemb_config is not None:
            self.combine_condition = cemb_config.pop('combine_condition', 'add')
            logger.info("Create conditional embedding for encoder.")
            self.en_cemb = get_embedding(cemb_config)
            if self.combine_condition == 'cat':
                encoder_config['encoder_additional_in_channel'] = \
                    encoder_config.get('encoder_additional_in_channel', 0) + self.en_cemb.out_channel

            else:
                assert self.in_channel == self.en_cemb.out_channel, \
                    "condition embedding of encoder and input data should have the same shape for addition."

        self.encoder = create_encoder(encoder_config=encoder_config, return_decoder=False)[0]

        self.is_generative = False
        self.is_conditional = hasattr(self, 'en_cemb')
        self.is_supervised = False

        self.loss_reduction = model_config.get('loss_reduction', 'mean')
        self.data_form = self.encoder.data_form
        self.channel_dim = -1 if self.data_form == DataForm.PCD else 1
        self.out_channel = self.encoder.out_channel
    def __str__(self):
        _str = '********************* OnlyEncoder ({}) *********************\n------- Encoder -------\n{}'.format(self.encoder_name, self.encoder.__str__())
        return _str
    def encode(self, x, c = None):
        if c is not None:
            if hasattr(self, 'en_cemb'):
                c_emb = self.en_cemb(c)
                if self.combine_condition == 'add':
                    # print(x.shape, c_emb.shape)
                    x = add_embedding(x, c_emb, self.channel_dim)
                elif self.combine_condition == 'cat':
                    x = concat_embedding(x, c_emb, self.channel_dim)
        res = self.encoder(x)
        return res

    def forward(self, x, c = None):
        if (not self.is_conditional) and (c is not None):
            logger.warning("There is no condition embedding for this model, the input condition will be ignored.")
        return self.encode(x, c = c)

    def get_input_shape(self):
        if self.data_form == DataForm.PCD:
            return [self.encoder.point_num, ] + [self.in_channel,]      
        if self.data_form == DataForm.IMG:
            return [self.in_channel, ] + self.encoder.image_size
        ### vector
        return [self.in_channel]
    
    def get_output_shape(self):
        if self.data_form == DataForm.PCD and not self.encoder.vector_embedding:
            return [self.encoder.point_num, ] + [self.out_channel,]      
        if self.data_form == DataForm.IMG and not self.encoder.vector_embedding:
            return [self.out_channel, ] + self.encoder.image_size
        return [self.out_channel]
    
    def get_latent_shape(self):
        return self.get_output_shape()
    
    

class OnlyDecoder(nn.Module):
    ''' 
    Base model for ae and vae. It follows a encoder-decoder structure, the output can be reconstruction or predicted mask.
    Loss is not specified.
    '''
    def __init__(self, model_config):
        super().__init__()
        decoder_config = model_config.get('encoder', None)
        decoder_config = decoder_config or model_config.get('decoder', None)

        assert decoder_config is not None, 'There is no decoder configuration.'
        self.decoder_name = decoder_config.get('name')
        self.in_channel = decoder_config.get('decoder_in_channel')
        self.out_channel = decoder_config.get('out_channel')

        ### condition_embedding
        cemb_config = model_config.get('condition_embedding', None)
        if cemb_config is not None:
            self.combine_condition = cemb_config.pop('combine_condition', 'add')
            logger.info("Create conditional embedding for decoder.")
            self.de_cemb = get_embedding(cemb_config)
            if self.combine_condition == 'cat':
                decoder_config['decoder_additional_in_channel'] = \
                     decoder_config.get('decoder_additional_in_channel', 0) + self.de_cemb.out_channel
            else:
                assert self.in_channel == self.de_cemb.out_channel, \
                    "condition embedding of decoder and the output of encoder should have the same shape for addition."
        ### create decoder
        self.decoder = create_encoder(encoder_config=decoder_config, return_encoder=False)[1]
        self.is_generative = False
        self.is_conditional = hasattr(self, 'de_cemb')
        self.is_supervised = False
        self.loss_reduction = model_config.get('loss_reduction', 'mean')
        self.data_form = self.decoder.data_form
        self.channel_dim = -1 if self.data_form == DataForm.PCD else 1

    def __str__(self):
        _str = '********************* OnlyDecoder ({}) *********************\n------- Decoder -------\n{}'.format(self.decoder_name, self.decoder.__str__())
        return _str
    ### usually t-embedding should be the same for encoder and decoder
    def decode(self, z, c = None):
        if c is not None:
            if hasattr(self, 'de_cemb'):
                c_emb = self.de_cemb(c)
                if self.combine_condition == 'add':
                    z = add_embedding(z, c_emb, self.channel_dim)
                elif self.combine_condition == 'cat':
                    z = concat_embedding(z, c_emb, self.channel_dim)
        return self.decoder(z)

    def forward(self, z, c = None):
        if (not self.is_conditional) and (c is not None):
            logger.warning("There is no condition embedding for this model, the input condition will be ignored.")
        return self.decode(z, c = c)
    ### decoder will always project the latent into original shape
    def get_output_shape(self):
        if self.data_form == DataForm.PCD:
            return [self.decoder.point_num, ] + [self.out_channel,]      
        if self.data_form == DataForm.IMG:
            return [self.out_channel, ] + self.decoder.image_size
        ### vector
        return [self.out_channel]
    
    #### should never be called
    # def get_input_shape(self):
    #     if self.data_form == DataForm.PCD and not self.decoder.vector_embedding:
    #         return [self.decoder.point_num, ] + [self.in_channel,]      
    #     if self.data_form == DataForm.IMG and not self.decoder.vector_embedding:
    #         return [self.in_channel, ] + self.decoder.image_size
    #     ### vector
    #     return [self.in_channel]


#### ddpm of 2D image and 3D point cloud
### ldm (latent diffusion model): the diffusion process is performed on latent space.
### a pre-trained auto-encoder (usually, a variational auto-encoder) is needed. 
from .ddpm import DiffusionProbabilistic as DDPM
from .ddim import DiffusionImplicit as DDIM
from .ae import AutoEncoder as AE
from .vae import VariationalAutoEncoder as VAE
from .edm import EDM
from flemme.utils import freeze, unfreeze, load_checkpoint
from flemme.logger import get_logger

import torch
import torch.nn as nn
logger = get_logger('model.ldm')
### perform diffusion on latent space
### LDM needs a pre-trained auto-encoder
supported_ae_models = {'AE': AE, 'VAE': VAE}
supported_diff_models = {'DDPM': DDPM, 'DDIM': DDIM, 'EDM': EDM}

def create_ae_model(ae_config, create_encoder_fn):
    model_name = ae_config.pop('name', 'VAE')
    if not model_name in supported_ae_models:
        raise RuntimeError(f'Unsupported model class: {model_name}')
    ae_model = supported_ae_models[model_name](ae_config, create_encoder_fn)
    return ae_model, model_name

def create_diff_model(diff_config, create_encoder_fn):
    model_name = diff_config.pop('name', 'DDPM')
    if not model_name in supported_diff_models:
        raise RuntimeError(f'Unsupported model class: {model_name}')
    diff_model = supported_diff_models[model_name](diff_config, create_encoder_fn)
    return diff_model, model_name


class LatentDiffusion(nn.Module):

    def __init__(self, model_config, create_encoder_fn):
        super().__init__()
        self.loss_reduction = model_config.get('loss_reduction', 'mean')

        ae_config = model_config.pop('autoencoder', None)
        ae_path = model_config.get('autoencoder_path', None)
        self.freezed_ae = model_config.get('freezed_ae', True)

        diff_config = model_config.pop('diffusion', None)
        
        assert ae_config is not None, 'LDM needs a pre-trained \
            auto-encoder to get the embedding of input data.'
        assert diff_config is not None, 'LDM needs a diffusion model to generate latents.'            
        ae_config['loss_reduction'] = self.loss_reduction
        diff_config['loss_reduction'] = self.loss_reduction
        self.ae_model, self.ae_model_name = create_ae_model(ae_config, create_encoder_fn)
        self.diff_model, self.diff_model_name = create_diff_model(diff_config, create_encoder_fn)
        
        if ae_path is not None:
            load_checkpoint(ae_path, self.ae_model)
            logger.info('Using pre-trained AE model.')
        if self.freezed_ae: freeze(self.ae_model)
        else:
            logger.info('AE model is not freezed, DDPM and AE will be trained at the same time.')

        self.is_conditional = self.ae_model.is_conditional or self.diff_model.is_conditional
        self.is_generative = True
        self.is_supervised = False
        if self.is_conditional:
            logger.info('Construct a conditional latent diffusion model.')
        self.data_form = self.ae_model.data_form
    @property
    def device(self):
        return self.diff_model.device
    def sample(self, zt, c = None, 
            return_processing = False, 
            **kwargs):
        ac, ec = None, None
        if self.ae_model.is_conditional:
            ac = c 
        if self.diff_model.is_conditional:
            ec = c
        z0 = self.diff_model.sample(xt = zt, c = ec, 
                return_processing = return_processing, 
                **kwargs)
        ### return processing
        if isinstance(z0, tuple) or isinstance(z0, list):
            return type(z0)(self.ae_model.decode(_z) for _z in z0)
        return self.ae_model.decode(z = z0, c = ac)
    def __str__(self):
        _str = "********************* LatentDiffusion *********************\n{}{}"\
            .format(self.ae_model.__str__(), self.diff_model.__str__())
        return _str
    def freeze_ae(self):
        freeze(self.ae_model)
        self.freezed_ae = True
    def unfreeze_ae(self):
        unfreeze(self.ae_model)
        self.freezed_ae = False
    def get_input_shape(self):
        return self.ae_model.get_input_shape()
    def get_output_shape(self):
        return self.ae_model.get_output_shape()
    def get_latent_shape(self):
        return self.diff_model.get_latent_shape()
    def get_loss_name(self):
        loss_names = self.diff_model.get_loss_name() 
        if not self.freeze_ae:
            loss_names += self.ae_model.get_loss_name()
        return loss_names
    ### ldm's forward will not call the forward function of auto-encoder
    def forward(self, x0, c = None):
        ac, ec = None, None
        if self.ae_model.is_conditional:
            ac = c 
        if self.diff_model.is_conditional:
            ec = c
        gauss = None
        z = self.ae_model.encode(x0, c = ac)
        if not torch.is_tensor(z):
            gauss = z
            z = gauss.sample()
        ### forward call sample, direct get the final reconstruction
        res = self.diff_model(x = z, c = ec)
        res['latent'] = z
        res['recon'] = self.ae_model.decode(res['recon_dpm'], c = ac)
        if gauss is not None: res['gaussian'] = gauss
        return res
    def compute_loss(self, x0, c = None):
        res = self.forward(x0, c = c)
        ## compute loss will not use the condition
        loss, _ = self.diff_model.compute_loss(res['latent'], res = res)
        if not self.freezed_ae:
            loss = loss + self.ae_model.compute_loss(x0, res = res)
        return loss, res

    

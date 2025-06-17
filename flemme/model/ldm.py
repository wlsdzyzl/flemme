#### ddpm of 2D image and 3D point cloud
### ldm (latent diffusion model): the diffusion process is performed on latent space.
### a pre-trained auto-encoder (usually, a variational auto-encoder) is needed. 

from .ddpm import DiffusionProbabilistic as DDPM
from .ddim import DiffusionImplicit as DDIM
from .ae import AutoEncoder as AE
from .vae import VariationalAutoEncoder as VAE
from .edm import EDM
from flemme.trainer_utils import load_checkpoint, freeze, unfreeze
from flemme.logger import get_logger
import torch
import torch.nn as nn
logger = get_logger('model.ldm')
### perform diffusion on latent space
### LDM needs a pre-trained auto-encoder
supported_ae_models = {'AE': AE, 'VAE': VAE}
supported_diff_models = {'DDPM': DDPM, 'DDIM': DDIM, 'EDM': EDM}

def create_ae_model(ae_config):
    model_name = ae_config.pop('name', 'VAE')
    if not model_name in supported_ae_models:
        raise RuntimeError(f'Unsupported model class: {model_name}')
    ae_model = supported_ae_models[model_name](ae_config)
    return ae_model, model_name

def create_diff_model(diff_config):
    model_name = diff_config.pop('name', 'DDPM')
    if not model_name in supported_diff_models:
        raise RuntimeError(f'Unsupported model class: {model_name}')
    diff_model = supported_diff_models[model_name](diff_config)
    return diff_model, model_name


class LatentDiffusion(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        ae_config = model_config.pop('autoencoder', None)
        ae_path = model_config.get('autoencoder_path', None)
        self.freezed_ae = model_config.get('freezed_ae', True)

        diff_config = model_config.pop('diffusion', None)
        
        assert ae_config is not None, 'LDM needs a pre-trained \
            auto-encoder to get the embedding of input data.'
        assert diff_config is not None, 'LDM needs a diffusion model to generate latents.'            
        self.ae_model, self.ae_model_name = create_ae_model(ae_config)
        self.diff_model, self.diff_model_name = create_diff_model(diff_config)
        
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
            return_processing = False, **kwargs):
        ac, ec = None, None
        if self.ae_model.is_conditional:
            ac = c 
        if self.diff_model.is_conditional:
            ec = c
        z0 = self.diff_model.sample(xt = zt, c = ec, 
                return_processing = return_processing, 
                **kwargs)
        #### what is this for?
        # if isinstance(z, tuple) or isinstance(z, list):
        #     return type(z)(self.ae_model.decode(_z) for _z in z)
        return self.ae_model.decode(z = z0, c = ac)
    def __str__(self):
        _str = "********************* LatentDiffusion *********************\n{}{}"\
            .format(self.ae_model.__str__(), self.diff_model.__str__())
        return _str
    ### ldm's forward will not call the forward function of auto-encoder
    def forward(self, x, c = None, **kwargs):
        ac = None
        if self.ae_model.is_conditional:
            ac = c 
        gauss = None
        z = self.ae_model.encode(x0, c = ac)
        if not torch.is_tensor(z):
            gauss = z
            z = gauss.sample()
        # print(z.max(), z.min())
        ### forward call sample, direct get the final reconstruction
        res = self.diff_model(x = z, c = c, **kwargs)
        res['latent'] = z
        res['recon_latent'] = res['recon']
        res['recon'] = self.ae_model.decode(res['recon_latent'], c = ac)
        if gauss is not None: res['gaussian'] = gauss
        return res
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
    def compute_loss(self, x0, c = None):
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
        
        loss, res = self.diff_model.compute_loss(z, c = ec)
        if res is not None:
            res['recon_latent'] = res['recon']
            res['latent'] = z
            res['recon'] = self.ae_model.decode(res['recon_latent'], c = ac)
            if gauss is not None: res['gaussian'] = gauss
        ### ae loss, althrough ae_model can be freezed.
        if not self.freezed_ae:
            loss = loss + self.ae_model.compute_loss(x0, res = res)[0]
        return loss, res

    

#### ddpm of 2D image and 3D point cloud
### ldm (latent diffusion model): the diffusion process is performed on latent space.
### a pre-trained auto-encoder (usually, a variational auto-encoder) is needed. 

from .ddpm import DiffusionProbabilistic as DDPM
from .ddim import DiffusionImplicit as DDIM
from .ae import AutoEncoder as AE
from .vae import VariationalAutoEncoder as VAE
from flemme.trainer_utils import load_checkpoint, freeze, unfreeze
from flemme.logger import get_logger
import torch
logger = get_logger('model.ldm')
### perform diffusion on latent space
### LDM needs a pre-trained auto-encoder
supported_ae_models = {'AE': AE, 'VAE': VAE}
def create_ae_model(ae_config):
    model_name = ae_config.pop('name', 'VAE')
    if not model_name in supported_ae_models:
        raise RuntimeError(f'Unsupported model class: {model_name}')
    ae_model = supported_ae_models[model_name](ae_config)
    return ae_model, model_name

class LatentDiffusionProbabilistic(DDPM):

    def __init__(self, model_config):
        super().__init__(model_config)
        ae_config = model_config.get('ae_model', None)
        assert ae_config is not None, 'LDM needs a pre-trained \
            auto-encoder to get the embedding of input data.'
        self.ae_model, self.ae_model_name = create_ae_model(ae_config)
        self.freezed_ae = model_config.get('freezed_ae', True)
        ae_path = model_config.get('ae_path', None)
        if ae_path is not None:
            load_checkpoint(ae_path, self.ae_model)
            logger.info('Using pre-trained AE model.')
        if self.freezed_ae: freeze(self.ae_model)
        else:
            logger.info('AE model is not freezed, DDPM and AE will be trained at the same time.')

        self.is_conditional = self.ae_model.is_conditional or self.eps_model.is_conditional
        if self.is_conditional:
            logger.info('Construct a conditional latent diffusion model.')
        self.data_form = self.ae_model.data_form
    def sample(self, zt, end_step = 100, c = None, clipped = True, 
               clip_range = (-1.0, 1.0), return_processing = False):
        ac, ec = None, None
        if self.ae_model.is_conditional:
            ac = c 
        if self.eps_model.is_conditional:
            ec = c
        z0 = super().sample(xt = zt, end_step = end_step, c = ec, clipped = clipped, 
            clip_range = clip_range, return_processing = return_processing)
        #### what is this for?
        # if isinstance(z, tuple) or isinstance(z, list):
        #     return type(z)(self.ae_model.decode(_z) for _z in z)
        return self.ae_model.decode(z = z0, c = ac)
    ### ldm's forward will not call the forward function of auto-encoder
    def forward(self, x, c = None, end_step = 100, clipped = True, clip_range = [-1.0, 1.0]):
        ac = None
        if self.ae_model.is_conditional:
            ac = c 
        z = self.ae_model.encode(x, c = ac)
        ### distribution, 
        ### sample results may lead to training instability
        if not torch.is_tensor(z):
            z = z.mean
        # print(z.max(), z.min())
        ### forward call sample, direct get the final reconstruction
        res = super().forward(x = z, c = c, end_step = end_step, 
            clipped = clipped, clip_range = clip_range)
        res['latent'] = z
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
    def get_loss_name(self):
        loss_names = super().get_loss_name() 
        if not self.freeze_ae:
            loss_names += self.ae_model.get_loss_name()
        return loss_names
    def compute_loss(self, x0, c = None):
        ac, ec = None, None
        if self.ae_model.is_conditional:
            ac = c 
        if self.eps_model.is_conditional:
            ec = c

        ae_res = self.ae_model(x0, c = ac)
        loss = super().compute_loss(ae_res['latent'], c = ec)[0]
        ### ae loss, althrough ae_model can be freezed.
        if not self.freezed_ae:
            loss = loss + self.ae_model.compute_loss(x0, res = ae_res)[0]
        return loss, None
    
### almost the same with ldpm
class LatentDiffusionImplicit(DDIM):
    def __init__(self, model_config):
        super().__init__(model_config)
        ae_config = model_config.get('ae_model', None)
        assert ae_config is not None, 'LDM needs a pre-trained \
            auto-encoder to get the embedding of input data.'
        self.ae_model, self.ae_model_name = create_ae_model(ae_config)
        self.freezed_ae = model_config.get('freezed_ae', True)
        ae_path = model_config.get('ae_path', None)
        if ae_path is not None:
            load_checkpoint(ae_path, self.ae_model)
            logger.info('Using pre-trained AE model.')
        if self.freezed_ae: freeze(self.ae_model)
        else:
            logger.info('AE model is not freezed, DDPM and AE will be trained at the same time.')

        self.is_conditional = self.ae_model.is_conditional or self.eps_model.is_conditional
        if self.is_conditional:
            logger.info('Construct a conditional latent diffusion model.')
        self.data_form = self.ae_model.data_form
    def sample(self, zt, c = None, clipped = True, 
               clip_range = (-1.0, 1.0), return_processing = False):
        ac, ec = None, None
        if self.ae_model.is_conditional:
            ac = c 
        if self.eps_model.is_conditional:
            ec = c
        z0 = super().sample(xt = zt, c = ec, clipped = clipped, 
            clip_range = clip_range, return_processing = return_processing)
        #### what is this for?
        # if isinstance(z, tuple) or isinstance(z, list):
        #     return type(z)(self.ae_model.decode(_z) for _z in z)
        return self.ae_model.decode(z = z0, c = ac)
    ### ldm's forward will not call the forward function of auto-encoder
    def forward(self, x, c = None, clipped = True, clip_range = [-1.0, 1.0]):
        ac = None
        if self.ae_model.is_conditional:
            ac = c 
        z = self.ae_model.encode(x, c = ac)
        ### distribution, 
        ### sample results may lead to training instability
        if not torch.is_tensor(z):
            z = z.mean
        # print(z.max(), z.min())
        ### forward call sample, direct get the final reconstruction
        res = super().forward(x = z, c = c,
            clipped = clipped, clip_range = clip_range)
        res['latent'] = z
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
    def get_loss_name(self):
        loss_names = super().get_loss_name() 
        if not self.freeze_ae:
            loss_names += self.ae_model.get_loss_name()
        return loss_names
    def compute_loss(self, x0, c = None):
        ac, ec = None, None
        if self.ae_model.is_conditional:
            ac = c 
        if self.eps_model.is_conditional:
            ec = c

        ae_res = self.ae_model(x0, c = ac)
        loss = super().compute_loss(ae_res['latent'], c = ec)[0]
        ### ae loss, althrough ae_model can be freezed.
        if not self.freezed_ae:
            loss = loss + self.ae_model.compute_loss(x0, res = ae_res)[0]
        return loss, None
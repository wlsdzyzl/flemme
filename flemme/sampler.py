import torch
import numpy as np
from flemme.model import AE, VAE, DDPM, DDIM, LDPM, LDIM
from flemme.logger import get_logger
from flemme.config import module_config
logger = get_logger('sampler')
def create_sampler(model, device, sampler_config):
    sampler_name = sampler_config.pop('name', 'NormalSampler')
    logger.info(f'using {sampler_name}.')
    if sampler_name == 'NormalSampler':
        return NormalSampler(model = model, device = device, **sampler_config)
    # if sampler_name == 'MNSampler':
    #     return MultinomialSampler(model = model, device = device, **sampler_config)
    # if sampler_name == 'HybridSampler':
    #     return HybridSampler(model = model, device = device, **sampler_config)

### sample from normal distribution
class NormalSampler:
    def __init__(self, model, device = "cpu", rand_seed = None,  
                 sample_steps = -1,
                 clipped = None, clip_range = None, **kwargs):
        if len(kwargs) > 0:
            logger.debug('Redundant parameters: {}'.format(kwargs))
        self.model = model
        assert model.is_generative, \
                    "NormalSampler can only be constructed with generative model."
        self.is_conditional = model.is_conditional
        self.model_type = type(self.model)
        self.device = device
        if rand_seed is not None:
            torch.manual_seed(rand_seed)
        self.clipped = clipped
        self.clip_range = clip_range
        self.sample_steps = sample_steps
        if isinstance(model, DDPM):
            if self.sample_steps <= 0:
                self.sample_steps = model.num_steps
            assert self.sample_steps <= model.num_steps, \
                "Number of sample steps cannot be greater than num_steps."
        
    def decode(self, z, c = None):
        if self.model_type in [DDPM, LDPM]:
            return self.model.sample(z, end_step=self.sample_steps - 1, c = c,
                                        clipped=self.clipped, clip_range=self.clip_range)
        elif self.model_type in [DDIM, LDIM]:
            return self.model.sample(z, c = c, clipped=self.clipped,
                                clip_range=self.clip_range)
        ### AE, VAE and so on.
        return self.model.decode(z, c)
    
    ## interpolation, condition should be the same for inter data
    def interpolate(self, corner_latents = None, corner_num = 2, inter_num = 8, cond = None):
        if corner_latents is not None:
            corner_num = corner_latents.shape[0]
        if not (corner_num == 2 or corner_num == 4):
            logger.warning('Corner number should be 2 or 4.')
            corner_num = 2
            
        if corner_latents is None:
            latents = torch.randn(*( [corner_num, ] + self.model.get_latent_shape())).to(self.device)
        else:
            latents = corner_latents[0:corner_num].to(self.device)
            if latents.shape[0] < corner_num:
                logger.error("Insufficient corner latents. The number of corcer latents is at least 2.")
                exit(1)
        if not self.is_conditional:
            cond = None
        if corner_num == 2:
            
            latent_a, latent_b  = latents[0][None,:], latents[1][None,:]
            wa = np.linspace(0, 1, inter_num + 2)
            wb = 1 - wa
            inter_latent = torch.vstack([_wa * latent_a + _wb * latent_b for _wa, _wb in zip(wa, wb)])
            
        ### corner_num = 4
        if corner_num == 4:
            latent_a, latent_b, latent_c, latent_d  = \
                latents[0][None,:], latents[1][None,:], latents[2][None,:], latents[3][None,:]

            x = np.linspace(0, 1, inter_num + 2)
            y = np.linspace(0, 1, inter_num + 2)
            
            xx, yy = np.meshgrid(x,y)
            wa = (xx * yy).flatten()
            wb = ((1 - xx) * yy).flatten()
            wc = (xx * (1 - yy)).flatten()
            wd = ((1 - xx) * (1 - yy)).flatten()
            
            inter_latent = torch.vstack([_wa * latent_a + _wb * latent_b + _wc * latent_c + _wd * latent_d 
                for _wa, _wb, _wc, _wd in zip(wa, wb, wc, wd)])
        if self.is_conditional and cond is not None:
            # logger.info("condition for interpolation: {}".format(cond))
            cond = torch.stack([cond for _ in range(inter_latent.shape[0])])
        
        x_bar = self.decode(inter_latent, cond)
        # print(x_bar)
        return x_bar

    def generate_rand(self, n = 16, cond = None):
        if not self.is_conditional:
            cond = None
        z = torch.randn(*( [n, ] + self.model.get_latent_shape())).to(self.device)
        # print('!!!!!!!!!!!!!', z.shape)
        if cond is not None and (len(cond.shape) == 0 
                                 or not cond.shape[0] == n):
            ## all data use the same condition
            cond = torch.stack([cond for _ in range(n)])
        return self.decode(z, cond)
    
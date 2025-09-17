import torch
import numpy as np
from flemme.model import DDPM, DDIM, EDM
from flemme.logger import get_logger
from tqdm import tqdm

logger = get_logger('sampler')
def create_sampler(model, sampler_config):
    sampler_name = sampler_config.pop('name', 'NormalSampler')
    logger.info(f'using {sampler_name}.')
    if sampler_name == 'NormalSampler':
        return NormalSampler(model = model, **sampler_config)
    # if sampler_name == 'MNSampler':
    #     return MultinomialSampler(model = model, device = device, **sampler_config)
    # if sampler_name == 'HybridSampler':
    #     return HybridSampler(model = model, device = device, **sampler_config)

## batch generator with different random seed.
# class StackedRandomGenerator:
#     def __init__(self, device, seeds):
#         super().__init__()
#         self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

#     def randn(self, size, **kwargs):
#         assert size[0] == len(self.generators)
#         return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

#     def randn_like(self, input):
#         return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

#     def randint(self, *args, size, **kwargs):
#         assert size[0] == len(self.generators)
#         return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])



### sample from normal distribution
class NormalSampler:
    def __init__(self, model, rand_seed = None,  
                 num_sample_steps = -1,
                 clipped = None, clip_range = None, 
                 batch_size = 16, 
                 **kwargs):
        if len(kwargs) > 0:
            logger.debug('Redundant parameters: {}'.format(kwargs))
        self.model = model
        assert model.is_generative, \
                    "NormalSampler can only be constructed with generative model."
        self.is_conditional = model.is_conditional or model.is_supervised
        self.device = model.device
        self.random_generator = torch.Generator(self.device)
        if rand_seed is not None:
            self.random_generator = self.random_generator.manual_seed(rand_seed)

        self.clipped = clipped
        self.clip_range = clip_range
        self.num_sample_steps = num_sample_steps
        self.batch_size = batch_size
        if isinstance(model, DDPM):
            if self.num_sample_steps <= 0:
                self.num_sample_steps = model.num_steps
            assert self.num_sample_steps <= model.num_steps, \
                "Number of sample steps cannot be greater than num_steps."
            if not isinstance(model, DDIM) and self.num_sample_steps < model.num_steps:
                logger.warning('Small number of sample steps leads to fast but poor results. We recommand you to use DDIM.')
        
    def sample(self, z, c = None):
        z_batch = torch.split(z, self.batch_size, dim = 0)
        if c is not None:
            c_batch = torch.split(c, self.batch_size, dim = 0)
        else:
            c_batch = [None, ] * len(z_batch)
        res = []
        ## only use tqdm when batch size is larger than 10
        if len(z_batch) > 10:
            vessel = tqdm(enumerate(z_batch), desc="Generating", total = len(z_batch))
        else:
            vessel = enumerate(z_batch)
        for bid, zb in vessel:
            if isinstance(self.model, DDIM) or \
                hasattr(self.model, 'diff_model') and isinstance(self.model.diff_model, DDIM):
                y = self.model.sample(zb, c = c_batch[bid], clipped=self.clipped,
                                    clip_range=self.clip_range)
            elif isinstance(self.model, DDPM) or \
                hasattr(self.model, 'diff_model') and isinstance(self.model.diff_model, DDPM):
                y = self.model.sample(zb, end_step=self.num_sample_steps - 1, c = c_batch[bid],
                                            clipped=self.clipped, clip_range=self.clip_range)
            elif isinstance(self.model, EDM) or\
                hasattr(self.model, 'diff_model') and isinstance(self.model.diff_model, EDM):
                y = self.model.sample(zb, c = c_batch[bid])
            else:
                y = self.model.decode(zb, c_batch[bid])
        
            res.append(y)
        ### AE, VAE and so on.
        if type(res[0]) == tuple:
            res = tuple(torch.cat([res[j][i] for j in range(len(res))], dim = 0) for i in range(len(res[0])))
        else:
            res = torch.cat(res, dim = 0)
        return res
    
    ## interpolation, condition should be the same for inter data
    def interpolate(self, corner_latents = None, corner_num = 2, inter_num = 8, cond = None):
        if corner_latents is not None:
            corner_num = corner_latents.shape[0]
        if not (corner_num == 2 or corner_num == 4):
            logger.warning('Corner number should be 2 or 4.')
            corner_num = 2
            
        if corner_latents is None:
            latents = torch.randn(*( [corner_num, ] + self.model.get_latent_shape()), 
                                  generator=self.random_generator, device = self.device)
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
        
        x_bar = self.sample(inter_latent, cond)
        return x_bar

    def generate_rand(self, n = 16, cond = None):
        if not self.is_conditional:
            cond = None
        z = torch.randn(*( [n, ] + self.model.get_latent_shape()),
                        generator=self.random_generator, device = self.device)
        if cond is not None and (len(cond.shape) == 0 
                                 or not cond.shape[0] == n):
            ## all data use the same condition
            cond = torch.stack([cond for _ in range(n)])
        return self.sample(z, cond)
    
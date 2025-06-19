### ddpm for medical image segmentation
import torch
import torch.nn as nn
from flemme.model.ldm import create_diff_model
from flemme.logger import get_logger
logger = get_logger('model.ddpmseg')


class SupervisedDiffusion(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        diff_config = model_config.pop('diffusion', None)
        
        assert diff_config is not None, 'LDM needs a diffusion model to generate latents.'            
        self.diff_model, self.diff_model_name = create_diff_model(diff_config)
        if not self.diff_model.is_conditional:
            logger.error('Diffusion in Supervised DDPM need to be conditional')
            exit(1)
        self.is_conditional = False
        self.is_supervised = True
        self.is_generative = True
        self.num_ensemble = model_config.get('num_ensemble', 10)
    @property
    def device(self):
        return self.diff_model.device
    @torch.no_grad()
    def sample(self, xt, c, return_processing = False, **kwargs):
        assert c is not None, 'Sample for Supervised Diffusion Model needs a condition (input).'
        return self.diff_model.sample(xt, c = c,  
            return_processing = return_processing, **kwargs)
    def __str__(self):
        _str = "********************* SupervisedDiffusion *********************\n{}"\
            .format(self.diff_model.__str__())
        return _str
    @torch.no_grad()
    def forward(self, x, num_ensemble = None, **kwargs):
        if num_ensemble is None:
            num_ensemble = self.num_ensemble
        batch_size = x.shape[0]
        sampled_recons = []
        for _ in range(num_ensemble):
            yt = torch.randn((batch_size,) + tuple(self.get_latent_shape()), device=x.device)
            sampled_recons.append(
                self.sample(yt, c=x, **kwargs))
        ### return the mean map of samples
        return {'recon': sum(sampled_recons) / len(sampled_recons) }
    
    def compute_loss(self, x, y):
        return self.diff_model.compute_loss(y, c = x)
    def get_loss_name(self):
        return self.diff_model.get_loss_name()
    def get_input_shape(self):
        return self.diff_model.get_input_shape()
    def get_output_shape(self):
        return self.diff_model.get_output_shape()
    def get_latent_shape(self):
        return self.diff_model.get_latent_shape()
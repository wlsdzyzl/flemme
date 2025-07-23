import numpy as np
import torch
from flemme.model.distribution import GaussianDistribution as Gaussian
from flemme.model.ddpm import DiffusionProbabilistic as DDPM
from flemme.logger import get_logger
### Diffusion Implifict has a different sampling strategy with DDPM
### However, the training steps are the same
### therefore, the model trained for DDPM can be directly used on DDIM
### It can also use ddpm's model weights

logger = get_logger('model.ddim')
class DiffusionImplicit(DDPM):
    def __init__(self, model_config):
        super().__init__(model_config)

        ### compute sample steps
        num_sample_steps = model_config.get('num_sample_steps', 100)
        self.sample_discretize = model_config.get('sample_discretize', 'uniform')
        self.eta = model_config.get('eta', 0.0)
        if self.sample_discretize == 'uniform':
            c = self.num_steps // num_sample_steps
            self.sample_steps = np.asarray(list(range(0, self.num_steps, c))) + 1
         ## τ to be quadratically distributed across [1,2,…,T]
        elif self.sample_discretize == 'quad':
            self.sample_steps = ((np.linspace(0, np.sqrt(self.num_steps * .8), num_sample_steps)) ** 2).astype(int) + 1
        else:
            raise NotImplementedError(self.sample_discretize)
        self.compute_consts()


    def compute_consts(self):
        ## some constant values used for ddim
        ## should be called after model.to(device)
        ## not that the alpha in ddim is the alpha bar in ddpm
        self.ddim_alpha = self.alpha_bar[self.sample_steps].clone().float()
        self.ddim_alpha_sqrt = torch.sqrt(self.ddim_alpha)
        self.ddim_alpha_prev = torch.concat([self.alpha_bar[0:1], self.alpha_bar[self.sample_steps[:-1]]])
        self.ddim_sigma = (self.eta * ((1 - self.ddim_alpha_prev) / (1 - self.ddim_alpha) *
                                        (1 - self.ddim_alpha / self.ddim_alpha_prev)) ** .5)
    ### after load from checkpoint or transfer to another device
    ### we recompute this consts
    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.compute_consts()
    def to(self, device):
        super().to(device)
        self.compute_consts()
        return self

    ### not used for training
    ### add noise through index, note that index is not tensor
    # @torch.no_grad()
    # def add_noise_i(self, x0, index):
    #     ## do we need gather here?
    #     mean = self.ddim_alpha_sqrt[index] * x0 
    #     var = (1.0 -  self.ddim_alpha[index]) * x0
    #     return Gaussian(mean, var = var).sample()

    ## t: actural sample step
    ## c: condition
    ## index: sample step index
    @torch.no_grad()    
    def denoise(self, xt, t, index, c = None, clipped = None, clip_range = None):
        eps_theta = self.get_eps_from_model(xt, t, c)
        alpha = self.ddim_alpha[index]
        alpha_prev = self.ddim_alpha_prev[index]
        sigma = self.ddim_sigma[index]
        sqrt_one_minus_alpha = (1. - alpha) ** 0.5
        ## same as ddpm
        x0_pred = (xt - sqrt_one_minus_alpha * eps_theta) / alpha ** 0.5
        if clipped:
            x0_pred = x0_pred.clamp(*clip_range)
        dir_to_xt =  (1 - alpha_prev - sigma ** 2) ** 0.5 * eps_theta
        x_prev = alpha_prev ** 0.5 * x0_pred + dir_to_xt + sigma * torch.randn_like(xt)
        return x_prev
    

    @torch.no_grad()
    def sample(self, xt, c = None, clipped = None, 
               clip_range = None, return_processing = False):
        if clipped is None:
            clipped = self.clipped
        if clip_range is None:
            clip_range = self.clip_range
        batch_size = xt.shape[0]
        if return_processing:
            processing = [xt.clone()]
        for index in range(len(self.sample_steps) - 1, -1, -1):
            t = self.sample_steps[index]
            bt = torch.ones((batch_size,), device=xt.device, dtype=torch.long) * t
            xt = self.denoise(xt, bt, index, c = c, clipped = clipped, clip_range = clip_range)
            if return_processing:
                processing.append(xt.clone())
        if return_processing:
            return processing
        return xt
    
    # def forward(self, x, c = None, clipped = None, clip_range = None):
    #     if clipped is None:
    #         clipped = self.clipped
    #     if clip_range is None:
    #         clip_range = self.clip_range
    #     batch_size = x.shape[0]
    #     end_step = self.num_steps - 1

    #     t = torch.ones((batch_size,), device=x.device, dtype=torch.long) * end_step
    #     xt, _ = self.add_noise(x, t)
    #     return {'recon':self.sample(xt, c=c, clipped = clipped, clip_range = clip_range)}
### ddpm for medical image segmentation
from flemme.model.ddpm import DiffusionProbabilistic as DDPM
from flemme.model.ddim import DiffusionImplicit as DDIM
import torch
from flemme.logger import get_logger
logger = get_logger('model.ddpmseg')
class SupervisedDiffusionProbabilistic(DDPM):
    def __init__(self, model_config):
        super().__init__(model_config)
        if not self.is_conditional:
            logger.error('Supervised DDPM need to be conditional')
        self.is_conditional = False
        self.is_supervised = True
        self.num_ensemble = model_config.get('num_ensemble', 10)
        
    def forward(self, x, end_step = -1, clipped = None, clip_range = None, num_ensemble = None):
        if clipped is None:
            clipped = self.clipped
        if clip_range is None:
            clip_range = self.clip_range
        if num_ensemble is None:
            num_ensemble = self.num_ensemble
        batch_size = x.shape[0]
        # Get random $t$ for each sample in the batch
        assert end_step < self.num_steps, "End step is larger than or equal to the amount of sample steps."
        if end_step < 0:
            end_step = self.num_steps - 1
        sampled_recons = []
        for _ in range(num_ensemble):
            yt = torch.randn((batch_size,) + tuple(self.get_latent_shape()), device=x.device)
            sampled_recons.append(
                self.sample(yt, end_step = end_step, c=x, clipped = clipped, clip_range = clip_range))
        ### return the mean map of samples
        return {'recon': sum(sampled_recons) / len(sampled_recons) }
    
    def compute_loss(self, x: torch.Tensor, y = None):
        return super().compute_loss(x0 = y, c = x)
    
class SupervisedDiffusionImplicit(DDIM):
    def __init__(self, model_config):
        super().__init__(model_config)
        if not self.is_conditional:
            logger.error('Supervised DDIM need to be conditional')
        self.is_conditional = False
        self.is_supervised = True
        self.num_ensemble = model_config.get('num_ensemble', 10)

    def forward(self, x, clipped = None, clip_range = None, num_ensemble = None):
        if clipped is None:
            clipped = self.clipped
        if clip_range is None:
            clip_range = self.clip_range
        if num_ensemble is None:
            num_ensemble = self.num_ensemble
        batch_size = x.shape[0]

        sampled_recons = []
        for _ in range(num_ensemble):
            yt = torch.randn((batch_size,) + tuple(self.get_latent_shape()), device=x.device)
            sampled_recons.append(
                self.sample(yt, c=x, clipped = clipped, clip_range = clip_range))
        ### return the mean map of samples
        return {'recon': sum(sampled_recons) / len(sampled_recons) }
    
    def compute_loss(self, x: torch.Tensor, y = None):
        return super().compute_loss(x0 = y, c = x)
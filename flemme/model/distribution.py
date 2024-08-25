import torch
import torch.nn.functional as F
from flemme.logger import get_logger
logger = get_logger('model.gaussian')
class GaussianDistribution:
    def __init__(self, mean, var = None, logvar = None):
        self.mean = mean
        self.device = mean.device
        if logvar is not None:
            ### clamp logvar
            self.logvar = torch.clamp(logvar, -30.0, 20.0)
        elif var is not None:
            self.logvar = torch.clamp(torch.log(var), -30.0, 20.0)
        else:
            logger.error('At least one of var or logvar need to be specified.')
            exit(1)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
    def sample(self, return_noise = False):
        noise = torch.randn_like(self.mean)
        if return_noise:
            return self.mean + self.std * noise, noise
        return self.mean + self.std * noise
class GumbelSoftmaxDistribution:
    def __init__(self, logits, tau = 1, dim = -1):
        self.logits = logits
        self.tau = tau
        self.device = self.logits.device
    def sample(self, return_prob = False):
        if return_prob:
            return F.gumbel_softmax(self.logits, self.tau, hard = True), F.gumbel_softmax(self.logits, self.tau, hard = False)
        return F.gumbel_softmax(self.logits, self.tau, hard = True)
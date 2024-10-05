# file to define different losses
from flemme.config import module_config
import torch
import torch.nn as nn
from flemme.logger import get_logger
logger = get_logger('loss')
class TorchLoss(nn.Module):
    def __init__(self, torch_loss = nn.MSELoss, reduction = 'mean', channel_dim = 1):
        super().__init__()
        self.reduction = reduction
        self.torch_loss = torch_loss(reduction = 'none')
        self.channel_dim = channel_dim
    def forward(self, x, y):
        #### channel mean, only matters for CrossEntropyLoss
        if self.channel_dim != 1:
            x, y = x.transpose(1, self.channel_dim), y.transpose(1, self.channel_dim)
        res = self.torch_loss(x, y)
        #### spatial mean
        if res.ndim > 1:
            res = res.mean(dim = tuple(range(1, res.ndim)))
        if self.reduction == 'none':    
            return res
        if self.reduction == 'sum':
            return res.sum()
        if self.reduction == 'mean':
            return res.mean()


def kl_distance(gauss1, gauss2 = None):
    if gauss2 is None:
        return - 0.5 *(1+ gauss1.logvar - gauss1.mean.pow(2) - gauss1.var)
    return  0.5 * (gauss2.logvar - gauss1.logvar + (gauss1.var + 
            (gauss1.mean - gauss2.mean).pow(2)) / gauss2.var - 1)
## KL Divergence
class KLLoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, gauss1, gauss2=None):
        res = kl_distance(gauss1, gauss2)
        ### compute mean over channel and spatial
        if res.ndim > 1:
            res = res.mean(dim = tuple(range(1, res.ndim)))

        if self.reduction == 'sum':
            return torch.sum(res)
        elif self.reduction == "mean":
            return torch.mean(res)
        else:
            return res
        
## Treat distribution as vector, naive loss
class DistriMSELoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, gauss1, gauss2=None):
        res = ((gauss1.logvar - gauss2.logvar) ** 2 + (gauss1.mean - gauss2.mean) ** 2) / 2
        ### compute mean over channel and spatial
        if res.ndim > 1:
            res = res.mean(dim = tuple(range(1, res.ndim)))
        if self.reduction == 'sum':
            return torch.sum(res)
        elif self.reduction == "mean":
            return torch.mean(res)
        else:
            return res

### medical image segmentation
class DiceLoss(nn.Module):
    def __init__(self, reduction = 'mean', normalization = 'sigmoid', channel_dim = 1):
        super().__init__()
        self.reduction = reduction
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=channel_dim)
        else:
            self.normalization = nn.Identity()
        self.channel_dim = channel_dim
    def forward(self, x, y):
        x = self.normalization(x)
        if self.channel_dim != 1:
            x, y = x.transpose(1, self.channel_dim), y.transpose(1, self.channel_dim)
        B, C = x.shape[0], x.shape[1]
        x = x.reshape((B, C, -1))
        y = y.reshape((B, C, -1))
        intersect = (x * y).sum(-1)
        denominator = torch.clamp((x*x).sum(-1) + (y*y).sum(-1), min=1e-8)
        numerator = torch.clamp(2 * intersect, min = 1e-8)
        dice_score = numerator / denominator  
        ### per channel dice loss
        loss = (1 - dice_score).mean(dim = 1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

if module_config['point-cloud']:    
    from .chamfer_distance import ChamferDistance
    from .emd import emdModule as EMD
    ### The following loss can be used as reconstruction loss for point clouds
    ## Chamfer distance
    class ChamferLoss(nn.Module):
        def __init__(self, reduction = 'mean'):
            super().__init__()
            self.reduction = reduction
            self.chamfer = ChamferDistance()
        def forward(self, x, y):
            d1, d2 = self.chamfer(x, y)
            d1, d2 = d1.mean(dim=-1), d2.mean(dim=-1)
            if self.reduction == 'sum':
                return 0.5 * (torch.sum(d1) + torch.sum(d2))
            elif self.reduction == "mean":
                return 0.5 * (torch.mean(d1) + torch.mean(d2))
            else:
                return d1, d2
    ## Wasserstein distance based on optimal transport
    ## a small value of eps leads to bad reconstruction
    class EMDLoss(nn.Module):
        def __init__(self, reduction = 'mean', eps = 0.005, iters = 50):
            super().__init__()
            self.loss = EMD(eps = eps, iters = iters)
            self.reduction = reduction
        def forward(self, x, y):
            loss = torch.sqrt(self.loss(x, y)[0])
            loss = loss.mean(dim=-1)
            if self.reduction == 'sum':
                return loss.sum()
            elif self.reduction == 'mean':
                return loss.mean()
            else:
                return loss

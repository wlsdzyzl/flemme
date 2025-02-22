# file to define different losses
from flemme.config import module_config
import torch
import torch.nn as nn
from flemme.logger import get_logger
from geomloss import SamplesLoss
from .ssim import create_window, create_window_3D, _ssim, _ssim_3D

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
        #### spatial and channel mean
        if res.ndim > 1:
            res = res.mean(dim = tuple(range(1, res.ndim)))
        if self.reduction == 'none':    
            return res
        if self.reduction == 'sum':
            return res.sum()
        if self.reduction == 'mean':
            return res.mean()

class SSIMLoss(torch.nn.Module):
    def __init__(self, image_dim = 2, image_channel = 1, window_size = 11, reduction = 'mean', sigma = 1.5, **kwargs):
        super().__init__()
        self.channel = image_channel
        self.dim = image_dim
        self.window = create_window(window_size, self.channel, sigma = sigma)
        if self.dim == 3:
            self.window = create_window_3D(window_size, self.channel, sigma = sigma)
        self.reduction = reduction
    def forward(self, x, y):
        assert x.shape[1] == self.channel and y.shape[1] == self.channel, \
                'Channel size mis-matched for SSIM loss.'
        assert (x.ndim - 2) == self.dim, 'Image dimension mis-matched for SSIM loss'
        window = self.window.to(x.device)
        if self.dim == 2:
            res = 1.0 - _ssim(x, y, window)
        else:
            res = 1.0 - _ssim_3D(x, y, window)
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
    from .ext_modules import ChamferDistance, emdModule as EMD 
    ### The following loss can be used as reconstruction loss for point clouds
    ## Chamfer distance
    class ChamferLoss(nn.Module):
        def __init__(self, reduction = 'mean', extended = False):
            super().__init__()
            self.reduction = reduction
            self.chamfer = ChamferDistance()
            self.extended = extended
        def calc_cd(self, x, gt):
            return self.chamfer(x, gt)
        def forward(self, x, y):
            d1, d2 = self.calc_cd(x, y)
            d1, d2 = d1.mean(dim=-1, keepdim=True), d2.mean(dim=-1, keepdim=True)
            if self.extended:
                ## for each pair of point cloud,  we compute the maximum distance
                dist, _ = torch.max(torch.cat([d1, d2], dim = -1), dim = -1)
            else:
                dist = ((d1 + d2) * 0.5).squeeze(-1)
            if self.reduction == 'sum':
                return dist.sum()
            elif self.reduction == "mean":
                return dist.mean()
            else:
                return dist
    ## from `Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion`, 
    ## https://arxiv.org/pdf/2111.12702
    class DensityAwareChamferLoss(ChamferLoss):
        def __init__(self, reduction = 'mean', extended = False, 
                alpha=1000, n_lambda=1, non_reg=False):
            super().__init__(reduction, extended)
            self.reduction = reduction
            self.chamfer = ChamferDistance()
            self.alpha = alpha
            self.n_lambda = n_lambda
            self.non_reg = non_reg
        def calc_cd(self, x, gt):
            x = x.float()
            gt = gt.float()
            batch_size, n_x, _ = x.shape
            batch_size, n_gt, _ = gt.shape
            assert x.shape[0] == gt.shape[0]

            if self.non_reg:
                frac_12 = max(1, n_x / n_gt)
                frac_21 = max(1, n_gt / n_x)
            else:
                frac_12 = n_x / n_gt
                frac_21 = n_gt / n_x

            dist1, dist2, idx1, idx2 = self.chamfer(x, gt, return_idx=True)
            # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
            # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
            # dist2 and idx2: vice versa
            exp_dist1, exp_dist2 = torch.exp(-dist1 * self.alpha), torch.exp(-dist2 * self.alpha)

            count1 = torch.zeros_like(idx2)
            count1.scatter_add_(1, idx1.long(), torch.ones_like(idx1))
            weight1 = count1.gather(1, idx1.long()).float().detach() ** self.n_lambda
            weight1 = (weight1 + 1e-6) ** (-1) * frac_21
            loss1 = (1 - exp_dist1 * weight1)

            count2 = torch.zeros_like(idx1)
            count2.scatter_add_(1, idx2.long(), torch.ones_like(idx2))
            weight2 = count2.gather(1, idx2.long()).float().detach() ** self.n_lambda
            weight2 = (weight2 + 1e-6) ** (-1) * frac_12
            loss2 = (1 - exp_dist2 * weight2)
            return loss1, loss2
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
    class SinkhornLoss(nn.Module):
        def __init__(self, reduction = 'mean', blur=0.01, scaling=0.9):
            super().__init__()
            self.loss = SamplesLoss("sinkhorn", blur = blur, scaling = scaling)
            self.reduction = reduction
        def forward(self, x, y):
            loss = self.loss(x, y)
            if self.reduction == 'sum':
                return loss.sum()
            elif self.reduction == 'mean':
                return loss.mean()
            else:
                return loss

if module_config['graph']:    
    ### loss for graph reconstruction
    ## mse
    class GraphNodeLoss(nn.Module):
        def __init__(self, reduction = 'mean', 
                    lambda_pos = 1.0, lambda_feature = 1.0):
            super().__init__()
            self.pos_loss = TorchLoss(torch_loss = nn.MSELoss, reduction = reduction)
            self.feature_loss = TorchLoss(torch_loss = nn.MSELoss, reduction = reduction)
        def forward(self, pred, data):
            assert type(pred) == tuple, 'the prediction of graph should be a tuple.'
            batch_size = data.batch_size
            recon_pos, recon_feature, _ = pred
            gt_pos, gt_feature, batch = data.pos, data.x, data.batch
            
            loss = 0
            if recon_pos is not None and gt_pos is not None:
                # BN * F -> B * N * F
                recon_pos = torch.stack(torch.chunk(recon_pos, batch_size, dim = 0), dim = 0)
                gt_pos = torch.stack(torch.chunk(gt_pos, batch_size, dim=0), dim = 0)
                loss = lambda_pos * self.pos_loss(recon_pos, gt_pos)
                
            if recon_feature is not None and gt_feature is not None:
                recon_feature = torch.stack(torch.chunk(recon_feature, batch_size, dim = 0), dim = 0)
                gt_feature = torch.stack(torch.chunk(gt_feature, batch_size, dim=0), dim = 0)
                lf = lambda_feature * self.feature_loss(recon_feature, gt_feature)
                loss = lf + loss if loss is not None else lf
            return loss
    ### BCE loss
    class GraphEdgeLoss(nn.Module):
        def __init__(self, reduction = 'mean'):
            super().__init__()
        def forward(self, pred, data):
            assert type(pred) == tuple, 'the prediction of graph should be a tuple.'
            batch_size = data.batch_size
            recon_edge = pred[-1]
            gt_edge_index = data.edge_index

            if recon_edge is not None and edge_index is not None:
                pp_recon_edge = torch.sparse_coo_tensor(indices = recon_edge.indices(), 
                        values = torch.log(torch.sigmoid(recon_edge.values())), size = recon_edge.shape).coalesce()
                np_recon_edge = torch.sparse_coo_tensor(indices = recon_edge.indices(), 
                        values = torch.log(1 - torch.sigmoid(recon_edge.values())), size = recon_edge.shape).coalesce()
                gt_edge = torch.sparse_coo_tensor(indices = gt_edge_index, 
                        values = torch.ones(gt_edge_index.shape[1], dtype=torch.float32, 
                        device = recon_edge.device), size = recon_edge.shape).coalesce()
                one_minus_gt_edge = torch.sparse_coo_tensor(indices = recon_edge.indices(), 
                        values = torch.ones(recon_edge.indices.shape[1], dtype=torch.float32, 
                        device = recon_edge.device), size = recon_edge.shape).coalesce() - gt_edge
                # eps = torch.sparse_coo_tensor(indices = recon_edge.indices(), 
                #         values = torch.ones(gt_edge_index.shape[1], dtype=torch.float32, 
                #         device = recon_edge.device) * 1e-8, size = recon_edge.shape).coalesce()
                le = -(pp_recon_edge * gt_edge + np_recon_edge * one_minus_gt_edge) 
                le = torch.stack(torch.chunk(le.values(), batch_size, dim = 0), dim = 0)
                le = le.mean(dim = -1)
                if self.reduction == 'mean':
                    le = le.mean()
                elif self.reduction == 'sum':
                    le = le.sum()
                return le
            else:
                logger.warning('There is no edge in graph or model doesn\'t predict edges.')
                return torch.Tensor([0.0])
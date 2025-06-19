from .loss import *
import torch.nn as nn
from flemme.utils import DataForm
logger = get_logger('loss.get_loss')
def get_loss(loss_config, data_form = DataForm.IMG):
    channel_dim = -1 if data_form == DataForm.PCD else 1
    if loss_config is None:
        logger.warning('No configuration for loss function. Using MSE loss ...')
        return TorchLoss(torch_loss = nn.MSELoss)
    loss_name = loss_config.pop('name', None)
    if loss_name is None:
        logger.warning('Name of loss function is not specified. Using MSE loss ...')
        return TorchLoss(torch_loss = nn.MSELoss)
    if loss_name == 'MSE':
        return TorchLoss(torch_loss = nn.MSELoss, **loss_config)
    if loss_name == 'L1':
        return TorchLoss(torch_loss = nn.L1Loss, **loss_config)
    if loss_name == 'SSIM':
        return SSIMLoss(**loss_config)
    if loss_name == 'BCE' or loss_name == 'BCEL':
        return TorchLoss(torch_loss = nn.BCEWithLogitsLoss, **loss_config)
    if loss_name == 'CE':
        return TorchLoss(torch_loss = nn.CrossEntropyLoss, channel_dim=channel_dim, **loss_config)
    if loss_name == 'Dice':
        return DiceLoss(channel_dim=channel_dim, **loss_config)
    ## boundary loss hasn't been tested.
    if loss_name == 'Boundary':
        return SurfaceLoss(**loss_config)
    if loss_name == "KL":
        return KLLoss(**loss_config)
    if loss_name == 'DMSE':
        return DistriMSELoss(**loss_config)
    if data_form == DataForm.IMG:
        logger.warning('Using unsupported loss for images.')
    if loss_name == 'EMD':
        return EMDLoss(**loss_config)
    if loss_name == 'Chamfer' or loss_name == 'CD':
        return ChamferLoss(**loss_config)
    if loss_name == 'DChamfer' or loss_name == 'DCD':
        return DensityAwareChamferLoss(**loss_config)
    if loss_name == 'Sinkhorn':
        return SinkhornLoss(**loss_config)
    ## graph node error
    if loss_name == 'GraphNodeLoss' or loss_name == 'GNE':
        return GraphNodeLoss(**loss_config)
    ## graph edge error
    if loss_name == 'GraphEdgeLoss' or loss_name == 'GEE':
        return GraphEdgeLoss(**loss_config)
    else:
        raise NotImplementedError
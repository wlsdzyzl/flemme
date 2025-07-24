import glob
import os
import torch
from flemme.utils import load_ply, save_ply
from flemme.augment.pcd_transforms import FixedPoints, ToTensor
from flemme.loss import ChamferLoss, DensityAwareChamferLoss as DCDLoss, EMDLoss, SinkhornLoss
from torch import optim
from flemme.logger import get_logger
import time
logger = get_logger('unittest::test_pcd_loss')

device = "cuda" if torch.cuda.is_available() else "cpu"

to_tensor = ToTensor(dtype = 'float')
fixed_points = FixedPoints(4096)

pcd_paths = sorted(glob.glob(os.path.join("./artery/", "*.ply")))
pcds = [ to_tensor( fixed_points(load_ply(p_path))) for p_path in pcd_paths]
### stack will create a new axis
pcds = torch.stack(pcds, dim = 0).to(device)
logger.info('input_size: ',pcds.shape)
losses = [ChamferLoss(), DCDLoss(), EMDLoss(), SinkhornLoss()]
loss_names = ['chamfer', 'DCD', 'EMD', 'sinkhorn']
init = torch.randn((pcds.shape[0], 4096, 3), device = device)
for loss, loss_name in zip(losses, loss_names):
    X = [init.clone().detach().requires_grad_(),]
    lr = 0.005
    ### training process
    optimizer = optim.Adam(X, lr)
    max_iter = 10000
    start_time = time.perf_counter()
    for iter_id in range(max_iter):
        l = loss(X[0], pcds)
        ## DCD loss is more sensitive to the learning rate
        ## use a smaller learning rate for DCD loss is preferred
        ## A combination of DCD and chamfer loss helps to reach the convergence faster
        if loss_name == 'DCD':
            l += losses[0](X[0], pcds)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        if (iter_id+1)%100 == 0 or iter_id==0:
            ## save pcd
            logger.info('iter{:03d}, {}: {}'.format(iter_id, loss_name, l.item()))
            X_np = X[0].detach().cpu().numpy()
            for _x, p_path in zip(X_np, pcd_paths):
                save_ply('./recon/{}_{}'.format(loss_name, os.path.basename(p_path)), _x)
    logger.info("Runtime: %lfs" % (time.perf_counter() - start_time))
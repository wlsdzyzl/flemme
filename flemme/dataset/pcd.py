import numpy as np

import torch
from torch.utils.data import Dataset
import os
import glob
from flemme.utils import load_pcd, save_pcd, get_random_state, set_random_state
from flemme.logger import get_logger
### Pure point cloud dataset
## there can be other types of datasets, such as point cloud segmentation or point cloud classification
## other types of datasets can be used for conditional generation
logger = get_logger('pcd_dataset')
shapenet_codemap = {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}

class PCDDataset(Dataset):
    def __init__(self, data_path, data_transform = None, mode = 'train', data_dir = '', 
                 data_suffix = '.ply', **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.pcd_path_list = sorted(glob.glob(os.path.join(data_path + '/' + data_dir,  "*" + data_suffix)))
        self.data_path = data_path
        self.mode = mode
        self.data_transform = data_transform
    def __len__(self):
        return len(self.pcd_path_list)

    def __getitem__(self, index):
        """Get the pcds"""
        pcd_path = self.pcd_path_list[index]
        pcd = load_pcd(pcd_path)
        if self.data_transform:
            pcd = self.data_transform(pcd)
            # save_pcd(pcd_path+'.transformed.ply', pcd.numpy())
        return pcd, 0, pcd_path
### pcd segmentation dataset
class PCDSegDataset(PCDDataset):
    def __init__(self, data_path, data_transform = None, label_transform = None, mode = 'train', data_dir = 'pcd', 
                 label_dir = 'label', data_suffix = '.ply', label_suffix='.seg', **kwargs):
        super().__init__(data_path = data_path, data_transform = data_transform, mode = mode,
            data_dir = data_dir, data_suffix = data_suffix)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.label_path_list = [ppath.replace(data_suffix, label_suffix).replace(data_dir, label_dir) for ppath in self.pcd_path_list]
        self.label_transform = label_transform
        
    def __getitem__(self, index):
        """Get the pcds"""
        pcd = load_pcd(self.pcd_path_list[index])
        label = np.loadtxt(self.label_path_list[index])
        # print(label.max(), label.min())
        if self.data_transform:
            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)
            set_random_state(n_state, t_state)
            label = self.label_transform(label)
            # save_pcd(pcd_path+'.transformed.ply', pcd.numpy())
        return pcd, label, self.pcd_path_list[index]

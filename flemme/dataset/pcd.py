import numpy as np

import torch
from torch.utils.data import Dataset
import os
import glob
from flemme.utils import load_pcd, get_random_state, set_random_state, rreplace
from flemme.logger import get_logger
### Pure point cloud dataset
## there can be other types of datasets, such as point cloud segmentation or point cloud classification
## other types of datasets can be used for conditional generation
logger = get_logger('pcd_dataset')

class PcdDataset(Dataset):
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
        return pcd, pcd_path
        
class PcdClsDataset(Dataset):
    def __init__(self, data_path, 
                data_transform = None, 
                label_transform = None,
                mode = 'train', 
                pre_shuffle = True,
                data_suffix = '.ply',
                cls_label = {},
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.pcd_path_list = []
        self.labels = []
        class_dirs = list(cls_label.keys())
        for cls_dir in class_dirs:
            sub_path_list = sorted(glob.glob(os.path.join(data_path + '/' + cls_dir,  "*" + data_suffix)))
            self.pcd_path_list = self.pcd_path_list + sub_path_list
            assert cls_dir in cls_label, f'Unknowk class: {cls_dir}'
            self.labels = self.labels + [cls_label[cls_dir], ] * len(sub_path_list)
        if pre_shuffle:
            shuffled_index = np.arange(len(self.pcd_path_list))
            np.random.shuffle(shuffled_index)
            self.pcd_path_list = [self.pcd_path_list[i] for i in shuffled_index]
            self.labels = [self.labels[i] for i in shuffled_index]
    def __len__(self):
        return len(self.pcd_path_list)
    def __getitem__(self, index):
        """Get the pcds"""
        pcd = load_pcd(self.pcd_path_list[index])
        label = self.labels[index]
        if self.data_transform:
            pcd = self.data_transform(pcd)
        if self.label_transform is not None:
            label = self.label_transform(label)
        return pcd, label, self.pcd_path_list[index]

### pcd segmentation dataset
class PcdSegDataset(PcdDataset):
    def __init__(self, data_path, data_transform = None, label_transform = None, mode = 'train', data_dir = 'pcd', 
                 label_dir = 'label', data_suffix = '.ply', label_suffix='.seg', **kwargs):
        super().__init__(data_path = data_path, data_transform = data_transform, mode = mode,
            data_dir = data_dir, data_suffix = data_suffix)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.label_path_list = [rreplace(rreplace(ppath, data_suffix, label_suffix, 1), data_dir, label_dir, 1) for ppath in self.pcd_path_list]
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


class PcdReconDataset(PcdDataset):
    def __init__(self, data_path, data_transform = None, target_transform = None, mode = 'train', data_dir = 'pcd', 
                 target_dir = 'target', data_suffix = '.ply', target_suffix='.ply', **kwargs):
        super().__init__(data_path = data_path, data_transform = data_transform, mode = mode,
            data_dir = data_dir, data_suffix = data_suffix)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.target_path_list = [rreplace(rreplace(ppath, data_suffix, target_suffix, 1), data_dir, target_dir, 1) for ppath in self.pcd_path_list]
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """Get the pcds"""
        pcd = load_pcd(self.pcd_path_list[index])
        target = load_pcd(self.target_path_list[index])
        # print(target.max(), target.min())
        if self.data_transform:
            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)
            set_random_state(n_state, t_state)
            target = self.target_transform(target)
            # save_pcd(pcd_path+'.transformed.ply', pcd.numpy())
        return pcd, target, self.pcd_path_list[index]
    
class PcdReconWithClassLabelDataset(Dataset):
    def __init__(self, data_path, 
                 data_transform = None, 
                 label_transform = None, 
                 target_transform = None,
                 mode = 'train', 
                 data_dir = 'partial', 
                 target_dir = 'label', 
                 data_suffix = '.ply', 
                 target_suffix='.ply', 
                 cls_label = {},
                 pre_shuffle = True,
                 **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.target_transform = target_transform
        self.pcd_path_list = []
        self.target_path_list = []
        self.labels = []

        class_dirs = list(cls_label.keys())

        for cls_dir in class_dirs:
            sub_path_list = sorted(glob.glob(os.path.join(data_path, cls_dir, data_dir,  "*" + data_suffix)))
            self.pcd_path_list = self.pcd_path_list + sub_path_list
            sub_target_path_list = [rreplace(rreplace(s, data_dir, target_dir, 1), data_suffix, target_suffix, 1) for s in sub_path_list]
            self.target_path_list = self.target_path_list + sub_target_path_list
            assert cls_dir in cls_label, f'Unknowk class: {cls_dir}'
            self.labels = self.labels + [cls_label[cls_dir], ] * len(sub_path_list)
        if pre_shuffle:
            shuffled_index = np.arange(len(self.pcd_path_list))
            np.random.shuffle(shuffled_index)
            self.pcd_path_list = [self.pcd_path_list[i] for i in shuffled_index]
            self.target_path_list = [self.target_path_list[i] for i in shuffled_index]
            self.labels = [self.labels[i] for i in shuffled_index]

    def __len__(self):
        return len(self.pcd_path_list)

    def __getitem__(self, index):
        """Get the pcds"""
        pcd = load_pcd(self.pcd_path_list[index])
        target = load_pcd(self.target_path_list[index])
        label = self.labels[index]
        # print(label.max(), label.min())
        if self.data_transform:
            n_state, t_state = get_random_state()
            pcd = self.data_transform(pcd)
            set_random_state(n_state, t_state)
            target = self.target_transform(target)
            if self.label_transform:
                label = self.label_transform(label)
            # save_pcd(pcd_path+'.transformed.ply', pcd.numpy())
        return pcd, target, label, self.pcd_path_list[index]


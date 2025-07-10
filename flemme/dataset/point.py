import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from flemme.utils import load_pcd, normalize
from flemme.logger import get_logger
from sklearn.datasets import make_swiss_roll
# vector dataset
logger = get_logger('vector_dataset')
## Point Dataset
class PointDataset(Dataset):
    def __init__(self, data_path, transform = None, mode = 'train',  **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.vec = load_pcd(self.vec_path)
        self.mode = mode
        self.transform = transform
        # self.vec = normalize(self.vec)
    def __len__(self):
        return len(self.vec)
    ### the dataset will not be stored in the memory
    def __getitem__(self, index):
        vec = self.vec[index]
        if self.transform:
            vec = self.transform(vec)
        return vec, ''
### generate by algorithm
class ToyDataset(Dataset):
    def __init__(self, n_sample, noise = 0.0, dim = 3, transform = None, mode = 'train', **kwargs):
        super().__init__()
        self.n_sample = n_sample
        self.points, self.t = make_swiss_roll(n_samples=n_sample, noise=noise)
        if dim == 2:
            self.points =self.points[:, [0, 2]]
        self.mode = mode
        self.transform = transform
        self.points = normalize(self.points)

    def __len__(self):
        return self.n_sample
    ### the dataset will not be stored in the memory
    def __getitem__(self, index):
        p = self.points[index]
        if self.transform:
            p = self.transform(p)
        return p, self.t[index], ''
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from torchvision.datasets import MNIST, CIFAR10, CelebA
from flemme.utils import load_img, load_itk, set_random_state, get_random_state, crop_boundingbox
from flemme.logger import get_logger
from functools import partial
# image segmentation dataloader
# there might be other type of image dataset, such as image classification dataset or pure image generation dataset.
logger = get_logger('image_dataset')
class ImgSegDataset(Dataset):
    def __init__(self, data_path, dim = 2, data_transform = None, mode = 'train', 
                 data_dir = 'raw', label_dir = 'label', data_suffix='.png', 
                 label_suffix = None, label_transform = None, crop_nonzero = None, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        label_suffix = label_suffix or data_suffix
        logger.info("loading data from the directory: {}".format(data_path))
        self.img_path_list = sorted(glob(os.path.join(data_path+'/' + data_dir, "*" + data_suffix)))
        self.mask_path_list = [ipath.replace(data_suffix, label_suffix).replace(data_dir, label_dir) for ipath in self.img_path_list]
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.dim = dim
        self.crop_by = None
        if crop_nonzero is not None:
            self.crop_nonzero = partial(crop_boundingbox, margin = crop_nonzero.get('margin', (0,0,0)), background=0)
            self.crop_by = crop_nonzero.get('crop_by', 'raw')
    def __len__(self):
        return len(self.img_path_list)
    ### the dataset will not be stored in the memory
    def __getitem__(self, index):
        """Get the images"""
        img_path = self.img_path_list[index]
        
        mask_path = self.mask_path_list[index]
        if self.dim == 2:
            img = load_img(img_path)
            mask = load_img(mask_path)
        else:
            img = load_itk(img_path)[0]
            mask = load_itk(mask_path)[0]
            ### currently, crop_by only support 3D images
            if self.crop_by == 'raw':
                img, mask, _ = self.crop_nonzero(data = img, follows = mask)
            elif self.crop_by == 'mask':
                mask, img, _ = self.crop_nonzero(data = mask, follows = img)
        if self.data_transform is not None:
            ## why do we need to set the state here?
            # the purpose is to make sure that two transforms are applied on the same state. 
            n_state, t_state = get_random_state()
            img = self.data_transform(img)
            ### to check if the mask and image has the same random transforms
            # x = np.random.randn(10)
            # x_torch = torch.randn(10)
            set_random_state(n_state, t_state)
            mask = self.label_transform(mask)
            # y = np.random.randn(10)
            # y_torch = torch.randn(10)
            # print((x == y).all(), torch.all(torch.eq(x_torch, y_torch)))
        # print(mask.max(), mask.shape)
        return img, mask, img_path

class MultiModalityImgSegDataset(Dataset):
    def __init__(self, data_path, dim = 2, data_transform = None, mode = 'train', 
                 data_dir = 'raw', label_dir = 'label', data_suffix='.png', 
                 label_suffix = None, label_transform = None, crop_nonzero = None, combine = 'mean', **kwargs):
        super().__init__()
        
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.dim = dim

        if self.dim == 2:
            raise NotImplementedError
        if not type(data_dir) == list:
            data_dir = [data_dir]
        if not type(label_dir) == list:
            label_dir = [label_dir]

        if type(data_suffix) == list:
            assert len(data_suffix) == len(data_dir), "length of suffix should be equal to data_channel dir"
        else:
            label_suffix = label_suffix or data_suffix
            data_suffix = [data_suffix] * len(data_dir)
        if type(label_suffix) == list:
            assert len(label_suffix) == len(label_dir), "length of label_suffix should be equal to label_channel dir"
        else:
            label_suffix = [label_suffix] * len(label_dir)

        self.img_path_list = []
        self.mask_path_list = []
        for rd, sf in zip(data_dir, data_suffix):
            if len(self.img_path_list) == 0:
                self.img_path_list.append(sorted(glob(os.path.join(data_path+'/' + rd, "*" + sf))))
            else:
                self.img_path_list.append([rpath.replace(data_dir[0], rd).replace(data_suffix[0], sf) for rpath in self.img_path_list[0]])
                
        for ld, lsf in zip(label_dir, label_suffix):
            self.mask_path_list.append([rpath.replace(data_dir[0], ld).replace(data_suffix[0], lsf) for rpath in self.img_path_list[0]])

        logger.info("loading data from the directory: {}".format(data_path))
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.crop_by = None
        if crop_nonzero is not None:
            self.crop_nonzero = partial(crop_boundingbox, margin = crop_nonzero.get('margin', (0,0,0)), background=0)
            self.crop_by = crop_nonzero.get('crop_by', 'raw')
        if combine == 'sum':
            self.combine = sum
        elif combine == 'mean':
            self.combine = lambda x: sum(x) / len(x)
        elif combine == 'cat':
            self.combine = np.stack
        else:
            logger.error('Unknown way to combine different modality datas.')
            exit(1)
    def __len__(self):
        return len(self.img_path_list[0])
    ### the dataset will not be stored in the memory
    def __getitem__(self, index):
        """Get the images"""
        
        img_paths = [ l[index] for l in self.img_path_list]
        mask_paths = [ l[index] for l in self.mask_path_list]
        if self.dim == 3:
            imgs = [load_itk(img_path)[0] for img_path in img_paths]
            masks = [load_itk(mask_path)[0] for mask_path in mask_paths]
            if len(imgs) == 1:
                img = imgs[0]
            else:
                img = self.combine(imgs)
            
            if len(masks) == 1:
                mask = masks[0]
            else:
                mask = self.combine(masks)
            ### currently, crop_by only support 3D images
            if self.crop_by == 'raw':
                img, mask, _ = self.crop_nonzero(data = img, follows = mask)
            elif self.crop_by == 'mask':
                mask, img, _ = self.crop_nonzero(data = mask, follows = img)
            if self.data_transform:
                ## why do we need to set the state here?
                # the purpose is to make sure that two transforms are applied on the same state. 
                n_state, t_state = get_random_state()
                img = self.data_transform(img)
                ### to check if the mask and image has the same random transforms
                # x = np.random.randn(10)
                # x_torch = torch.randn(10)
                set_random_state(n_state, t_state)
                mask = self.label_transform(mask)
                # y = np.random.randn(10)
                # y_torch = torch.randn(10)
                # print((x == y).all(), torch.all(torch.eq(x_torch, y_torch)))
                return img, mask, img_paths

class MNISTWrapper(MNIST):
    def __init__(self, data_path, data_transform = None, mode = 'train', **kwargs):
        super().__init__(root = data_path, train=(mode == 'train'), transform = data_transform, download=True)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.mode = mode

    def __getitem__(self, index):
        """Get the images"""
        img, label = super().__getitem__(index)
        return img, label, ''

class CIFAR10Wrapper(CIFAR10):
    def __init__(self, data_path, data_transform = None, mode = 'train', **kwargs):
        super().__init__(root = data_path, train=(mode == 'train'), transform = data_transform, download=True)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.mode = mode

    def __getitem__(self, index):
        """Get the images"""
        img, label = super().__getitem__(index)
        return img, label, ''
class CelebAWrapper(CelebA):
    def __init__(self, data_path, data_transform = None, mode = 'train', **kwargs):
        super().__init__(root = data_path, split=mode, transform = data_transform, download=True)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.mode = mode
    def __getitem__(self, index):
        """Get the images"""
        img, label = super().__getitem__(index)
        return img, label, ''
    
### transfer volume to patches for training with less memory
# class PatchSingleImgSegDataset(Dataset):
#     pass

# class PatchImgSegDataset(Dataset):
#     pass
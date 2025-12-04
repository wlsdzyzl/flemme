import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from torchvision.datasets import MNIST, CIFAR10, CelebA
from flemme.utils import load_img_as_numpy, load_itk, set_random_state, get_random_state, crop_boundingbox, rreplace
from flemme.logger import get_logger
from functools import partial
# image segmentation dataloader
# there might be other type of image dataset, such as image classification dataset or pure image generation dataset.
logger = get_logger('image_dataset')

class ImgDataset(Dataset):
    def __init__(self, data_path, dim = 2, data_transform = None, 
                 mode = 'train', data_dir = '', 
                 data_suffix = '.png', crop_nonzero = None, 
                 **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.img_path_list = sorted(glob(os.path.join(data_path + '/' + data_dir, "*" + data_suffix)))
        self.mode = mode
        self.data_transform = data_transform
        self.dim = dim
        self.data_path = data_path
        self.crop_nonzero = None
        if crop_nonzero:
            if type(crop_nonzero) == dict:
                self.crop_nonzero = partial(crop_boundingbox, **crop_nonzero)
            else:
                self.crop_nonzero = crop_boundingbox
    def __len__(self):
        return len(self.img_path_list)
    ### the dataset will not be stored in the memory
    def __getitem__(self, index):
        """Get the images"""
        img_path = self.img_path_list[index]
        if self.dim == 2:
            img = load_img_as_numpy(img_path)
        else:
            img = load_itk(img_path)
            if self.crop_nonzero:
                img = self.crop_nonzero(data = img)[0]
        if self.data_transform is not None:
            img = self.data_transform(img)
        return img, img_path

class ImgClsDataset(Dataset):
    def __init__(self, data_path, 
                dim = 2,
                data_transform = None, 
                class_label_transform = None,
                mode = 'train',
                data_dir = '', 
                data_suffix = '.png',
                pre_shuffle = False,
                cls_label = {},
                crop_nonzero = None,
                **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.mode = mode
        self.dim = dim
        self.data_transform = data_transform
        self.class_label_transform = class_label_transform
        self.img_path_list = []
        self.labels = []
        class_dirs = list(cls_label.keys())
        for cls_dir in class_dirs:
            sub_path_list = sorted(glob(os.path.join(data_path + '/' + cls_dir + '/' + data_dir,  "*" + data_suffix)))
            self.img_path_list = self.img_path_list + sub_path_list
            assert cls_dir in cls_label, f'Unknowk class: {cls_dir}'
            self.labels = self.labels + [cls_label[cls_dir], ] * len(sub_path_list)
        if pre_shuffle:
            shuffled_index = np.arange(len(self.img_path_list))
            np.random.shuffle(shuffled_index)
            self.img_path_list = [self.img_path_list[i] for i in shuffled_index]
            self.labels = [self.labels[i] for i in shuffled_index]
        self.crop_nonzero = None
        if crop_nonzero:
            if type(crop_nonzero) == dict:
                self.crop_nonzero = partial(crop_boundingbox, **crop_nonzero)
            else:
                self.crop_nonzero = crop_boundingbox
    def __len__(self):
        return len(self.img_path_list)
    ### the dataset will not be stored in the memory
    def __getitem__(self, index):
        """Get the images"""
        img_path = self.img_path_list[index]
        label = self.labels[index]
        if self.dim == 2:
            img = load_img_as_numpy(img_path)
        else:
            img = load_itk(img_path)
            if self.crop_nonzero:
                img = self.crop_nonzero(data = img)[0]
        if self.data_transform is not None:
            img = self.data_transform(img)
        if self.class_label_transform is not None:
            label = self.class_label_transform(label)
        return img, label, img_path

class ImgSegDataset(Dataset):
    def __init__(self, data_path, dim = 2, data_transform = None, mode = 'train', 
                 data_dir = 'raw', label_dir = 'label', data_suffix='.png', 
                 label_suffix = None, label_transform = None, crop_nonzero = None, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        label_suffix = label_suffix or data_suffix
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.img_path_list = sorted(glob(os.path.join(data_path + '/' + data_dir, "*" + data_suffix)))
        self.mask_path_list = [rreplace(rreplace(ipath, data_suffix, label_suffix, 1), data_dir, label_dir, 1) for ipath in self.img_path_list]
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.dim = dim
        self.crop_by = None
        if crop_nonzero:
            self.crop_by = crop_nonzero.pop('crop_by', 'raw')
            if type(crop_nonzero) == dict:
                self.crop_nonzero = partial(crop_boundingbox, **crop_nonzero)
            else:
                self.crop_nonzero = crop_boundingbox
            
    def __len__(self):
        return len(self.img_path_list)
    ### the dataset will not be stored in the memory
    def __getitem__(self, index):
        """Get the images"""
        img_path = self.img_path_list[index]
        
        mask_path = self.mask_path_list[index]
        if self.dim == 2:
            img = load_img_as_numpy(img_path)
            mask = load_img_as_numpy(mask_path)
        else:
            img = load_itk(img_path)
            mask = load_itk(mask_path)
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
        return img, mask, img_path


class ImgReconDataset(Dataset):
    def __init__(self, data_path, dim = 2, data_transform = None, mode = 'train', 
                 data_dir = 'raw', target_dir = 'target', data_suffix='.png', 
                 target_suffix = None, target_transform = None, crop_nonzero = None, **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        target_suffix = target_suffix or data_suffix
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.img_path_list = sorted(glob(os.path.join(data_path+'/' + data_dir, "*" + data_suffix)))
        self.target_path_list = [rreplace(rreplace(ipath, data_suffix, target_suffix, 1), data_dir, target_dir, 1) for ipath in self.img_path_list]
        self.mode = mode
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.dim = dim
        self.crop_by = None
        if crop_nonzero:
            self.crop_by = crop_nonzero.pop('crop_by', 'raw')
            if type(crop_nonzero) == dict:
                self.crop_nonzero = partial(crop_boundingbox, **crop_nonzero)
            else:
                self.crop_nonzero = crop_boundingbox
    def __len__(self):
        return len(self.img_path_list)
    ### the dataset will not be stored in the memory
    def __getitem__(self, index):
        """Get the images"""
        img_path = self.img_path_list[index]
        
        target_path = self.target_path_list[index]
        if self.dim == 2:
            img = load_img_as_numpy(img_path)
            target = load_img_as_numpy(target_path)
        else:
            img = load_itk(img_path)
            target = load_itk(target_path)
            ### currently, crop_by only support 3D images
            if self.crop_by == 'raw':
                img, target, _ = self.crop_nonzero(data = img, follows = target)
            elif self.crop_by == 'target':
                target, img, _ = self.crop_nonzero(data = target, follows = img)
        if self.data_transform is not None:
            ## why do we need to set the state here?
            # the purpose is to make sure that two transforms are applied on the same state. 
            n_state, t_state = get_random_state()
            img = self.data_transform(img)
            ### to check if the target and image has the same random transforms
            # x = np.random.randn(10)
            # x_torch = torch.randn(10)
            set_random_state(n_state, t_state)
            target = self.target_transform(target)
        return img, target, img_path
    

class MultiModalityImgSegDataset(Dataset):
    def __init__(self, data_path, dim = 2, data_transform = None, mode = 'train', 
                 data_dir = 'raw', label_dir = 'label', data_suffix='.png', 
                 label_suffix = None, label_transform = None, crop_nonzero = None, 
                 data_combine = 'mean', label_combine = None, **kwargs):
        super().__init__()
        
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.dim = dim
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
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
                self.img_path_list.append([rreplace( rreplace(rpath, data_suffix[0], sf, 1), data_dir[0], rd, 1) for rpath in self.img_path_list[0]])
        for ld, lsf in zip(label_dir, label_suffix):
            self.mask_path_list.append([rreplace(rreplace(rpath, data_suffix[0], lsf, 1), data_dir[0], ld, 1) for rpath in self.img_path_list[0]])
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.crop_by = None
        if crop_nonzero:
            self.crop_by = crop_nonzero.pop('crop_by', 'raw')
            if type(crop_nonzero) == dict:
                self.crop_nonzero = partial(crop_boundingbox, **crop_nonzero)
            else:
                self.crop_nonzero = crop_boundingbox
        if data_combine is None:
            self.data_combine = lambda x: x
        elif data_combine == 'sum':
            self.data_combine = sum
        elif data_combine == 'mean':
            self.data_combine = lambda x: sum(x) / len(x)
        elif data_combine == 'cat':
            self.data_combine = np.stack
        else:
            logger.error('Unknown way to combine different modality datas.')
            exit(1)
            
        if label_combine is None:
            label_combine = lambda x: x
        elif label_combine == 'sum':
            self.label_combine = sum
        elif label_combine == 'mean':
            self.label_combine = lambda x: sum(x) / len(x)
        elif label_combine == 'cat':
            self.label_combine = np.stack
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
            imgs = [load_itk(img_path) for img_path in img_paths]
            masks = [load_itk(mask_path) for mask_path in mask_paths]
            if len(imgs) == 1:
                img = imgs[0]
            else:
                img = self.data_combine(imgs)
            
            if len(masks) == 1:
                mask = masks[0]
            else:
                mask = self.label_combine(masks)
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
                set_random_state(n_state, t_state)
                mask = self.label_transform(mask)
                return img, mask, img_paths[0]

class ImgSegWithClassLabelDataset(Dataset):
    def __init__(self, data_path, dim = 2,
                 data_transform = None, 
                 label_transform = None,
                 class_label_transform = None, 
                 mode = 'train', 
                 data_dir = 'partial', 
                 label_dir = 'label', 
                 data_suffix = '.png', 
                 label_suffix = None, 
                 cls_label = {},
                 pre_shuffle = False,
                 crop_nonzero = None, 
                 **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.dim = dim
        self.mode = mode
        self.data_transform = data_transform
        self.class_label_transform = class_label_transform
        self.label_transform = label_transform
        self.img_path_list = []
        self.mask_path_list = []
        self.class_labels = []
        label_suffix = label_suffix or data_suffix
        class_dirs = list(cls_label.keys())

        for cls_dir in class_dirs:
            sub_path_list = sorted(glob(os.path.join(data_path, data_dir, cls_dir, "*" + data_suffix)))
            self.img_path_list = self.img_path_list + sub_path_list
            sub_mask_path_list = [rreplace(rreplace(s, data_dir, label_dir, 1), data_suffix, label_suffix, 1) for s in sub_path_list]
            self.mask_path_list = self.mask_path_list + sub_mask_path_list
            assert cls_dir in cls_label, f'Unknowk class: {cls_dir}'
            self.class_labels = self.class_labels + [cls_label[cls_dir], ] * len(sub_path_list)
        if pre_shuffle:
            shuffled_index = np.arange(len(self.img_path_list))
            np.random.shuffle(shuffled_index)
            self.img_path_list = [self.img_path_list[i] for i in shuffled_index]
            self.mask_path_list = [self.mask_path_list[i] for i in shuffled_index]
            self.class_labels = [self.class_labels[i] for i in shuffled_index]
        self.crop_by = None
        if crop_nonzero:
            self.crop_by = crop_nonzero.pop('crop_by', 'raw')
            if type(crop_nonzero) == dict:
                self.crop_nonzero = partial(crop_boundingbox, **crop_nonzero)
            else:
                self.crop_nonzero = crop_boundingbox
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        """Get the images"""
        img_path = self.img_path_list[index]
        
        mask_path = self.mask_path_list[index]
        if self.dim == 2:
            img = load_img_as_numpy(img_path)
            mask = load_img_as_numpy(mask_path)
        else:
            img = load_itk(img_path)
            mask = load_itk(mask_path)
            ### currently, crop_by only support 3D images
            if self.crop_by == 'raw':
                img, mask, _ = self.crop_nonzero(data = img, follows = mask)
            elif self.crop_by == 'mask':
                mask, img, _ = self.crop_nonzero(data = mask, follows = img)
            
        cls_label = self.class_labels[index]
        if self.data_transform:
            n_state, t_state = get_random_state()
            img = self.data_transform(img)
            set_random_state(n_state, t_state)
            mask = self.label_transform(mask)
            if self.class_label_transform:
                cls_label = self.class_label_transform(cls_label)
        return img, mask, cls_label, self.img_path_list[index]


class ImgReconWithClassLabelDataset(Dataset):
    def __init__(self, data_path, dim = 2,
                 data_transform = None, 
                 target_transform = None,
                 class_label_transform = None, 
                 mode = 'train', 
                 data_dir = 'partial', 
                 target_dir = 'target', 
                 data_suffix = '.png', 
                 target_suffix = None, 
                 cls_label = {},
                 pre_shuffle = False,
                 crop_nonzero = None, 
                 **kwargs):
        super().__init__()
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        logger.info("loading data from the directory: {}".format(data_path))
        self.data_path = data_path
        self.dim = dim
        self.mode = mode
        self.data_transform = data_transform
        self.class_label_transform = class_label_transform
        self.target_transform = target_transform
        self.img_path_list = []
        self.target_path_list = []
        self.class_labels = []
        target_suffix = target_suffix or data_suffix
        class_dirs = list(cls_label.keys())

        for cls_dir in class_dirs:
            sub_path_list = sorted(glob(os.path.join(data_path, data_dir, cls_dir, "*" + data_suffix)))
            self.img_path_list = self.img_path_list + sub_path_list
            sub_target_path_list = [rreplace(rreplace(s, data_dir, target_dir, 1), data_suffix, target_suffix, 1) for s in sub_path_list]
            self.target_path_list = self.target_path_list + sub_target_path_list
            assert cls_dir in cls_label, f'Unknowk class: {cls_dir}'
            self.class_labels = self.class_labels + [cls_label[cls_dir], ] * len(sub_path_list)
        if pre_shuffle:
            shuffled_index = np.arange(len(self.img_path_list))
            np.random.shuffle(shuffled_index)
            self.img_path_list = [self.img_path_list[i] for i in shuffled_index]
            self.target_path_list = [self.target_path_list[i] for i in shuffled_index]
            self.class_labels = [self.class_labels[i] for i in shuffled_index]
        self.crop_by = None
        if crop_nonzero:
            self.crop_by = crop_nonzero.pop('crop_by', 'raw')
            if type(crop_nonzero) == dict:
                self.crop_nonzero = partial(crop_boundingbox, **crop_nonzero)
            else:
                self.crop_nonzero = crop_boundingbox
    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        """Get the images"""
        img_path = self.img_path_list[index]
        
        target_path = self.target_path_list[index]
        if self.dim == 2:
            img = load_img_as_numpy(img_path)
            target = load_img_as_numpy(target_path)
        else:
            img = load_itk(img_path)
            target = load_itk(target_path)
            ### currently, crop_by only support 3D images
            if self.crop_by == 'raw':
                img, target, _ = self.crop_nonzero(data = img, follows = target)
            elif self.crop_by == 'target':
                target, img, _ = self.crop_nonzero(data = target, follows = img)
            
        cls_label = self.class_labels[index]
        if self.data_transform:
            n_state, t_state = get_random_state()
            img = self.data_transform(img)
            set_random_state(n_state, t_state)
            target = self.target_transform(target)
            if self.class_label_transform:
                cls_label = self.class_label_transform(cls_label)
        return img, target, cls_label, self.img_path_list[index]

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
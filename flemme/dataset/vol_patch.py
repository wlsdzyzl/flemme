import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from flemme.utils import load_itk, set_random_state, get_random_state, crop_boundingbox, rreplace
from flemme.logger import get_logger
from .img import ImgSegDataset, MultiModalityImgSegDataset
from functools import partial
from .slice_utils import get_slice_builder
logger = get_logger('vol_patch_dataset')

# def _create_padded_indexes(indexes, halo_shape):
#     return tuple(slice(index.start, index.stop + 2 * halo) for index, halo in zip(indexes, halo_shape))

## transfer volume to patches for training with less memory
class PatchSingleImgSegDataset:
    def __init__(self, data, label, 
                slice_builder, mode, 
                **kwargs):
        self.mode = mode
        # self.halo_shape = tuple(slice_builder.get('halo_shape', [0, 0, 0]))
        slice_builder = get_slice_builder(data, label, slice_builder)
        self.data_slices = slice_builder.raw_slices
        self.label_slices = slice_builder.label_slices 
        self.patch_count = len(self.data_slices)
        logger.info(f'Built {self.patch_count} patches for input with a shape of {data.shape}.')
    def __getitem__(self, idx, raw, mask):
        if idx >= len(self):
            raise StopIteration
        data_idx = self.data_slices[idx]
        label_idx = self.label_slices[idx]
        ## we only return slice indices related to shape 
        return raw[data_idx], mask[label_idx], data_idx[-3:]

    def __len__(self):
        return self.patch_count

class PatchImgSegDataset(ImgSegDataset):
    def __init__(self, data_path, slice_builder, data_transform = None, mode = 'train', 
                 data_dir = 'raw', label_dir = 'label', data_suffix='.png', 
                 label_suffix = None, label_transform = None, crop_nonzero = None, 
                 lazy_loading = True, **kwargs):
        super().__init__(dim = 3, data_path = data_path,
            data_transform = data_transform,
            mode = mode, data_dir = data_dir,
            label_dir = label_dir,
            data_suffix = data_suffix,
            label_suffix = label_suffix,
            label_transform = label_transform,
            crop_nonzero = crop_nonzero )
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.lazy_loading = lazy_loading
        if not self.lazy_loading:
            self.raw_list = []
            self.mask_list = []
        patch_count = 0
        self.end_indices = []
        self.datasets = []
        for did in range(len(self.img_path_list)):
            raw, mask = self.load_data(did)
            tmp_dataset = PatchSingleImgSegDataset(
                data = raw, label = mask, 
                slice_builder = slice_builder,
                mode = mode)
            self.datasets.append(tmp_dataset)
            patch_count += len(tmp_dataset)
            self.end_indices.append(patch_count) 
            if not self.lazy_loading:
                self.raw_list.append(raw)
                self.mask_list.append(mask)
        self.patch_count = patch_count
        
    def load_data(self, dataset_id):
        _raw = load_itk(self.img_path_list[dataset_id])[0]
        _mask = load_itk(self.mask_path_list[dataset_id])[0]
        if self.crop_by == 'raw':
            _raw, _mask, _ = self.crop_nonzero(data = _raw, follows = _mask)
        elif self.crop_by == 'mask':
            _mask, _raw, _ = self.crop_nonzero(data = _mask, follows = _raw)
        return _raw, _mask
    def __getidx__(self, idx):
        if idx >= len(self) or idx < 0:
            raise StopIteration
        ## dataset id of this idx
        dataset_idx = -1
        ## local patch idx in the dataset
        local_idx = -1
        for did, eid in enumerate(self.end_indices):
            if idx < eid:
                dataset_idx = did
                break
        local_idx = idx - (0 if dataset_idx == 0 else self.end_indices[dataset_idx - 1])
        return dataset_idx, local_idx
    def __getitem__(self, idx):
        dataset_idx, local_idx = self.__getidx__(idx)
        if self.lazy_loading:
            raw, mask = self.load_data(dataset_idx)
        else:
            raw, mask = self.raw_list[dataset_idx], self.mask_list[dataset_idx]
        raw, mask, slice_idx = self.datasets[dataset_idx].__getitem__(local_idx, raw, mask)
        if self.data_transform is not None:
            n_state, t_state = get_random_state()
            raw = self.data_transform(raw)
            set_random_state(n_state, t_state)
            mask = self.label_transform(mask)
        return raw, mask, slice_idx, self.img_path_list[dataset_idx]
    def __len__(self):
        return self.patch_count


class MultiModalityPatchImgSegDataset(MultiModalityImgSegDataset):
    def __init__(self, data_path, slice_builder, data_transform = None, mode = 'train', 
                 data_dir = 'raw', label_dir = 'label', data_suffix='.png', 
                 label_suffix = None, label_transform = None, crop_nonzero = None, 
                 data_combine = 'mean', label_combine = None,
                 lazy_loading = True, **kwargs):
        super().__init__(dim = 3, data_path = data_path, 
                    data_transform = data_transform, mode = mode, 
                    data_dir = data_dir, 
                    label_dir = label_dir, 
                    data_suffix=data_suffix, 
                    label_suffix = label_suffix, 
                    label_transform = label_transform, 
                    crop_nonzero = crop_nonzero, 
                    data_combine = data_combine, 
                    label_combine = label_combine,)
        if len(kwargs) > 0:
            logger.debug("redundant parameters: {}".format(kwargs))
        self.lazy_loading = lazy_loading
        if not self.lazy_loading:
            self.raw_list = []
            self.mask_list = []
        patch_count = 0
        self.end_indices = []
        self.datasets = []
        for did in range(len(self.img_path_list[0])):
            raw, mask = self.load_data(did)
            tmp_dataset = PatchSingleImgSegDataset(
                data = raw, label = mask, 
                slice_builder = slice_builder,
                mode = mode)
            self.datasets.append(tmp_dataset)
            patch_count += len(tmp_dataset)
            self.end_indices.append(patch_count) 
            if not self.lazy_loading:
                self.raw_list.append(raw)
                self.mask_list.append(mask)
        self.patch_count = patch_count
        
    def load_data(self, dataset_id):
        img_paths = [ l[dataset_id] for l in self.img_path_list]
        mask_paths = [ l[dataset_id] for l in self.mask_path_list]
        imgs = [load_itk(img_path)[0] for img_path in img_paths]
        masks = [load_itk(mask_path)[0] for mask_path in mask_paths]
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
        return img, mask
    def __getidx__(self, idx):
        if idx >= len(self) or idx < 0:
            raise StopIteration
        ## dataset id of this idx
        dataset_idx = -1
        ## local patch idx in the dataset
        local_idx = -1
        for did, eid in enumerate(self.end_indices):
            if idx < eid:
                dataset_idx = did
                break
        local_idx = idx - (0 if dataset_idx == 0 else self.end_indices[dataset_idx - 1])
        return dataset_idx, local_idx
    def __getitem__(self, idx):
        dataset_idx, local_idx = self.__getidx__(idx)
        if self.lazy_loading:
            raw, mask = self.load_data(dataset_idx)
        else:
            raw, mask = self.raw_list[dataset_idx], self.mask_list[dataset_idx]
        raw, mask, slice_idx = self.datasets[dataset_idx].__getitem__(local_idx, raw, mask)
        if self.data_transform is not None:
            n_state, t_state = get_random_state()
            raw = self.data_transform(raw)
            set_random_state(n_state, t_state)
            mask = self.label_transform(mask)
        return raw, mask, slice_idx, self.img_path_list[0][dataset_idx]
    def __len__(self):
        return self.patch_count
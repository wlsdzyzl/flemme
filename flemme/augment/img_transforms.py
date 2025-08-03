from torchvision.transforms import Resize as TResize, InterpolationMode, ToTensor,\
    RandomHorizontalFlip, RandomVerticalFlip, Normalize,\
    RandomRotation, GaussianBlur, CenterCrop, RandomCrop, Compose
from torchvision.transforms.functional import rgb_to_grayscale
from functools import partial
from scipy.ndimage import map_coordinates, gaussian_filter
import numpy as np
import torch
import torch.nn.functional as F
from flemme.utils import label_to_onehot
from scipy.ndimage import distance_transform_edt as eucl_distance


class ToOneHot:
    """
    To one hot label, background value should be 0
    """
    def __init__(self, num_classes, **kwargs):
        self.to_onehot = partial(label_to_onehot, 
            num_classes = num_classes, 
            channel_dim = 0)
    def __call__(self, m):
        if not type(m) == int:
            assert m.ndim == 2 or m.ndim == 3, "Not a 2D image or class label"
            if m.ndim == 3:
                assert m.shape[0] == 1, \
                    "Label is a multi channel 2D image. Check if it's already a one hot embedding."
                m = m[0]
        return self.to_onehot(m)

class Resize:
    def __init__(self, size, mode = 'nearest'):
        if mode == 'nearest':
            inter_mode =  InterpolationMode.NEAREST
        elif mode == 'bilinear':
            inter_mode = InterpolationMode.BILINEAR
        elif mode == 'bicubic':
            inter_mode = InterpolationMode.BICUBIC
        else:
            raise NotImplementedError
        self.resize = TResize(size = size, 
            interpolation = inter_mode)
    def __call__(self, data):
        return self.resize(data)
        
class ToBinaryMask:
    def __init__(self, threshold=0):
        self.threshold = threshold
    def __call__(self, m):
        return m > self.threshold
    
class GrayScale:
    def __init__(self, out_channel = 1):
        self.out_channel = out_channel
    def __call__(self, m):
        return rgb_to_grayscale(m, self.out_channel)

### change white to black
class InverseColor:
    def __init__(self):
        pass
    def __call__(self, m):
        return 1.0 - m
class Relabel:
    """
    Relabel labels into a consecutive numbers, e.g.
    [10, 10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.
    """

    def __init__(self, map = [], offset = None, **kwargs):
        self.map = map
        self.offset = offset
    def __call__(self, m):
        if self.offset is not None:
            m = m + self.offset
        elif len(self.map) > 0:
            if type(m) == int:
                if m in self.map:
                    m = self.map(m)
            else:              
                for kv in self.map:
                    k, v = kv
                    m[m == k] = v
        else:
            if torch.is_tensor(m):
                _, unique_labels = torch.unique(m, return_inverse = True)
            else:
                _, unique_labels = np.unique(m, return_inverse = True)
            m = unique_labels.reshape(m.shape)
        return m

class ElasticDeform:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, spline_order = 3, alpha=2000, sigma=50, execution_probability=0.1,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        """
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, m):
        is_tensor = torch.is_tensor(m)
        if is_tensor:
            m = m.cpu().numpy()
        if np.random.uniform() < self.execution_probability:
            assert m.ndim in [2, 3]

            if m.ndim == 3:
                image_shape = m.shape
            else:
                image_shape = m[0].shape

            dy, dx = [
                gaussian_filter(
                    np.random.randn(*image_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            y_dim, x_dim = image_shape
            y, x = np.meshgrid(np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = y + dy, x + dx

            if m.ndim == 2:
                res = map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                res = np.stack(channels, axis=0)
        if is_tensor:
            res = torch.tensor(res, dtype=torch.float32)
        return res


def one_hot2dist(seg, resolution = [1, 1]):
    # assert one_hot(torch.tensor(seg), axis=0)
    is_tensor = torch.is_tensor(seg)
    if is_tensor:
        seg = seg.cpu().numpy()
    K: int = len(seg)
    res = np.zeros_like(seg)
    for k in range(K):
        posmask = seg[k].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel
    if is_tensor:
        res = torch.tensor(res, dtype=torch.float32)
    return res

class DistMap:
    def __init__(self, resolution = [1, 1]):
        self.to_distmap = partial(one_hot2dist, resolution=resolution)
    def __call__(self, m):
        ## m need to be one-hot embedding.
        m = self.to_distmap(m)
        return m

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class ClipTransform:
    def __init__(self):
        size = 224
        self.comp = Compose([
            Resize(size, mode='bicubic'),
            CenterCrop(size),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    def __call__(self, m):
        return self.comp(m)

import importlib
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, binary_opening, binary_closing
from skimage.filters import gaussian
from flemme.utils import label_to_onehot
from functools import partial
from .img_transforms import DistMap, Relabel, ToInt
eps = 1e-8

class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, axis_prob=0.5):
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if np.random.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class RandomRotate90:
    """
    Rotate an array by 90 degrees around xy plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self):
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = np.random.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, angle_spectrum=30, axes=None, mode='reflect', order=0):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[np.random.randint(len(self.axes))]
        angle = np.random.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomContrast:
    """
    Adjust contrast by scaling each voxel to `mean + alpha * (v - mean)`.
    """

    def __init__(self, alpha=(0.5, 1.5), mean=0.0, execution_probability=0.1):
        assert len(alpha) == 2
        self.alpha = alpha
        self.mean = mean
        self.execution_probability = execution_probability

    def __call__(self, m):
        if np.random.uniform() < self.execution_probability:
            alpha = np.random.uniform(self.alpha[0], self.alpha[1])
            result = self.mean + alpha * (m - self.mean)
            return np.clip(result, -1, 1)

        return m


# it's relatively slow, i.e. ~1s per patch of size 64x200x200, so use multiple workers in the DataLoader
# remember to use spline_order=0 when transforming the labels
class ElasticDeform:
    """
    Apply elasitc deformations of 3D patches on a per-voxel mesh. Assumes ZYX axis order (or CZYX if the data is 4D).
    Based on: https://github.com/fcalvet/image_tools/blob/master/image_augmentation.py#L62
    """

    def __init__(self, spline_order = 3, alpha=2000, sigma=50, execution_probability=0.1, apply_3d=True,
                 **kwargs):
        """
        :param spline_order: the order of spline interpolation (use 0 for labeled images)
        :param alpha: scaling factor for deformations
        :param sigma: smoothing factor for Gaussian filter
        :param execution_probability: probability of executing this transform
        :param apply_3d: if True apply deformations in each axis
        """
        self.spline_order = spline_order
        self.alpha = alpha
        self.sigma = sigma
        self.execution_probability = execution_probability
        self.apply_3d = apply_3d

    def __call__(self, m):
        if np.random.uniform() < self.execution_probability:
            assert m.ndim in [3, 4]

            if m.ndim == 3:
                volume_shape = m.shape
            else:
                volume_shape = m[0].shape

            if self.apply_3d:
                dz = gaussian_filter(np.random.randn(*volume_shape), self.sigma, mode="reflect") * self.alpha
            else:
                dz = np.zeros_like(m)

            dy, dx = [
                gaussian_filter(
                    np.random.randn(*volume_shape),
                    self.sigma, mode="reflect"
                ) * self.alpha for _ in range(2)
            ]

            z_dim, y_dim, x_dim = volume_shape
            z, y, x = np.meshgrid(np.arange(z_dim), np.arange(y_dim), np.arange(x_dim), indexing='ij')
            indices = z + dz, y + dy, x + dx

            if m.ndim == 3:
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)

        return m


class CenterCropXY:
    def __init__(self, size=(256, 256)):
        self.crop_y, self.crop_x = size
    @staticmethod
    def _padding(pad_total):
        half_total = pad_total // 2
        return (half_total, pad_total - half_total)

    @staticmethod
    def _start_and_pad(crop_size, max_size):
        if crop_size < max_size:
            return (max_size - crop_size) // 2, (0, 0)
        else:
            return 0, _padding(crop_size - max_size)

    def __call__(self, m):

        assert m.ndim in (3, 4)
        if m.ndim == 3:
            _, y, x = m.shape
        else:
            _, _, y, x = m.shape

        y_start, y_pad = self._start_and_pad(self.crop_y, y)
        x_start, x_pad = self._start_and_pad(self.crop_x, x)

        if m.ndim == 3:
            result = m[:, y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
            return np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode='reflect')
        else:
            channels = []
            for c in range(m.shape[0]):
                result = m[c][:, y_start:y_start + self.crop_y, x_start:x_start + self.crop_x]
                channels.append(np.pad(result, pad_width=((0, 0), y_pad, x_pad), mode='reflect'))
            return np.stack(channels, axis=0)

class RandomCropXY(CenterCropXY):
    def __init__(self, size=(256, 256)):
        super().__init__(size = size)
    @staticmethod
    def _start_and_pad(crop_size, max_size):
        if crop_size < max_size:
            return np.random.randint(max_size - crop_size), (0, 0)
        else:
            return 0, _padding(crop_size - max_size)

class Normalize:
    """
    Apply Z-score normalization to a given input tensor, i.e. re-scaling the values to be 0-mean and 1-std.
    """

    def __init__(self, mean=None, std=None, channelwise=False):
        if mean is not None or std is not None:
            assert mean is not None and std is not None
        self.mean = mean
        self.std = std
        self.channelwise = channelwise

    def __call__(self, m):
        if self.mean is not None:
            mean, std = self.mean, self.std
        else:
            if self.channelwise:
                # normalize per-channel
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                mean = np.mean(m, axis=axes, keepdims=True)
                std = np.std(m, axis=axes, keepdims=True)
            else:
                mean = np.mean(m)
                std = np.std(m)

        return (m - mean) / np.clip(std, a_min=eps, a_max=None)




class MinMaxNormalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data
    in a fixed range of [-1, 1] or in case of norm01==True to [0, 1]. In addition, data can be
    clipped by specifying min_value/max_value either globally using single values or via a
    list/tuple channelwise if enabled.
    """

    def __init__(self, min_value=None, max_value=None, norm01=True, channelwise=False):
        if min_value is not None and max_value is not None:
            assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.norm01 = norm01
        self.channelwise = channelwise

    def __call__(self, m):
        if self.channelwise:
            # get min/max channelwise
            axes = list(range(m.ndim))
            axes = tuple(axes[1:])
            if self.min_value is None or 'None' in self.min_value:
                min_value = np.min(m, axis=axes, keepdims=True)

            if self.max_value is None or 'None' in self.max_value:
                max_value = np.max(m, axis=axes, keepdims=True)

            # check if non None in self.min_value/self.max_value
            # if present and if so copy value to min_value
            if self.min_value is not None:
                for i,v in enumerate(self.min_value):
                    if v != 'None':
                        min_value[i] = v

            if self.max_value is not None:
                for i,v in enumerate(self.max_value):
                    if v != 'None':
                        max_value[i] = v
        else:
            if self.min_value is None:
                min_value = np.min(m)
            else:
                min_value = self.min_value

            if self.max_value is None:
                max_value = np.max(m)
            else:
                max_value = self.max_value

        # calculate norm_0_1 with min_value / max_value with the same dimension
        # in case of channelwise application
        norm_0_1 = (m - min_value) / (max_value - min_value + eps)

        if self.norm01 is True:
          return np.clip(norm_0_1, 0, 1)
        else:
          return np.clip(2 * norm_0_1 - 1, -1, 1)

class Resize(object):
    def __init__(self, size, mode = 'nearest'):
        self.size = size
        self.mode = mode
    def __call__(self, data):
        origin_ndim = data.ndim
        ### unsqueeze
        if data.ndim == 3:
            data = data[None, None, ...]
        elif data.ndim == 4:
            data = data[None, ...]

        data = F.interpolate(torch.from_numpy(data+0.0), size = self.size, mode = self.mode, 
            recompute_scale_factor = False).numpy()
        ## squeeze
        data = data[0]
        if data.ndim != origin_ndim:
            data = data[0]
        return data

class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
        dtype (np.dtype): the desired output data type
    """

    def __init__(self, expand_dims=True):
        self.expand_dims = expand_dims
        # assert dtype in ['float', 'double'], "Only support 'float' or 'double'"
        # if dtype == 'float':
        #     self.dtype = torch.float
        # elif dtype == 'double':
        #     self.dtype = torch.double


    def __call__(self, m):
        if type(m) == float or type(m) == int:
            return torch.tensor(m)
        assert m.ndim in [1, 3, 4], 'Supports only onehot-label (C), 3D (DxHxW) or 4D (CxDxHxW) images'
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)
        if m.dtype == int or m.dtype == np.int32:
            return torch.tensor(m).long()
        return torch.tensor(m).float()


class ToOneHot:
    """
    To one hot label, background value should be 0
    """
    def __init__(self, num_classes):
        self.to_onehot = partial(label_to_onehot, 
            num_classes = num_classes, 
            channel_dim = 0)
    def __call__(self, m):
        if not type(m) == int:
            assert m.ndim == 3 or m.ndim == 4, "Not a 3D image or class label"
            if m.ndim == 4:
                assert m.shape[0] == 1, \
                    "Label is a multi channel 3D image. Check if it's already a one hot embedding."
                m = m[0]
        return self.to_onehot(m)


class GaussianBlur:
    def __init__(self, sigma=[.1, 2.], execution_probability=0.5):
        self.sigma = sigma
        self.execution_probability = execution_probability

    def __call__(self, x):
        if random.random() < self.execution_probability:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            x = gaussian(x, sigma=sigma)
            return x
        return x

class RemoveSmallGap:
    def __init__(self, iterations):
        self.iterations = iterations
    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        dtype = m.dtype
        m = m > 0.0
        if m.ndim == 4:
            for c in m.shape[0]:
                m[c] = binary_closing(m[c], iterations = self.iterations)
        else:
            m = binary_closing(m, iterations = self.iterations)
        return m.astype(dtype)

class RemoveThinConnection:
    def __init__(self, iterations = 1):
        self.iterations = iterations
    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'
        dtype = m.dtype
        m = m > 0.0
        if m.ndim == 4:
            for c in m.shape[0]:
                m[c] = binary_opening(m[c], iterations = self.iterations)
        else:
            m = binary_opening(m, iterations = self.iterations)
        return m.astype(dtype)

class ToBinaryMask:
    def __init__(self, threshold=0):
        self.threshold = threshold
    def __call__(self, m):
        return m > self.threshold
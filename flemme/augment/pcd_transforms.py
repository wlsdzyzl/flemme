import torch
import numpy as np
import math
import random
import numbers
from itertools import repeat
from flemme.utils import label_to_onehot, normalize
from functools import partial
from .hilbert_sort import HilbertSort3D
import fpsample
from .img_transforms import Relabel
#### transfoms for point cloud
## modified from diffusion-point-cloud
class ToTensor(object):
    r"""Numpy to tensor"""

    def __init__(self, dtype = 'float'):
        assert dtype in ['float', 'double'], "Only support 'float' or 'double'"
        if dtype == 'float':
            self.dtype = torch.float
        elif dtype == 'double':
            self.dtype = torch.double
    def __call__(self, data):
        if type(data) == int:
            return torch.tensor(data)
        return torch.tensor(data).type(self.dtype)



class Normalize(object):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`.
    """

    def __init__(self, method = 'minmax'):
        self.method = method

    def __call__(self, data):
        return normalize(data, channel_dim = -1, method = self.method)

class FixedPoints(object):
    r"""Samples a fixed number of :obj:`num` points and features from a point
    cloud.
    Args:
        num (int): The number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples fixed
            points without replacement. In case :obj:`num` is greater than
            the number of points, duplicated points are kept to a
            minimum. (default: :obj:`True`)
    """

    def __init__(self, num, replace=True, method = 'random', kdtree_height = 5):
        self.num = num
        self.replace = replace
        self.fpsampler = None
        if method == 'fps':
            self.fpsampler = partial(fpsample.fps_sampling, n_samples = self.num)
        elif method == 'qfps':
            self.fpsampler = partial(fpsample.bucket_fps_kdline_sampling, n_samples = self.num, h = kdtree_height)
        else:
            assert method == 'random', f'Unsupported sample strategy: {method}'
    def __call__(self, data):
        point_num = len(data)
        if point_num < self.num or self.fpsampler is None:
            if self.replace:
                choice = np.random.choice(point_num, self.num, replace=True)
            else:
                choice = torch.cat([
                    torch.randperm(point_num)
                    for _ in range(math.ceil(self.num / point_num))
                ], dim=0)[:self.num]
        else:
            choice = self.fpsampler(data)
        data = data[choice]
        return data

class LinearTransformation(object):
    r"""Transforms node positions with a square transformation matrix computed
    offline.
    Args:
        matrix (Tensor): tensor with shape :math:`[D, D]` where :math:`D`
            corresponds to the dimensionality of node positions.
    """

    def __init__(self, matrix):
        assert matrix.dim() == 2, (
            'Transformation matrix should be two-dimensional.')
        assert matrix.size(0) == matrix.size(1), (
            'Transformation matrix should be square. Got [{} x {}] rectangular'
            'matrix.'.format(*matrix.size()))

        self.matrix = matrix

    def __call__(self, data):
        data = torch.matmul(data, self.matrix.to(data.dtype).to(data.device))
        return data

class Rotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): rotation degree
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degree, axis=0):
        self.degree = degree
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * self.degree / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix), )(data)

class RandomRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a given interval.
    Args:
        degrees (tuple or float): Rotation interval from which the rotation
            angle is sampled. If :obj:`degrees` is a number instead of a
            tuple, the interval is given by :math:`[-\mathrm{degrees},
            \mathrm{degrees}]`.
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degrees, axis=0):
        if isinstance(degrees, numbers.Number):
            degrees = (-abs(degrees), abs(degrees))
        assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        self.degrees = degrees
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        ## get transformation matrix
        if self.axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif self.axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix))(data)


class AddNoise(object):

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        data = data + torch.normal(mean=0, std=self.std, size=data.size())
        return data

## std is random
class AddRandomNoise(object):

    def __init__(self, std_range=[0, 0.10]):
        self.std_range = std_range

    def __call__(self, data):
        noise_std = random.uniform(*self.std_range)
        data = data + torch.normal(mean=0, std=noise_std, size=data.size())
        return data


class RandomScale(object):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix
    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & s \\
        \end{bmatrix}
    for three-dimensional positions.
    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """

    def __init__(self, scales):
        assert isinstance(scales, (tuple, list)) and len(scales) == 2
        self.scales = scales

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data = data * scale
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.scales)


class RandomTranslate(object):
    r"""Translates node positions by randomly sampled translation values
    within a given interval. In contrast to other random transformations,
    translation is applied separately at each position.
    Args:
        translate (sequence or float or int): Maximum translation in each
            dimension, defining the range
            :math:`(-\mathrm{translate}, +\mathrm{translate})` to sample from.
            If :obj:`translate` is a number instead of a sequence, the same
            range is used for each dimension.
    """

    def __init__(self, translate):
        self.translate = translate

    def __call__(self, data):
        (n, dim), t = data.size(), self.translate
        if isinstance(t, numbers.Number):
            t = list(repeat(t, times=dim))
        assert len(t) == dim

        ts = []
        for d in range(dim):
            ts.append(data.new_empty(n).uniform_(-abs(t[d]), abs(t[d])))

        data = data + torch.stack(ts, dim=-1)

        return data



class ShufflePoints(object):
    r"""Shuffle order of points in point cloud
    """
    def __init__(self):
        ...
    def __call__(self, data):
        rand_idx = np.arange(len(data))
        np.random.shuffle(rand_idx)
        return data[rand_idx]

class ReorderByAxis(object):
    def __init__(self, axis=0):
        self.axis = axis
    def __call__(self, data):
        index = np.argsort(data[:, self.axis])
        return data[index]

class ReorderByHilbert(object):
    def __init__(self, bins = 16, radius = 1.0, origin = (0,0,0)):
        self.hilbert = HilbertSort3D(bins = bins, radius = radius, origin = origin)
    def __call__(self, data):
        return self.hilbert.sort(data)

class ToOneHot:
    """
    To one hot label, background value should be 0
    """
    def __init__(self, num_classes, ignore_background = False, **kwargs):
        self.to_onehot = partial(label_to_onehot, num_classes = num_classes, 
            ignore_background = ignore_background, channel_dim = -1)
    def __call__(self, m):
        if not type(m) == int:
            assert m.ndim == 1 or m.ndim == 2, "Not a per-point label or class label"
            if m.ndim == 2:
                assert m.shape[-1] == 1, \
                    "Label is a multi channel point cloud. Check if it's already a one hot embedding."
                m = m[..., 0]
        return self.to_onehot(m)

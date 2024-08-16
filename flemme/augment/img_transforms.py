from torchvision.transforms import Resize as TResize, InterpolationMode, ToTensor,\
    RandomHorizontalFlip, RandomVerticalFlip, Normalize,\
    RandomRotation, GaussianBlur, CenterCrop, RandomCrop
from torchvision.transforms.functional import rgb_to_grayscale
from functools import partial
import torch
import torch.nn.functional as F

class ToOneHot:
    """
    To one hot label, background value should be 0
    """
    def __init__(self, num_classes = None, ignore_background = False, **kwargs):
        self.to_onehot = partial(F.one_hot, num_classes = num_classes)
        self.ignore_background = ignore_background
    def __call__(self, m):
        assert m.ndim == 2 or m.ndim == 3, "Not a 2D image"
        if m.ndim == 3:
            assert m.shape[0] == 1, \
                "Label is a multi channel image. Check if it's already a one hot embedding."
            m = m[0]
        m = self.to_onehot(m).float()
        return m.permute(2, 0, 1 )

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
    Relabel a numpy array of labels into a consecutive numbers, e.g.
    [10, 10, 0, 6, 6] -> [2, 2, 0, 1, 1]. Useful when one has an instance segmentation volume
    at hand and would like to create a one-hot-encoding for it. Without a consecutive labeling the task would be harder.
    """

    def __init__(self, map = [], **kwargs):
        self.map = map
    def __call__(self, m):
        if len(self.map) > 0:
            for kv in self.map:
                k, v = kv
                m[m == k] = v
        else:
            _, unique_labels = torch.unique(m, return_inverse=True)
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
                return map_coordinates(m, indices, order=self.spline_order, mode='reflect')
            else:
                channels = [map_coordinates(c, indices, order=self.spline_order, mode='reflect') for c in m]
                return np.stack(channels, axis=0)

        return m

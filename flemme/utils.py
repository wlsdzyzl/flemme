## define some functions about load and save data
from flemme.config import module_config
import numpy as np
# import sys, getopt
import os
# import glob
# import imageio
# import mcubes
from PIL import Image
import SimpleITK as sitk
from enum import Enum, auto
import nibabel as nb
from matplotlib import figure as mpl_figure
from flemme.logger import get_logger
import torch
import torch.nn.functional as F
import argparse
import yaml
from flemme.block import channel_recover
import shutil

logger = get_logger('utils')

class DataForm(Enum):
    VEC = auto()
    IMG = auto()
    PCD = auto()

def get_random_state():
    return np.random.get_state(), torch.get_rng_state()
def get_class(class_name, module):
    clazz = getattr(module, class_name, None)
    return clazz
def set_random_state(np_state, torch_state):
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)
def relabel(m):
    _, unique_labels = np.unique(m, return_inverse=True)
    m = unique_labels.reshape(m.shape)
    return m
def get_coordinates(volume_size, dtype = float):
    dimension = len(volume_size)
    return np.stack(np.meshgrid(*[np.arange(0.0, size).astype(dtype) for size in volume_size], indexing='ij'), axis=-1).reshape(-1, dimension)

def onehot_to_label(label, channel_dim = 0, keepdim = False):
    if torch.is_tensor(label):
        return label.argmax(dim=channel_dim, keepdim=keepdim)
    else:
        return label.argmax(axis=channel_dim, keepdims=keepdim)

def label_to_onehot(m, channel_dim = 0, num_classes = None, 
    ignore_background = False):
    assert channel_dim == 0 or channel_dim == -1, "channel dim should be 0 or -1."
    if torch.is_tensor(m):
        if ignore_background:
            m = m - 1
            num_classes = num_classes - 1
        m_shape = m.shape
        m = m.flatten()
        res = torch.zeros(( m.numel(), num_classes, ))
        res[m >= 0] = F.one_hot(m[m >= 0], num_classes = num_classes).float()
        res_shape = m_shape +  (num_classes, )
        if channel_dim == 0:
            res = res.transpose(0, 1)
            res_shape = (num_classes, )  + m_shape
        return res.reshape(res_shape)
    else:
        m = m.astype(int)
        if not num_classes:
            num_classes = m.max() + 1
        if ignore_background:
            m = m - 1
            num_classes = num_classes - 1
        m_shape = m.shape
        m = m.flatten()
        res = np.zeros(( num_classes, m.size, ))
        res[m[m >= 0], np.arange(m.size)[m >= 0]] = 1
        res_shape = (num_classes, )  + m_shape
        if channel_dim == -1:
            res = res.transpose()
            res_shape = m_shape + (num_classes, ) 
        return res.reshape(res_shape)

def logits_to_onehot_label(logits, data_form):
    if torch.is_tensor(logits):
        ### in batch form
        if data_form == DataForm.IMG:
            channel_dim = 1
        else:
            channel_dim = -1
        c = logits.shape[channel_dim]
        ### binary segmentation
        if c == 1:
            return (logits > 0).float()
        res = F.one_hot(logits.argmax(dim = channel_dim), 
            num_classes = c)
        if channel_dim == -1:
            return res
        else:
            return channel_recover(res)
    else:
        ### without batch
        if data_form == DataForm.IMG:
            channel_dim = 0
        else:
            channel_dim = -1
        c = logits.shape[channel_dim]
        if logits.shape[channel_dim] == 1:
            return logits > 0.5
        return label_to_onehot(logits.argmax(axis = channel_dim),
        channel_dim = channel_dim, num_classes = c)

### normalize to [-1, 1] with min-max or mean
def normalize(data, channel_dim = None, 
              scaling_per_channel = False, 
              method='minmax', return_transform = False, 
              center = None, scaling = None):
    assert center is not None or method in ['minmax', 'mean'], \
        'At least one of center or method should be specified. method should be minmax or mean.'
    if channel_dim is None: reduced_dims = None 
    else:
        if channel_dim < 0: 
            channel_dim = data.ndim + channel_dim
        reduced_dims = tuple(filter(lambda x: not x == channel_dim, range(data.ndim)))
    ### numpy array
    if not torch.is_tensor(data):
        if center is None:
            if method == 'minmax':
                center = (np.max(data, axis = reduced_dims, keepdims = True) + \
                    np.min(data, axis = reduced_dims, keepdims = True)) / 2
            else:
                center = np.mean(data, axis = reduced_dims, keepdims = True)
        cdata = data - center
        if scaling is None:
            data_range = np.max(np.abs(cdata), axis = reduced_dims, keepdims = True) 
            if not channel_dim is None and not scaling_per_channel: 
                scaling =  1.0 / np.max(data_range, keepdims = True) 
            else:
                scaling = 1.0 / data_range
        res = np.clip( np.nan_to_num(cdata * scaling ), -1.0, 1.0)
        if return_transform:
            return res, (center, scaling)
        return res
    else:
        if center is None:
            if method == 'minmax':
                center = (torch.amax(data, dim = reduced_dims, keepdim = True) + \
                    torch.amin(data, dim = reduced_dims, keepdim = True)) / 2
            else:
                center = torch.mean(data, dim = reduced_dims, keepdim = True)
        cdata = data - center
        if scaling is None:
            data_range = torch.amax(cdata.abs(), dim = reduced_dims, keepdim = True) 
            if not channel_dim is None and not scaling_per_channel: 
                scaling =  1.0 / torch.amax(data_range, keepdim = True) 
            else:
                scaling = 1.0 / data_range
        res = torch.clamp(torch.nan_to_num( cdata * scaling  ), -1.0, 1.0 )
        if return_transform:
            return res, (center, scaling)
        return res
#### to [0, 1]
def normalize_img(img, method = 'minmax', 
                  channel_dim = 0,
                  percentile_range = None, 
                  return_transform = False,
                  center = None, scaling = None):
    if percentile_range is not None:
        min_value = np.nanpercentile(img, percentile_range[0])
        max_value = np.nanpercentile(img, percentile_range[1])
        img = np.clip(img, min_value, max_value)

    res = normalize(img, channel_dim = channel_dim, 
                        method=method, center=center, 
                        scaling=scaling, 
                        return_transform=return_transform) 
    if return_transform:
        return 0.5 * (res[0] + 1.0), res[1]
    return 0.5 * (res + 1.0)
#### normalize each image in the batch
def batch_normalize(bdata, channel_dim = None, 
                    scaling_per_channel = False, 
                    method = 'minmax', return_transform = False, 
                    center = None, scaling = None):
    ### keep 0 and channel dims
    if not channel_dim is None: 
        if channel_dim < 0: 
            channel_dim = bdata.ndim + channel_dim
    reduced_dims = tuple(filter(lambda x: not x == channel_dim, range(1, bdata.ndim)))
    if center is None:
        if method == 'minmax':
            center = (torch.amax(bdata, dim = reduced_dims, keepdim = True) + \
                torch.amin(bdata, dim = reduced_dims, keepdim = True)) / 2
        else:
            center = torch.mean(bdata, dim = reduced_dims, keepdim = True)
    cdata = bdata - center
    data_range = torch.amax(cdata.abs(), dim = reduced_dims, keepdim = True) 
    if scaling is None:
        if not channel_dim is None and not scaling_per_channel: 
            scaling =  1.0 / torch.amax(data_range, dim = channel_dim, keepdim=True)
        else:
            scaling = 1.0 / data_range
    res = torch.clamp(torch.nan_to_num(cdata * scaling), -1.0, 1.0)
    if return_transform:
        return res, (center, scaling)
    return res

def load_config(config_path = None):
    if config_path is None:
        parser = argparse.ArgumentParser(description='flemme')
        parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
        args = parser.parse_args()
        config = yaml.safe_load(open(args.config, 'r'))
    else:
        config = yaml.safe_load(open(config_path, 'r'))
        return config
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        logger.info(f"Device specified in config: '{device_str}'")
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            logger.info('CUDA not available, using CPU')
            device_str = 'cpu'
    else:
        device_str = "cuda:0" if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config

### load itk, support nii, mhd, nrrd
def load_itk(filename):
    # loads the image using SimpleITK
    try:
        itkimage = sitk.ReadImage(filename)
    except:
        logger.debug('Orthonormal direction error occurs, try to fix it')
        img = nb.load(filename)
        qform = img.get_qform()
        sform = img.get_sform()
        img.set_qform(qform)
        img.set_sform(sform)
        nb.save(img, filename)
        itkimage = sitk.ReadImage(filename)
    imageArray = sitk.GetArrayFromImage(itkimage)
    origin = itkimage.GetOrigin()
    spacing = itkimage.GetSpacing()
    return imageArray, origin, spacing

### save itk, support nii, mhd, nrrd
def save_itk(filename, imageArray, origin = None, spacing = None):
    itkimage = sitk.GetImageFromArray(imageArray)
    if origin is not None:
        itkimage.SetOrigin(origin)
    if spacing is not None:
        itkimage.SetSpacing(spacing)
    sitk.WriteImage(itkimage, filename, useCompression = True)

# from mhd to nii.gz
def mhd2nii(mhd_file, nii_file):
    data, origin, spacing = load_itk(mhd_file)
    save_itk(nii_file, data, origin = origin, spacing = spacing)
def nii2mhd(nii_file, mhd_file):
    data, origin, spacing = load_itk(nii_file)
    save_itk(mhd_file, data, origin = origin, spacing = spacing)

def npy2nii(npy_file, nii_file):
    data = np.load(npy_file)
    save_itk(nii_file, data)
def nii2npy(nii_file, npy_file):
    data, _, _ = load_itk(nii_file)
    np.save(npy_file, data)

def nrrd2nii(nrrd_file, nii_file):
    data, origin, spacing = load_itk(nrrd_file)
    save_itk(nii_file, data, origin = origin, spacing = spacing)
def nii2nrrd(nii_file, nrrd_file):
    data, origin, spacing= load_itk(nii_file)
    save_itk(nrrd_file, data, origin = origin, spacing = spacing)

def load_img(input_path):
    img = Image.open(input_path)
    return img

    # img = np.asarray(imageio.imread(input_path))
    # if len(img.shape) == 2:
    #     img = img[...,None]
    # ### ToTensor transform will do the transpose
    # # else:
    # #     # np.transpose is equal to torch.permute
    # #     img = img.transpose(2, 0, 1)
    # return img

def save_img(img_path, img):
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
            if img.shape[2] == 1:
                img = img[:, :, 0]
        img = Image.fromarray(img)
        img.save(img_path)
    elif isinstance(img, mpl_figure.Figure):
        img.savefig(img_path)
def save_npy(npy_path, data):
    np.save(npy_path, data)
def load_npy(npy_path):
    return np.load(npy_path)
def mkdirs(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
def rkdirs(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


def get_coordinates(volume_size, dtype = float):
    dimension = len(volume_size)
    return np.stack(np.meshgrid(*[np.arange(0.0, size).astype(dtype) for size in volume_size], 
                            indexing='ij'), axis=-1).reshape(-1, dimension)

def get_boundingbox(X, background = 0):
    if background is None:
        return np.array([0] * len(X.shape)), np.array(X.shape) - 1
    coor = get_coordinates(X.shape)
    selector = (X.reshape(-1) != background)
    valid_points = coor[selector]
    return np.min(valid_points, axis=0), np.max(valid_points, axis=0)

def get_boundingbox_from_list(Xs = None, boundingboxes = None, background = 0):
    assert Xs is not None or boundingboxes is not None, \
    "At lease one of the Xs or boundingboxes should not be None."
    if boundingboxes is None:
        bbs_min = []
        bbs_max = []
        for X in Xs:
            bb_min, bb_max = get_boundingbox(X, background)
            bbs_min.append(bb_min)
            bbs_max.append(bb_max)
    else:
        bbs_min = [bb[0] for bb in boundingboxes]
        bbs_max = [bb[1] for bb in boundingboxes]
    bbs_min = np.stack(bbs_min)
    bbs_max = np.stack(bbs_max)
    return np.min(bbs_min, axis = 0), np.max(bbs_max, axis = 0)
#### crop non-background area by data
def crop_boundingbox(data = None, margin = (0, 0, 0), background = 0, follows = None, boundingbox = None):
    
    assert data is not None or (boundingbox is not None and follows is not None), \
        "At lease one of the data or boundingboxes should not be None."
    ### use the first follows as data to get start and end idx
    return_data = True
    if data is None:
        data = follows 
        if type(data) == list:
            data = data[0]
        return_data = False
    assert data.ndim == 3 or data.ndim == 4, \
        "Function get_boundingbox() only supports 3D images."
    ### data without channel
    data_without_c = data
    if data.ndim == 4:
        data_without_c = data.sum(axis=0)
    if boundingbox is None:
        start_idx, end_idx = get_boundingbox(data_without_c, background=background)
    else:
        start_idx, end_idx = boundingbox
    start_idx = start_idx - margin
    start_idx[start_idx < 0] = 0
    end_idx = end_idx + margin
    end_idx = np.minimum(end_idx, np.array(data_without_c.shape))
    start_idx = start_idx.astype(int)
    end_idx = end_idx.astype(int)

    if follows is not None:
        is_list = True
        if not type(follows) == list:
            follows = [follows]
            is_list = False
        cropped_follows = []
        for follow in follows:
            assert follow.ndim == 3 or follow.ndim == 4, \
                "Function get_boundingbox() only supports 3D images."

            if follow.ndim == 4:
                cfollow = follow[:, start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
            else:
                cfollow = follow[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
            cropped_follows.append(cfollow)
        if not is_list:
            cropped_follows = cropped_follows[0]
    else:
        cropped_follows = None
    if return_data:
        if data.ndim == 4:
            cropped_data = data[:, start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
        else:
            cropped_data = data[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
        
        return cropped_data, cropped_follows, (start_idx, end_idx)
    
    return cropped_follows, (start_idx, end_idx)

if module_config['point-cloud']:
    from plyfile import PlyData,PlyElement
    ##### load ply file for training
    ## here we only focus on the coordinate information.
    ## later we can add more informations
    def load_ply(inputfile):
        plydata = PlyData.read(inputfile)
        pcd = np.zeros((plydata['vertex']['x'].shape[0], 3 ))
        pcd[:, 0] = plydata['vertex']['x']
        pcd[:, 1] = plydata['vertex']['y']
        pcd[:, 2] = plydata['vertex']['z']
        return pcd

    ### save ply file, with points and colors
    ## here, the face information will be ignored
    ## We can use other packages or cpp program for mesh extraction from point cloud
    def save_ply(filename, points):
        pcolors = None
        if type(points) == tuple:
            points, pcolors = points
        if pcolors is not None:
            if np.max(pcolors) > 1 or np.min(pcolors) < 0:
                pcolors = (pcolors - np.min(pcolors))/ (np.max(pcolors) - np.min(pcolors))
            pcolors = (pcolors * 255).astype(int)
            # not a 3-channel color, we use the first channel as intensity.
            if len(pcolors.shape) < 2 or pcolors.shape[1] < 3:
                pcolors = pcolors.reshape((-1))
                points = [(points[i, 0], points[i, 1], points[i, 2], pcolors[i], pcolors[i], pcolors[i]) for i in range(points.shape[0])]
            else:
                points = [(points[i, 0], points[i, 1], points[i, 2], pcolors[i, 0], pcolors[i, 2], pcolors[i, 2]) for i in range(points.shape[0])]
            vertex = np.array(points, dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1') ])
        else:
            points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
            vertex = np.array(points, dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        el = PlyElement.describe(vertex, 'vertex', comments = ['vertices'])
        PlyData([el], text=True).write(filename)

    def save_xyz(filename, x):
        np.savetxt(filename, x)
    def load_xyz(filename):
        return np.loadtxt(filename)

    def load_pcd(filename):
        basename = os.path.basename(filename)
        suffix = basename.split('.')[-1]
        if suffix == 'ply':
            return load_ply(filename)
        elif suffix == 'xyz' or suffix == 'pts':
            return load_xyz(filename)
        else:
            logger.warning('unsupported file format.')
            raise NotImplementedError
        
    def save_pcd(filename, x):
        basename = os.path.basename(filename)
        suffix = basename.split('.')[-1]
        if suffix == 'ply':
            save_ply(filename, x)
        elif suffix == 'xyz' or suffix == 'pts':
            save_xyz(filename, x)
        else:
            logger.warning('unknow file format, save as ply file.')
            save_ply(filename+'.ply', x)

    ##### generate batch random rotation
    def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Return a tensor where each element has the absolute value taken from the,
        corresponding element of a, with sign taken from the corresponding
        element of b. This is like the standard copysign floating-point operation,
        but is not careful about negative 0 and NaN.

        Args:
            a: source tensor.
            b: tensor whose signs will be used, of the same shape as a.

        Returns:
            Tensor of the same shape as a with the signs of b.
        """
        signs_differ = (a < 0) != (b < 0)
        return torch.where(signs_differ, -a, a)
    def random_quaternions(
        n, dtype = None, device = None
    ) -> torch.Tensor:
        """
        Generate random quaternions representing rotations,
        i.e. versors with nonnegative real part.

        Args:
            n: Number of quaternions in a batch to return.
            dtype: Type to return.
            device: Desired device of returned tensor. Default:
                uses the current device for the default tensor type.

        Returns:
            Quaternions as tensor of shape (N, 4).
        """
        if isinstance(device, str):
            device = torch.device(device)
        o = torch.randn((n, 4), dtype=dtype, device=device)
        s = (o * o).sum(1)
        o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
        return o

    def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Convert rotations given as quaternions to rotation matrices.

        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).

        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def random_rotations(
        n, dtype = None, device = None
    ) -> torch.Tensor:
        """
        Generate random rotations as 3x3 rotation matrices.

        Args:
            n: Number of rotation matrices in a batch to return.
            dtype: Type to return.
            device: Device of returned tensor. Default: if None,
                uses the current device for the default tensor type.

        Returns:
            Rotation matrices as tensor of shape (n, 3, 3).
        """
        quaternions = random_quaternions(n, dtype=dtype, device=device)
        return quaternion_to_matrix(quaternions)

    def batch_transform(x, rotation = None, translation = None):
        ### x: (B, N, 3)
        ### rot: (B, 3, 3)
        ### trans: (B, 3)
        if rotation is not None:
            x = torch.bmm(x, rotation)
        if translation is not None:
            x = x + translation[:, None, :]
        return x

    def batch_random_rotate(x):
        ## x: B * N * 3
        rot_mat = random_rotations(x.shape[0], dtype=x.dtype, device = x.device)
        rot_mat = rot_mat.transpose(1, 2)
        ## batch matrix mulitplation
        return batch_transform(x, rotation=rot_mat)


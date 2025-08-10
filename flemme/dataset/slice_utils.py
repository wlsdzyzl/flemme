### Part of the code is adopted from https://github.com/wolny/pytorch-3dunet 

import numpy as np
from flemme.logger import get_logger

logger = get_logger('slice_builder')
### used to build small slices
class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label ndarray based on the patch and stride shape.

    Args:
        raw_array (ndarray): raw data
        label_array (ndarray): ground truth labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
        kwargs: additional metadata
    """
    def __init__(
            self,
            raw_array,
            label_array,
            patch_shape,
            stride_shape,
            **kwargs
    ):
        patch_shape = tuple(patch_shape)
        stride_shape = tuple(stride_shape)
        skip_shape_check = kwargs.get('skip_shape_check', False)
        if not skip_shape_check:
            self._check_patch_shape(patch_shape)

        self._raw_slices = self._build_slices(raw_array, patch_shape, stride_shape)
        if label_array is None:
            self._label_slices = None
        else:
            # take the first element in the label_array to build slices
            self._label_slices = self._build_slices(label_array, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape
        else:
            i_z, i_y, i_x = dataset.shape

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x),
                    )
                    if dataset.ndim == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k

    @staticmethod
    def _check_patch_shape(patch_shape):
        assert len(patch_shape) == 3, 'patch_shape must be a 3D tuple'
        assert patch_shape[1] >= 32 and patch_shape[2] >= 32, 'Height and Width must be greater or equal 64'


class FilterSliceBuilder(SliceBuilder):
    """
    Filter patches containing less than the `threshold` of non-zero values.

    Args:
        raw_array (ndarray): raw data
        label_array (ndarray): ground truth labels
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
        ignore_index (int): ignore index in the label dataset; this label will be matched to 0 before filtering
        threshold (float): the threshold of non-zero values in the label patch
        slack_acceptance (float): the probability of accepting a patch that does not meet the threshold criteria
        kwargs: additional metadata
    """

    def __init__(
            self,
            raw_array,
            label_array,
            patch_shape,
            stride_shape,
            ignore_index = None,
            threshold = 0.1,
            slack_acceptance = 0.01,
            **kwargs
    ):
        super().__init__(raw_array, label_array, patch_shape, stride_shape, **kwargs)
        if label_array is None:
            return

        rand_state = np.random.RandomState(47)

        def ignore_predicate(raw_label_idx):
            label_idx = raw_label_idx[1]
            patch = label_array[label_idx]
            if ignore_index is not None:
                patch = np.copy(patch)
                patch[patch == ignore_index] = 0
            non_ignore_counts = np.count_nonzero(patch != 0)
            non_ignore_counts = non_ignore_counts / patch.size
            return non_ignore_counts > threshold or rand_state.rand() < slack_acceptance

        zipped_slices = zip(self.raw_slices, self.label_slices)
        # ignore slices containing too much ignore_index
        filtered_slices = list(filter(ignore_predicate, zipped_slices))
        # log number of filtered patches
        logger.info(
            f"Loading {len(filtered_slices)} out of {len(self.raw_slices)} patches: "
            f"{int(100 * len(filtered_slices) / len(self.raw_slices))}%"
        )
        # unzip and save slices
        raw_slices, label_slices = zip(*filtered_slices)
        self._raw_slices = list(raw_slices)
        self._label_slices = list(label_slices)



def get_slice_builder(raw: np.ndarray, label: np.ndarray, config: dict) -> SliceBuilder:
    builder_name = config.get('name', 'SliceBuilder')
    SliceBuilderClass = SliceBuilder
    if builder_name == 'FilterSliceBuilder':
        SliceBuilderClass = FilterSliceBuilder
    elif not builder_name == 'SliceBuilder':
        logger.error(f'Unsupported slice builder: {builder_name}')
        exit(1)
    return SliceBuilderClass(raw, label, **config)

def mirror_pad(image, padding_shape):
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape before it is split into patches.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    assert len(padding_shape) == 3, "Padding shape must be specified for each dimension: ZYX"

    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]

    if image.ndim == 4:
        pad_width = [(0, 0)] + pad_width
    return np.pad(image, pad_width, mode='reflect')

def remove_padding(m, padding_shape):
    """
    Removes padding from the margins of a multi-dimensional array.

    Args:
        m (np.ndarray): The input array to be unpadded.
        padding_shape (tuple of int, optional): The amount of padding to remove from each dimension.
            Assumes the tuple length matches the array dimensions.

    Returns:
        np.ndarray: The unpadded array.
    """
    if padding_shape is None:
        return m

    # Correctly construct slice objects for each dimension in padding_shape and apply them to m.
    return m[(..., *(slice(p, -p or None) for p in padding_shape))]
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

This file has been modified by `wlsdzyzl` for knee singlecoil dataset generation.

Note that it's better to create a new environment for fastMRI, because it may corrupt your environment.
Or you can contact the author of flemme to get the pre-processed dataset.
"""
from argparse import ArgumentParser
from pathlib import Path

import h5py
from tqdm import tqdm

import fastmri
from fastmri.data import transforms
from fastmri.data.subsample import RandomMaskFunc
def save_zero_filled(data_dir, out_dir, which_challenge):
    reconstructions = {}
    mask_func = RandomMaskFunc(center_fractions=[0.08], accelerations=[8])  # Create the mask function object
    for fname in tqdm(list(data_dir.glob("*.h5"))):
        with h5py.File(fname, "r") as hf:
            masked_kspace, _, _ = transforms.apply_mask(transforms.to_tensor(hf["kspace"][()]), mask_func=mask_func)
            crop_size = (
                320, 320
            )
            # inverse Fourier Transform to get zero filled solution
            image = fastmri.ifft2c(masked_kspace)

            # check for FLAIR 203
            if image.shape[-2] < crop_size[1]:
                crop_size = (image.shape[-2], image.shape[-2])

            # crop input image
            image = transforms.complex_center_crop(image, crop_size)

            # absolute value
            image = fastmri.complex_abs(image)

            # apply Root-Sum-of-Squares if multicoil data
            if which_challenge == "multicoil":
                image = fastmri.rss(image, dim=1)

            reconstructions[fname.name] = image

    fastmri.save_reconstructions(reconstructions, out_dir)


def create_arg_parser():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the data",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        required=True,
        help="Path to save the reconstructions to",
    )
    parser.add_argument(
        "--challenge",
        type=str,
        required=True,
        help="Which challenge",
    )

    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    save_zero_filled(args.data_path, args.output_path, args.challenge)

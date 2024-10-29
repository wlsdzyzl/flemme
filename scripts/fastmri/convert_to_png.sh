#!/bin/bash
# python scripts/h5_to_png.py -p path/to/datasets/FastMRI/knee_single/raw --sub_dirs singlecoil_train,singlecoil_train_masked_zero_filled_1 --suffix .h5 --keys reconstruction_esc,reconstruction --output_dir path/to/datasets/FastMRI/knee_single/png/
# python scripts/h5_to_png.py -p path/to/datasets/FastMRI/knee_single/raw --sub_dirs singlecoil_val,singlecoil_val_masked_zero_filled_1 --suffix .h5 --keys reconstruction_esc,reconstruction --output_dir path/to/datasets/FastMRI/knee_single/png/

### extract files
# python scripts/extract_files.py --source_dir  path/to/datasets/FastMRI/knee_single/png/singlecoil_train_masked_zero_filled_1/ --output_dir  path/to/datasets/FastMRI/knee_single/png/fold1/mzf1/ --template_dir  path/to/datasets/FastMRI/knee_single/png/fold1/esc/ --suffix png
# python scripts/extract_files.py --source_dir  path/to/datasets/FastMRI/knee_single/png/singlecoil_train_masked_zero_filled_1/ --output_dir  path/to/datasets/FastMRI/knee_single/png/fold2/mzf1/ --template_dir  path/to/datasets/FastMRI/knee_single/png/fold2/esc/ --suffix png
# python scripts/extract_files.py --source_dir  path/to/datasets/FastMRI/knee_single/png/singlecoil_train_masked_zero_filled_1/ --output_dir  path/to/datasets/FastMRI/knee_single/png/fold3/mzf1/ --template_dir  path/to/datasets/FastMRI/knee_single/png/fold3/esc/ --suffix png
# python scripts/extract_files.py --source_dir  path/to/datasets/FastMRI/knee_single/png/singlecoil_train_masked_zero_filled_1/ --output_dir  path/to/datasets/FastMRI/knee_single/png/fold4/mzf1/ --template_dir  path/to/datasets/FastMRI/knee_single/png/fold4/esc/ --suffix png
# python scripts/extract_files.py --source_dir  path/to/datasets/FastMRI/knee_single/png/singlecoil_train_masked_zero_filled_1/ --output_dir  path/to/datasets/FastMRI/knee_single/png/fold5/mzf1/ --template_dir  path/to/datasets/FastMRI/knee_single/png/fold5/esc/ --suffix png

### scp to server
scp -r path/to/datasets/FastMRI/knee_single/png/fold1/mzf1 guoqingzhang@10.103.69.251:path/to/datasets/fastMRI/knee_single_png/fold1/
scp -r path/to/datasets/FastMRI/knee_single/png/fold2/mzf1 guoqingzhang@10.103.69.251:path/to/datasets/fastMRI/knee_single_png/fold2/
scp -r path/to/datasets/FastMRI/knee_single/png/fold3/mzf1 guoqingzhang@10.103.69.251:path/to/datasets/fastMRI/knee_single_png/fold3/
scp -r path/to/datasets/FastMRI/knee_single/png/fold4/mzf1 guoqingzhang@10.103.69.251:path/to/datasets/fastMRI/knee_single_png/fold4/
scp -r path/to/datasets/FastMRI/knee_single/png/fold5/mzf1 guoqingzhang@10.103.69.251:path/to/datasets/fastMRI/knee_single_png/fold5/

scp -r path/to/datasets/FastMRI/knee_single/png/fold1/mzf1 guoqingzhang@10.103.69.253:path/to/datasets/biomed_2d_dataset/fastMRI/knee_single_png/fold1/
scp -r path/to/datasets/FastMRI/knee_single/png/fold2/mzf1 guoqingzhang@10.103.69.253:path/to/datasets/biomed_2d_dataset/fastMRI/knee_single_png/fold2/
scp -r path/to/datasets/FastMRI/knee_single/png/fold3/mzf1 guoqingzhang@10.103.69.253:path/to/datasets/biomed_2d_dataset/fastMRI/knee_single_png/fold3/
scp -r path/to/datasets/FastMRI/knee_single/png/fold4/mzf1 guoqingzhang@10.103.69.253:path/to/datasets/biomed_2d_dataset/fastMRI/knee_single_png/fold4/
scp -r path/to/datasets/FastMRI/knee_single/png/fold5/mzf1 guoqingzhang@10.103.69.253:path/to/datasets/biomed_2d_dataset/fastMRI/knee_single_png/fold5/
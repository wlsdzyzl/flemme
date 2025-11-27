import numpy as np
import torch
from torch.utils.data import random_split
import sys, getopt
import os
import glob
from flemme.utils import rkdirs, mkdirs
from flemme.dataset.label_dict import get_label_cls
import shutil
from flemme.logger import get_logger
logger = get_logger('scripts.get_split_for_gem3d')
### this script converts sub-folder of medsdf to a single hdf5 file, for point cloud diffusion model training
### comparison experiment

def f(input_file_dict, output_dir, train_ratio=0.9, split_seed=None, followed_folders=[]):
    for cate_id, (train_val_files, test_files) in input_file_dict.items():
        train_files, val_files = random_split(train_val_files, [train_ratio, 1 - train_ratio], 
                                            generator=torch.Generator().manual_seed(split_seed) if split_seed is not None else None)
        split_dir = os.path.join(output_dir, '{:08d}'.format(cate_id), 'splits')
        rkdirs(split_dir)
        with open(os.path.join(split_dir, 'train.lst'), 'w') as f:
            for input_file in train_files:
                f.write(os.path.basename(input_file) + '\n')
        with open(os.path.join(split_dir, 'val.lst'), 'w') as f:    
            for input_file in val_files:
                f.write(os.path.basename(input_file) + '\n')
        with open(os.path.join(split_dir, 'test.lst'), 'w') as f:    
            for input_file in test_files:
                f.write(os.path.basename(input_file) + '\n')


# python pcd2hdf5.py -p /media/wlsdzyzl/DATA/datasets/pcd/MedSDF --train_sub_dirs fold1,fold2,fold3,fold4 --test_sub_dirs fold5 --mesh_dir raw -o ./medsdf.hdf5
def main(argv):
    dataset_path = None
    output_dir = None
    train_sub_dirs = ['.']
    test_sub_dirs = []
    mesh_dir = '.'
    dataset_name = 'medsdf'
    split_seed = 42
    train_ratio = 0.9
    followed_folders = []
    method = shutil.copy
    opts, args = getopt.getopt(argv, "hp:o:", ['help', 'dataset_path=', 'train_sub_dirs=', 'test_sub_dirs=', 'mesh_dir=', \
                                               'dataset_name=', 'split_seed=', 'output_dir=', \
                                               'train_ratio=', 'followed_folders=', 'method='])
    if len(opts) == 0:
        logger.info('unknown options, usage: pcd2hdf5.py -p <dataset_path> --train_sub_dirs <train_sub_dirs=[.]> --test_sub_dirs <test_sub_dirs=[]> -o <output_dir>  --mesh_dir <mesh_dir=.> --dataset_name <dataset_name="medsdf"> --split_seed <split_seed=42> --train_ratio <train_ratio=0.9> --followed_folders <followed_folders=[]> --method <method=copy>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: pcd2hdf5.py -p <dataset_path> -o <output_dir> --train_sub_dirs <train_sub_dirs=[.]> --test_sub_dirs <test_sub_dirs=[]> --mesh_dir <mesh_dir=.> --dataset_name <dataset_name="medsdf"> --split_seed <split_seed=42> --train_ratio <train_ratio=0.9> --followed_folders <followed_folders=[] --method <method=copy>')
            sys.exit()
        elif opt in ('-p', '--dataset_path'):
            dataset_path = arg
        elif opt in ('--train_sub_dirs',):
            train_sub_dirs = arg.split(',')
        elif opt in ('--test_sub_dirs',):
            test_sub_dirs = arg.split(',')
        elif opt in ('--mesh_dir',):
            mesh_dir = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('--dataset_name',):
            dataset_name = arg
        elif opt in ('--split_seed',):
            split_seed = int(arg)
        elif opt in ('--train_ratio',):
            train_ratio = float(arg)
        elif opt in ('--followed_folders',):
            followed_folders = arg.split(',')
        elif opt in ('--method',):
            mn = arg
            if mn == 'move':
                method = shutil.move
            elif not mn == 'copy':
                logger.info('Unknow operation.')
                exit(1)
        else:
            logger.error('unknown option, usage: pcd2hdf5.py -p <dataset_path> -o <output_dir> --train_sub_dirs <train_sub_dirs=[.]> --test_sub_dirs <test_sub_dirs=[]> --mesh_dir <mesh_dir=.> --dataset_name <dataset_name="medsdf"> --split_seed <split_seed=42> --train_ratio <train_ratio=0.9> --followed_folders <followed_folders=[] --method <method=copy>')
            sys.exit()
    assert not dataset_path is None, 'dataset_path is not provided.'
    assert not output_dir is None, 'output_dir is not provided.'
    file_dict = {}
    cate_id_name = get_label_cls(dataset_name)
    logger.info('category id and names: {}'.format(cate_id_name))
    for sd in train_sub_dirs:
        for cate_id, cate_dir in cate_id_name.items():
            pcd_files = sorted(glob.glob(os.path.join(dataset_path + '/' + sd + '/' + mesh_dir + '/' + cate_dir, "*.ply")))
            if cate_id not in file_dict:
                file_dict[cate_id] = [pcd_files, []]
            else:
                file_dict[cate_id][0] = file_dict[cate_id][0] + pcd_files
            for fo in followed_folders:
                followed_files = sorted(glob.glob(os.path.join(dataset_path + '/' + sd + '/' + mesh_dir + '/' + cate_dir + '/' + fo, "*")))
                target_dir = os.path.join(output_dir, '{:08d}'.format(cate_id), fo)
                mkdirs(target_dir)
                for ff in followed_files:
                    logger.info(f'{method.__name__} file {ff} to {os.path.join(target_dir, os.path.basename(ff))}')
                    method(ff, os.path.join(target_dir, os.path.basename(ff))) 
            
    for sd in test_sub_dirs:
        for cate_id, cate_dir in cate_id_name.items():
            pcd_files = sorted(glob.glob(os.path.join(dataset_path + '/' + sd + '/' + mesh_dir + '/' + cate_dir, "*.ply")))
            if cate_id not in file_dict:
                logger.error(f'Category {cate_id} doesn\'t appear in train set.')
                exit(1)
            else:
                file_dict[cate_id][1] = file_dict[cate_id][1] + pcd_files
            for fo in followed_folders:
                followed_files = sorted(glob.glob(os.path.join(dataset_path + '/' + sd + '/' + mesh_dir + '/' + cate_dir + '/' + fo, "*")))
                target_dir = os.path.join(output_dir, '{:08d}'.format(cate_id), fo)
                mkdirs(target_dir)
                for ff in followed_files:
                    method(ff, os.path.join(target_dir, os.path.basename(ff))) 
    f(file_dict, output_dir, 
        split_seed=split_seed, 
        train_ratio=train_ratio, 
        followed_folders=followed_folders)
if __name__ == "__main__":
    main(sys.argv[1:])

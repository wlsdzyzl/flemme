import h5py
import numpy as np
import torch
from torch.utils.data import random_split
import sys, getopt
import os
import glob
from flemme.utils import load_pcd
from flemme.dataset.label_dict import get_label_cls

from flemme.logger import get_logger
logger = get_logger('scripts.pcd2hdf5')
### this script converts sub-folder of medsdf to a single hdf5 file, for point cloud diffusion model training
### comparison experiment

def f(input_file_dict, output_file, sample_num=2048, train_ratio=0.9, split_seed=None,):
    h5file = h5py.File(output_file, 'w')
    for cate_id, (train_val_files, test_files) in input_file_dict.items():
        train_pcds = []
        val_pcds = []
        test_pcds = []
        train_files, val_files = random_split(train_val_files, [train_ratio, 1 - train_ratio], 
                                            generator=torch.Generator().manual_seed(split_seed) if split_seed is not None else None)
        for input_file in train_files:
            pcd = load_pcd(input_file)
            choice = np.random.choice(pcd.shape[0], sample_num, replace=True)
            pcd = pcd[choice]
            train_pcds.append(pcd)
        for input_file in val_files:
            pcd = load_pcd(input_file)
            choice = np.random.choice(pcd.shape[0], sample_num, replace=True)
            pcd = pcd[choice]
            val_pcds.append(pcd)
        for input_file in test_files:
            pcd = load_pcd(input_file)
            choice = np.random.choice(pcd.shape[0], sample_num, replace=True)
            pcd = pcd[choice]
            test_pcds.append(pcd)
        logger.info('create group for category: {}, total samples: {}, train samples: {}, val samples: {}, test samples: {}'.format(
            cate_id, len(train_val_files) + len(test_files), len(train_pcds), len(val_pcds), len(test_pcds)
        ))
        cate_g = h5file.create_group('{:08d}'.format(cate_id))
        cate_g.create_dataset('train', data=np.array(train_pcds), dtype='f')
        cate_g.create_dataset('val', data=np.array(val_pcds), dtype='f')
        cate_g.create_dataset('test', data=np.array(test_pcds), dtype='f')
    h5file.close()
# python pcd2hdf5.py -p /media/wlsdzyzl/DATA/datasets/pcd/MedSDF --train_sub_dirs fold1,fold2,fold3,fold4 --test_sub_dirs fold5 --pcd_dir raw -o ./medsdf.hdf5 --sample_num 2560
def main(argv):
    dataset_path = None
    output_file = None
    train_sub_dirs = ['.']
    test_sub_dirs = []
    pcd_dir = '.'
    sample_num = 2048
    dataset_name = 'medsdf'
    split_seed = 42
    train_ratio = 0.9
    opts, args = getopt.getopt(argv, "hp:o:", ['help', 'dataset_path=', 'train_sub_dirs=', 'test_sub_dirs=', 'pcd_dir=', \
                                               'dataset_name=', 'split_seed=', 'output_file=', 'sample_num=', \
                                               'train_ratio='])
    if len(opts) == 0:
        logger.info('unknown options, usage: pcd2hdf5.py -p <dataset_path> --train_sub_dirs <train_sub_dirs=[.]> --test_sub_dirs <test_sub_dirs=[]> -o <output_file>  --pcd_dir <pcd_dir=.> --dataset_name <dataset_name="medsdf"> --split_seed <split_seed=42> --train_ratio <train_ratio=0.9> --sample_num <sample_num = 2048> ')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: pcd2hdf5.py -p <dataset_path> -o <output_file> --train_sub_dirs <train_sub_dirs=[.]> --test_sub_dirs <test_sub_dirs=[]> --pcd_dir <pcd_dir=.> --dataset_name <dataset_name="medsdf"> --split_seed <split_seed=42> --train_ratio <train_ratio=0.9> --sample_num <sample_num = 2048>')
            sys.exit()
        elif opt in ('-p', '--dataset_path'):
            dataset_path = arg
        elif opt in ('--train_sub_dirs',):
            train_sub_dirs = arg.split(',')
        elif opt in ('--test_sub_dirs',):
            test_sub_dirs = arg.split(',')
        elif opt in ('--pcd_dir',):
            pcd_dir = arg
        elif opt in ("-o", '--output_file'):
            output_file = arg
        elif opt in ('--dataset_name',):
            dataset_name = arg
        elif opt in ('--sample_num',):
            sample_num = int(arg)
        elif opt in ('--split_seed',):
            split_seed = int(arg)
        elif opt in ('--train_ratio',):
            train_ratio = float(arg)
        else:
            logger.error('unknown option, usage: pcd2hdf5.py -p <dataset_path> -o <output_file> --train_sub_dirs <train_sub_dirs=[.]> --test_sub_dirs <test_sub_dirs=[]> --pcd_dir <pcd_dir=.> --dataset_name <dataset_name="medsdf"> --split_seed <split_seed=42> --train_ratio <train_ratio=0.9> --sample_num <sample_num = 2048> ')
            sys.exit()
    assert not dataset_path is None, 'dataset_path is not provided.'
    assert not output_file is None, 'output_file is not provided.'
    file_dict = {}
    cate_id_name = get_label_cls(dataset_name)
    logger.info('category id and names: {}'.format(cate_id_name))
    for sd in train_sub_dirs:
        for cate_id, cate_dir in cate_id_name.items():
            pcd_files = sorted(glob.glob(os.path.join(dataset_path + '/' + sd + '/' + pcd_dir + '/' + cate_dir, "*.ply")))
            if cate_id not in file_dict:
                file_dict[cate_id] = [pcd_files, []]
            else:
                file_dict[cate_id][0] = file_dict[cate_id][0] + pcd_files
    for sd in test_sub_dirs:
        for cate_id, cate_dir in cate_id_name.items():
            pcd_files = sorted(glob.glob(os.path.join(dataset_path + '/' + sd + '/' + pcd_dir + '/' + cate_dir, "*.ply")))
            if cate_id not in file_dict:
                logger.error(f'Category {cate_id} doesn\'t appear in train set.')
                exit(1)
            else:
                file_dict[cate_id][1] = file_dict[cate_id][1] + pcd_files
    f(file_dict, output_file, sample_num=sample_num, 
        split_seed=split_seed, train_ratio=train_ratio)
if __name__ == "__main__":
    main(sys.argv[1:])

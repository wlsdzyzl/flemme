import numpy as np
import torch
from torch.utils.data import random_split
import sys, getopt
import os
import glob
from flemme.utils import load_pcd, save_ply
from flemme.dataset.label_dict import get_label_cls
from flemme.logger import get_logger
logger = get_logger('scripts.npy2ply_recon')
### this script converts sub-folder of medsdf to a single hdf5 file, for point cloud diffusion model training
### comparison experiment

def f(input_file_dict, result_file, output_dir):
    input_id = 0
    input_array = np.load(result_file)
    num_test_files = sum([ len(v) for v in input_file_dict.values()])
    assert len(input_array) == num_test_files, \
        'input array length {} doesn\'t match input file dict length {}'.format(input_array.shape, num_test_files)
    for cate_dir, test_files in input_file_dict.items():
        logger.info('processing category: {}, number of files: {}'.format(cate_dir, len(test_files)))
        for test_file in test_files:
            pcd = input_array[input_id]
            input_id += 1
            save_path = os.path.join(output_dir, cate_dir)
            os.makedirs(save_path, exist_ok=True)
            save_ply(os.path.join(save_path, os.path.basename(test_file)), pcd)
def main(argv):
    dataset_path = None
    output_dir = None
    result_file = None
    test_sub_dirs = []
    pcd_dir = '.'
    dataset_name = 'medsdf'
    category = 'all'
    opts, args = getopt.getopt(argv, "hp:o:", ['help', 'dataset_path=', 'test_sub_dirs=', 'result_file=', 'pcd_dir=', \
                                               'dataset_name=', 'output_dir=', 'category=',])
    if len(opts) == 0:
        logger.info('unknown options, usage: npy2ply_recon.py -p <dataset_path> --test_sub_dirs <test_sub_dirs=[]> --result_file <result_file> -o <output_dir>  --pcd_dir <pcd_dir=.> --dataset_name <dataset_name="medsdf"> --category <category=all>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: npy2ply_recon.py -p <dataset_path> -o <output_dir> --test_sub_dirs <test_sub_dirs=[]> --result_file <result_file> --pcd_dir <pcd_dir=.> --dataset_name <dataset_name="medsdf"> --category <category=all>')
            sys.exit()
        elif opt in ('-p', '--dataset_path'):
            dataset_path = arg
        elif opt in ('--test_sub_dirs',):
            test_sub_dirs = arg.split(',')
        elif opt in ('--pcd_dir',):
            pcd_dir = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('--dataset_name',):
            dataset_name = arg
        elif opt in ('--category',):
            category = arg
        elif opt in ('--result_file',):
            result_file = arg
        else:
            logger.error('unknown option, usage: npy2ply_recon.py -p <dataset_path> -o <output_dir> --test_sub_dirs <test_sub_dirs=[]> --result_file <result_file> --pcd_dir <pcd_dir=.> --dataset_name <dataset_name="medsdf"> --category <category=all>')
            sys.exit()

    assert not dataset_path is None, 'dataset_path is not provided.'
    assert not output_dir is None, 'output_dir is not provided.'
    assert not result_file is None, 'result_file is not provided.'
    file_dict = {}
    cate_id_name = get_label_cls(dataset_name)
    logger.info('category id and names: {}'.format(cate_id_name))

    for sd in test_sub_dirs:
        for _, cate_dir in cate_id_name.items():
            if category == 'all' or category == cate_dir:
                pcd_files = sorted(glob.glob(os.path.join(dataset_path + '/' + sd + '/' + pcd_dir + '/' + cate_dir, "*.ply")))
                if cate_dir not in file_dict:
                    file_dict[cate_dir] = []
                file_dict[cate_dir] += pcd_files
    f(file_dict, result_file=result_file, output_dir=output_dir)
if __name__ == "__main__":
    main(sys.argv[1:])

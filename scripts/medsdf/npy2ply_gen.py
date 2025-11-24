import numpy as np
import torch
from torch.utils.data import random_split
import sys, getopt
import os
import glob
from flemme.utils import load_pcd, save_ply, rkdirs
from flemme.dataset.label_dict import get_label_cls
from flemme.logger import get_logger
logger = get_logger('scripts.npy2ply_gen')
### this script converts sub-folder of medsdf to a single hdf5 file, for point cloud diffusion model training
### comparison experiment

def f(result_file, output_dir):
    input_array = np.load(result_file)
    logger.info(f'Saving ply to {output_dir}, number of files: {len(input_array)}.')
    for iid, pcd in enumerate(input_array):
        save_ply(os.path.join(output_dir, f'gen_rand_{iid}.ply'), pcd)
def main(argv):
    output_dir = None
    result_file = None
    opts, args = getopt.getopt(argv, "ho:", ['help', 'result_file='])
    if len(opts) == 0:
        logger.info('unknown options, usage: npy2ply_gen.py --result_file <result_file> -o <output_dir>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: npy2ply_gen.py --result_file <result_file> -o <output_dir>')
            sys.exit()
        elif opt in ('--result_file',):
            result_file = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        else:
            logger.error('unknown option, usage: npy2ply_gen.py --result_file <result_file> -o <output_dir>')
            sys.exit()
    assert not output_dir is None, 'output_dir is not provided.'
    assert not result_file is None, 'result_file is not provided.'
    rkdirs(output_dir)
    f(result_file=result_file, output_dir=output_dir)
if __name__ == "__main__":
    main(sys.argv[1:])

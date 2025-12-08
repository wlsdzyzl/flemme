from flemme.utils import *
import sys, getopt
import os
import glob
from flemme.logger import get_logger
import open3d as o3d
from tqdm import tqdm
logger = get_logger('scripts.pcd2ply')

def main(argv):
    input_dir = ''
    output_dir = ''
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input_dir=', 'output_dir='])

    if len(opts) == 0:
        logger.error('unknow options, usage: pcd2ply.py -i <input_dir> -o <output_dir>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: pcd2ply.py -i <input_dir> -o <output_dir>')
            sys.exit()
        elif opt in ("-i", '--input_dir'):
            input_dir = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        else:
            logger.error('unknow option, usage: pcd2ply.py -i <input_dir> -o <output_dir>')
            sys.exit()
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.pcd")))
    mkdirs(output_dir)
    for ifile in tqdm(input_files, 'transfering'):
        pcd = o3d.io.read_point_cloud(ifile)
        ofile = os.path.join(output_dir, os.path.basename(ifile)[:-4]+'.ply')
        o3d.io.write_point_cloud(ofile, pcd)
if __name__ == "__main__":
    main(sys.argv[1:])
from flemme.utils import *
import sys, getopt
import os
import glob
from flemme.logger import get_logger
import open3d as o3d
from tqdm import tqdm

logger = get_logger('scripts.pcd_transfer')

def main(argv):
    input_dir = ''
    output_dir = ''
    input_suffix = '.pcd'
    output_suffix = '.ply'
    key = None
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input_dir=', 'output_dir=', 'input_suffix=', 'key='])

    if len(opts) == 0:
        logger.error('unknow options, usage: pcd_transfer.py -i <input_dir> -o <output_dir> --input_suffix <input_suffix=.ply> --output_suffix <output_suffix=.ply> --key <key=None>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: pcd_transfer.py -i <input_dir> -o <output_dir> --input_suffix <input_suffix=.ply> --output_suffix <output_suffix=.ply> --key <key=None>')
            sys.exit()
        elif opt in ("-i", '--input_dir'):
            input_dir = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('--input_suffix', ):
            input_suffix = arg
        elif opt in ('--output_suffix', ):
            output_suffix = arg
        elif opt in ('--key', ):
            key = arg
        else:
            logger.error('unknow option, usage: pcd_transfer.py -i <input_dir> -o <output_dir> --input_suffix <input_suffix=.ply> --output_suffix <output_suffix=.ply> --key <key=None>')
            sys.exit()
    input_files = sorted(glob.glob(os.path.join(input_dir, f"*{input_suffix}")))
    mkdirs(output_dir)
    for ifile in tqdm(input_files, 'transfering'):
        if ifile.lower().endswith('.pcd'):
            pcd = o3d.io.read_point_cloud(ifile)
        elif ifile.lower().endswith('.npz'):
            pcd = np.load(ifile)[key]
        else:
            pcd = load_pcd(ifile)
        if pcd.ndim == 3:
            pcd = pcd.squeeze()
        ofile = os.path.join(output_dir, os.path.basename(ifile).replace(input_suffix, output_suffix))
        save_pcd(ofile, pcd)
if __name__ == "__main__":
    main(sys.argv[1:])
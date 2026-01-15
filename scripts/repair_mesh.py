from flemme.utils import *
import sys, getopt
import os
import glob
from flemme.logger import get_logger
import open3d as o3d
from tqdm import tqdm
import trimesh

logger = get_logger('scripts.repair_mesh')

def main(argv):
    input_dir = ''
    output_dir = None
    suffix = '.ply'
    opts, args = getopt.getopt(argv, "hi:o:r", ['help', 'input_dir=', 'output_dir=', 'suffix=','smoothing', 'recursive'])
    smoothing = False
    recursive = False
    if len(opts) == 0:
        logger.error('unknow options, usage: repair_mesh.py -i <input_dir> -o <output_dir> --suffix <suffix=.ply> --smoothing --recursive')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: repair_mesh.py -i <input_dir> -o <output_dir> --suffix <suffix=.ply> --smoothing --recursive')
            sys.exit()
        elif opt in ("-i", '--input_dir'):
            input_dir = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('--suffix', ):
            suffix = arg
        elif opt in ('--smoothing',):
            smoothing = True
        elif opt in ('-r', '--recursive'):
            recursive = True
        else:
            logger.error('unknow option, usage: repair_mesh.py -i <input_dir> -o <output_dir> --suffix <suffix=.ply> --smoothing --recursive')
            sys.exit()
    if not output_dir:
        output_dir = input_dir
    mkdirs(output_dir)
    if recursive:
        input_files = sorted(glob.glob(os.path.join(input_dir, f"*/*{suffix}"), recursive = True))
    else:
        input_files = sorted(glob.glob(os.path.join(input_dir, f"*{suffix}")))
    for ifile in tqdm(input_files, 'fixing'):
        mesh = load_mesh(ifile, clean=True, repair=True)
        mesh = remove_small_holes(mesh, 100)
        mesh = remove_small_components(mesh, 50)
        if smoothing:
            trimesh.smoothing.filter_humphrey(
                mesh,
                alpha=0.1,
                beta=0.5,
                iterations=10
            )
        ofile = ifile.replace(input_dir, output_dir)
        if recursive:
            ofile_dir = os.path.dirname(ofile)
            mkdirs(ofile_dir)
        save_ply(ofile, mesh.vertices, faces = mesh.faces)
if __name__ == "__main__":
    main(sys.argv[1:])
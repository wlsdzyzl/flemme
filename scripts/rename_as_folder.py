import sys, getopt
import os
import shutil
from flemme.logger import get_logger
from flemme.utils import get_file_extension, mkdirs
from tqdm import tqdm
logger = get_logger('scripts.rename_as_folder')
def process_files(src, filename, tar):
    for r, _, files in os.walk(src):
        for f in files:
            if filename in f:
                rel_path = os.path.relpath(r, src)
                rel_path = rel_path.rsplit('/', 1)[0]
                target_dir = os.path.join(tar, rel_path)
                mkdirs(target_dir)
                
                suffix = get_file_extension(f, False)
                file_path = os.path.join(r, f)
                target_basename = r.rsplit('/', 1)[-1]+'.'+suffix
                target_path = os.path.join(target_dir, target_basename)
                logger.info('copy from {} to {}'.format(file_path, target_path))
                shutil.copy(file_path, target_path)
## rename files using their parent folder's name, for example: rename ShapeNet/XXXX/123/model.binvox to ShapeNet/XXXX/123.binvox
## only file with {filename} are processed.
## python rename_as_folder.py -i /data/guoqingzhang/datasets/ShapeNet/ShapeNet/ -o /data/guoqingzhang/datasets/ShapeNet/ShapeSDF/ -f model.binvox
def main(argv):
    input_dir = None
    output_dir = None
    filename = None

    opts, args = getopt.getopt(argv, "hi:o:f:", ['help', 'input_dir', 'output_dir', 'filename'])
    if len(opts) == 0:
        logger.error('unknow options, usage: rename_as_folder.py -i <input_dir> -o <output_dir> -f <filename>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: rename_as_folder.py -i <input_dir> -o <output_dir> -f <filename> ')
            sys.exit()
        elif opt in ("-i", '--input'):
            input_dir = arg
        elif opt in ("-o", '--output'):
            output_dir = arg
        elif opt in ("-f", '--filename'):
            filename = arg
        else:
            logger.error('unknow option, usage: rename_as_folder.py -i <input_dir> -o <output_dir> -f <filename>')
            sys.exit()
    assert input_dir is not None, 'input dir is not provided.'
    assert output_dir is not None, 'output dir is not provided.'
    assert filename is not None, 'filename is not provided.'
    process_files(input_dir, filename, output_dir)
                
if __name__ == "__main__":
    main(sys.argv[1:])
from flemme.utils import *
import sys, getopt
import os
import glob
from flemme.logger import get_logger
from tqdm import tqdm

logger = get_logger('scripts.process_img')
## python process_img.py -i /data/guoqingzhang/flemme-results/gen/imageCAS_mip2d/edm_unet_atten/ -o /data/guoqingzhang/flemme-results/gen/imageCAS_mip2d/edm_unet_atten_336/ --target_shape 336,336
def main(argv):
    input_dir = ''
    output_dir = ''
    input_suffix = '.png'
    output_suffix = None
    target_shape = None
    scaling = None
    order = 0
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input_dir=', 'output_dir=', 
                                    'input_suffix=', 'output_suffix=',
                                    'target_shape=', 'scaling=',
                                    'order='])

    if len(opts) == 0:
        logger.error('unknow options, usage: process_img.py -i <input_dir> --input_suffix <input_suffix=.png> -o <output_dir> --output_suffix <output_suffix=.png> --target_shape <target_shape=None> --scaling <scaling=None> --order <order=0>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: process_img.py -i <input_dir> --input_suffix <input_suffix=.png> -o <output_dir> --output_suffix <output_suffix=.png> --target_shape <target_shape=None> --scaling <scaling=None> --order <order=0>')
            sys.exit()
        elif opt in ("-i", '--input_dir'):
            input_dir = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('--input_suffix',):
            input_suffix = arg
        elif opt in ('--output_suffix',):
            output_suffix = arg
        elif opt in ('--target_shape',):
            target_shape = [int(s) for s in arg.split(',')]
        elif opt in ('--scaling',):
            scaling = [float(s) for s in arg.split(',')]
            if len(scaling) == 1:
                scaling = scaling[0]
        elif opt in ('--order',):
            order = int(arg)
        else:
            logger.error('unknow option, usage: process_img.py -i <input_dir> --input_suffix <input_suffix=.png> -o <output_dir> --output_suffix <output_suffix=.png> --target_shape <target_shape=None> --scaling <scaling=None> --order <order=0>')
            sys.exit()
    
    output_suffix = output_suffix or input_suffix
    input_files = sorted(glob.glob(os.path.join(input_dir, f"*{input_suffix}")))
    mkdirs(output_dir)
    for ifile in tqdm(input_files, 'processing'):
        img_np = load_img_as_numpy(ifile)
        if scaling or target_shape:
            img_np = zoom(img_np, scaling = scaling, 
                target_shape = target_shape, 
                order = order).astype(np.uint8)
        ofile = os.path.join(output_dir, os.path.basename(ifile).replace(input_suffix, output_suffix))
        save_img(ofile, img_np)
if __name__ == "__main__":
    main(sys.argv[1:])
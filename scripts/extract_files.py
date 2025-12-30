### extract files from a directory to target directory based on a template dir
import sys, getopt
import os
import shutil
from glob import glob
from flemme.logger import get_logger
from flemme.utils import rkdirs, is_in_one_of, rreplace
logger = get_logger('scripts.extract_files')
## python ./extract_files.py --source_dir /data/guoqingzhang/datasets/MedSDF/split/fold5/rmesh022/ --template_dir /data/guoqingzhang/vcg-for-figure/recon/ours/ --output_dir /data/guoqingzhang/vcg-for-figure/recon/GT -r --suffix _mesh.ply,.ply,.ply
def main(argv):
    source_dir = None
    template_dir = None
    output_dir = None
    suffix = ['']
    opts, _ = getopt.getopt(argv, "hrt:o:", ['help', 'source_dir=', 'template_dir=', 'output_dir=', 'suffix=', 'method=', 'recursive'])
    method = shutil.copy
    mn = 'copy'
    R = False
    ### move is faster, but with higher risk for losing data.
    if len(opts) == 0:
        logger.info('unknow options, usage: extract_files.py --source_dir <source_dir> --template_dir <template_dir=.> --output_dir <output_dir=> --suffix <suffix=\'\',\'\',\'\'> --method <method=copy> --recursive')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: extract_files.py --source_dir <source_dir> --template_dir <template_dir=.> --output_dir <output_dir=> <suffix=\'\',\'\',\'\'> --method <method=copy> --recursive')
            sys.exit()
        if opt in ('--source_dir',):
            source_dir = arg
        elif opt in ('-t','--template_dir'):
            template_dir = arg
        elif opt in ('-o', '--output_dir'):
            output_dir = arg
        elif opt in ('--suffix', ):
            suffix = arg.split(',')
        elif opt in ('--method',):
            mn = arg
            if mn == 'move':
                method = shutil.move
            elif not mn == 'copy':
                logger.info('Unknow operation.')
                exit(1)
        elif opt in ('-r', '--recursive'):
            R = True
        else:
            logger.info('unknow options, usage: extract_files.py --source_dir <source_dir> --template_dir <template_dir=.> --output_dir <output_dir=> <suffix=\'\',\'\',\'\'> --method <method=copy> --recursive')
            sys.exit()
    if source_dir is None:
        logger.error('source_dir is required.')
        sys.exit()
    if output_dir is None:
        logger.error('output_dir is required.')
        sys.exit()
    if template_dir is None:
        logger.error('template_dir is required.')
        sys.exit()
    assert not source_dir == template_dir and not source_dir == output_dir and not template_dir == output_dir,\
        "source dir, template dir and output dir shouldn't be the same."
    if len(suffix) == 1:
        suffix = suffix * 3
    rkdirs(output_dir)
    template_files = sorted(glob(os.path.join(template_dir, '*'+suffix[1])))
    
    if not R:
        for tf in template_files:
            sf = rreplace(tf.replace(template_dir, source_dir), suffix[1], suffix[0], 1)
            of = rreplace(tf.replace(template_dir, output_dir), suffix[1], suffix[2], 1)
            logger.info('{} from {} to {}'.format(mn, sf, of))
            method(sf, of)
    else:
        source_files = sorted(glob(os.path.join(source_dir, '**/*'+suffix[0]), recursive = True))
        for sf in source_files:
            basename = rreplace(os.path.basename(sf), suffix[0], suffix[1], 1)
            if is_in_one_of(basename, template_files):
                of = rreplace(os.path.join(output_dir, basename), suffix[1], suffix[2], 1)
                logger.info('{} from {} to {}'.format(mn, sf, of))
                method(sf, of)

if __name__ == "__main__":
    main(sys.argv[1:])
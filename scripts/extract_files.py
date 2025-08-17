### extract files from a directory to target directory based on a template dir
import sys, getopt
import os
import shutil
from glob import glob
from flemme.logger import get_logger
from flemme.utils import rkdirs
logger = get_logger('scripts.extract_files')
def main(argv):
    source_dir = None
    template_dir = None
    output_dir = None
    suffix = ''
    opts, _ = getopt.getopt(argv, "ht:o:", ['help', 'source_dir=', 'template_dir=', 'output_dir=', 'suffix=', 'method='])
    method = shutil.copy
    mn = 'copy'
    ### move is faster, but with higher risk for losing data.
    if len(opts) == 0:
        logger.info('unknow options, usage: extract_files.py --source_dir <source_dir> --template_dir <template_dir=.> --output_dir <output_dir=> --suffix <suffix=\'\'> --method <method=copy>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: extract_files.py --source_dir <source_dir> --template_dir <template_dir=.> --output_dir <output_dir=> <suffix=\'\'> --method <method=copy>')
            sys.exit()
        if opt in ('--source_dir',):
            source_dir = arg
        elif opt in ('-t','--template_dir'):
            template_dir = arg
        elif opt in ('-o', '--output_dir'):
            output_dir = arg
        elif opt in ('--suffix', ):
            suffix = arg
        elif opt in ('--method',):
            mn = arg
            if mn == 'move':
                method = shutil.move
            elif not mn == 'copy':
                logger.info('Unknow operation.')
                exit(1)
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
    rkdirs(output_dir)
    template_files = sorted(glob(os.path.join(template_dir, '*'+suffix)))
    for tf in template_files:
        sf = tf.replace(template_dir, source_dir+'/')
        of = tf.replace(template_dir, output_dir+'/')
        logger.info('{} from {} to {}'.format(mn, sf, of))
        method(sf, of)

if __name__ == "__main__":
    main(sys.argv[1:])
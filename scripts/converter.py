from flemme.utils import mhd2nii
import sys, getopt
import os
import glob
from flemme.logger import get_logger
logger = get_logger('scripts.converter')
def main(argv):
    input_file = ''
    output_file = ''
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input', 'output'])
    if len(opts) == 0:
        logger.error('unknow options, usage: converter.py -i <input_file> -o <output_file>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: converter.py -i <input_file> -o <output_file> ')
            sys.exit()
        elif opt in ("-i", '--input'):
            input_file = arg
        elif opt in ("-o", '--output'):
            output_file = arg
        else:
            logger.error('unknow option, usage: converter.py -i <input_file> -o <output_file>')
            sys.exit()

    if os.path.isdir(input_file):
        if not os.path.isdir(output_file):
            os.makedirs(output_file)
        input_files = sorted(glob.glob(os.path.join(input_file, '*.mhd')))
        for ifile in input_files:
            _, filename = os.path.split(ifile)
            filename, _ = os.path.splitext(filename)
            # filename, _ = os.path.splitext(filename)
            filename = filename.split('_')[0]
            ofile = output_file+'/'+ filename + '_g6.nii.gz'
            logger.info(f'Converting from {ifile} to {ofile} ...')
            mhd2nii(ifile, ofile)
    else:
        mhd2nii(input_file, output_file)
if __name__ == "__main__":
    main(sys.argv[1:])
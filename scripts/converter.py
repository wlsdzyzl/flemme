from flemme.utils import mhd2nii
import sys, getopt
import os
import glob

def main(argv):
    inputfile = ''
    outputfile = ''
    opts, args = getopt.getopt(argv, "hi:o:", ['help', 'input', 'output'])
    if len(opts) == 0:
        print('unknow options, usage: converter.py -i <inputfile> -o <outputfile>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: converter.py -i <inputfile> -o <outputfile> ')
            sys.exit()
        elif opt in ("-i", '--input'):
            inputfile = arg
        elif opt in ("-o", '--output'):
            outputfile = arg
        else:
            print('unknow option, usage: converter.py -i <inputfile> -o <outputfile>')
            sys.exit()

    if os.path.isdir(inputfile):
        if not os.path.isdir(outputfile):
            os.makedirs(outputfile)
        input_files = sorted(glob.glob(os.path.join(inputfile, '*.mhd')))
        for ifile in input_files:
            _, filename = os.path.split(ifile)
            filename, _ = os.path.splitext(filename)
            # filename, _ = os.path.splitext(filename)
            filename = filename.split('_')[0]
            ofile = outputfile+'/'+ filename + '_g6.nii.gz'
            print(f'Converting from {ifile} to {ofile} ...')
            mhd2nii(ifile, ofile)
    else:
        mhd2nii(inputfile, outputfile)
if __name__ == "__main__":
    main(sys.argv[1:])
from flemme.utils import *
import numpy as np
import sys, getopt
import os
import glob
import shutil
from scipy.ndimage import measurements as mea, binary_erosion, binary_dilation
from skimage.morphology import skeletonize as skeletonize
from scipy.spatial import cKDTree

def f(inputfile, mask_file = None, surface_file = None, skeleton_file = None, xyzr_file = None, normalized = True, prob_threshold = None, label_value = None):
    if (inputfile[-3:] == 'npy'):
        mask_array = load_npy(inputfile)
    else:
        mask_array, _, _ = load_itk(inputfile)
    if prob_threshold is not None:
        mask_array >= prob_threshold
    else:
        mask_array = (mask_array == label_value)

    assert label_value is not None or prob_threshold is not None, "At least one of [prob_threshold, label_value] should not be None."
    mask_array = binary_erosion(binary_dilation(mask_array, iterations=4), iterations=4)


    vol_coor = get_coordinates(mask_array.shape)
    # normalize to [-1, 1]
    if normalized:
        vol_coor = vol_coor / np.array(mask_array.shape).astype(float) * 2 - 1.0
    selector = (mask_array).reshape((-1))

    mask = vol_coor[selector]

    ## save mask
    if mask_file is not None:
        save_ply(mask_file, mask)        

        print('extract {} points for mask, save to {}'.format(len(mask), mask_file))
    ## save surface
    if surface_file is not None:
        sur_array = np.logical_xor(mask_array, binary_erosion(mask_array))
        
        selector = sur_array.reshape((-1))
        
        sur = vol_coor[selector]

        save_ply(surface_file, sur)
    

        print('extract {} points for surface, save to {}'.format(len(sur), surface_file))  
    ## save skeleton
    if skeleton_file is not None:
        ske_array = skeletonize(mask_array)
        # print('finish skeletonization')
        selector = (ske_array >= 0.5).reshape((-1))
        ske = vol_coor[selector]
        # print(len(ske))
        save_ply(skeleton_file, ske)
        print('extract {} points for skeleton, save to {}'.format(len(ske), skeleton_file))  
    if xyzr_file is not None and surface_file is not None and skeleton_file is not None:
        # find radius of left skeleton
        tree = cKDTree(sur)
        dist, _ = tree.query(ske, k = 4)
        # compute the average distance to the nearest 4 points
        radius = np.mean(dist, axis = 1)
        # save radius to npy file
        xyzr = np.hstack((ske, radius[:, None]))
        np.savetxt(xyzr_file, xyzr)          
def main(argv):
    inputfile = ''
    outputdir = ''
    opts, args = getopt.getopt(argv, "hi:o:p:v:", ['help', 'input=', 'input_suffix=', 'output=', 'prob_threshold=', 'label_value=' , 'mask', 'surface','skeleton', 'xyzr', 'no_normalized'])
    mask = False
    surface = False
    skeleton = False
    xyzr = False
    normalized = True
    prob_threshold = None
    label_value = None
    suffix = 'nii.gz'
    if len(opts) == 0:
        print('unknow options, usage: mask2xyzr.py -i <inputfile> --input_suffix <input_suffix=nii.gz> -o <outputdir> -p <prob_threshold = None> -v <label_value = None> --mask --surface --skeleton --xyzr --no_normalized')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('usage: mask2xyzr.py -i <inputfile> --input_suffix <input_suffix=nii.gz> -o <outputdir> -p <prob_threshold = None> -v <label_value = None> --mask --surface --skeleton --xyzr --no_normalized')
            sys.exit()
        elif opt in ("-i", '--input'):
            inputfile = arg
        elif opt in ('--input_suffix'):
            suffix = arg
        elif opt in ("-o", '--output'):
            outputdir = arg
        elif opt in ("-p", '--prob_threshold'):
            prob_threshold = float(arg)
        elif opt in ('-v', '--label_value'):
            label_value = int(arg)
        elif opt in ('--mask',):
            mask = True
        elif opt in ('--surface',):
            surface = True
        elif opt in ('--skeleton',):
            skeleton = True
        elif opt in ('--xyzr',):
            xyzr = True
        elif opt in ('--no_normalized',):
            normalized = False
        else:
            print('unknow option, usage: mask2xyzr.py -i <inputfile> --input_suffix <input_suffix=nii.gz> -o <outputdir> -p <prob_threshold = None> -v <label_value = None> --mask --surface --skeleton --xyzr --no_normalized')
            sys.exit()

    input_files = []
    filenames = []
    if os.path.isdir(outputdir):
        shutil.rmtree(outputdir)
    if os.path.isdir(inputfile):
        input_files = sorted(glob.glob(os.path.join(inputfile, "*" + suffix)))
        filenames = [os.path.splitext(os.path.splitext(os.path.split(ifile)[1])[0])[0] for ifile in input_files]
    else:
        input_files = [inputfile]
        filenames = [os.path.splitext(os.path.splitext(os.path.split(inputfile)[1])[0])[0]]

    mask_file_list = [None for _ in range(len(input_files))]
    surface_file_list = [None for _ in range(len(input_files))]
    skeleton_file_list = [None for _ in range(len(input_files))]
    xyzr_file_list = [None for _ in range(len(input_files))]
    if mask:
        os.makedirs(outputdir+'/mask')
        mask_file_list = [ outputdir+'/mask/'+ filename + '.ply' for filename in filenames]
    if surface:
        os.makedirs(outputdir+'/surface')
        surface_file_list = [ outputdir+'/surface/'+ filename + '.ply' for filename in filenames]
    if skeleton:
        os.makedirs(outputdir+'/skeleton')
        skeleton_file_list = [ outputdir+'/skeleton/'+ filename + '.ply' for filename in filenames]
    if xyzr:
        os.makedirs(outputdir+'/xyzr')
        xyzr_file_list = [ outputdir+'/xyzr/'+ filename + '.xyzr' for filename in filenames]

    for ifile, mask_file, surface_file, skeleton_file, xyzr_file in zip(input_files, mask_file_list, surface_file_list, skeleton_file_list, xyzr_file_list):
        f(ifile, mask_file, surface_file, skeleton_file, xyzr_file, normalized, prob_threshold = prob_threshold, label_value = label_value)
if __name__ == "__main__":
    main(sys.argv[1:])
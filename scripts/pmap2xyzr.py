from flemme.utils import *
import numpy as np
import sys, getopt
import os
import glob
from scipy.ndimage import binary_erosion, binary_dilation, label
from skimage.morphology import skeletonize as skeletonize
from scipy.spatial import cKDTree
from flemme.logger import get_logger
logger = get_logger('scripts.pmap2xyzr')
def f(input_file, mask_file = None, surface_file = None, skeleton_file = None, xyzr_file = None, normalized = True, prob_threshold = None, 
        label_value = None, cc_axis = None, dilate = False):
    if (input_file[-3:] == 'npy'):
        mask_array = load_npy(input_file)
    else:
        mask_array = load_itk(input_file)
    if prob_threshold is not None:
        mask_array >= prob_threshold
    else:
        mask_array = (mask_array == label_value)

    
    mask_array = binary_dilation(mask_array, iterations=4)
    if not dilate:
        mask_array = binary_erosion(mask_array, iterations=4)
    else:
        ### dilated vessel
        mask_array = binary_erosion(mask_array, iterations=2)
    vol_coor = get_coordinates(mask_array.shape)
    # normalize to [-1, 1]
    if normalized:
        vol_coor = vol_coor / np.array(mask_array.shape).astype(float) * 2 - 1.0
    def save_points(_mask_array, suffix = ''):
        selector = (_mask_array).reshape((-1))
        mask = vol_coor[selector]
        ## save mask
        if mask_file is not None:
            save_ply(mask_file + suffix +'.ply', mask)        

            logger.info('extract {} points for mask, save to {}'.format(len(mask), mask_file + suffix +'.ply'))
        ## save surface
        if surface_file is not None:
            sur_array = np.logical_xor(_mask_array, binary_erosion(_mask_array))
            
            selector = sur_array.reshape((-1))
            
            sur = vol_coor[selector]

            save_ply(surface_file + suffix +'.ply', sur)
        

            logger.info('extract {} points for surface, save to {}'.format(len(sur), surface_file + suffix +'.ply'))  
        ## save skeleton
        if skeleton_file is not None:
            ske_array = skeletonize(_mask_array)
            selector = (ske_array >= 0.5).reshape((-1))
            ske = vol_coor[selector]
            save_ply(skeleton_file + suffix +'.ply', ske)
            logger.info('extract {} points for skeleton, save to {}'.format(len(ske), skeleton_file + suffix +'.ply'))  
        if xyzr_file is not None and surface_file is not None and skeleton_file is not None:
            # find radius of left skeleton
            tree = cKDTree(sur)
            dist, _ = tree.query(ske, k = 16)
            # compute the average distance to the nearest 4 points
            radius = np.mean(dist, axis = 1)
            # save radius to npy file
            xyzr = np.hstack((ske, radius[:, None]))
            np.savetxt(xyzr_file + suffix +'.xyzr', xyzr)      

    if not cc_axis:
        save_points(mask_array)
    else:
        assert cc_axis in ['x', 'y', 'z'], "Axis should be one of ['x', 'y','z']"
        label_array, num_features = label(mask_array, structure = np.ones((3, 3, 3), dtype = int))
        mask_mean = []
        along_axis = 0 if cc_axis == 'x' else 1 if cc_axis == 'y' else 2 
        logger.info(f'found {num_features} connected components.')
        valid_label = []
        for l in range(1, num_features + 1):
            selector = (label_array == l).reshape((-1))
            if selector.sum() < 1000:
                logger.info(f'remove label {l}')
                continue
            valid_label.append(l)
            mask = vol_coor[selector]
            mask_mean.append(mask.mean(axis=0)[along_axis])
        idx = np.array(valid_label)[np.argsort(np.array(mask_mean))]

        for i, l in enumerate(idx):
            _mask_array = (label_array == l)
            _mask_array = binary_dilation(_mask_array, iterations=2)
            save_points(_mask_array, suffix = f'_{i}')
        

## command: python -u pmap2xyzr.py -i /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/label/ -o /media/wlsdzyzl/DATA/datasets/pcd/imageCAS/output_lr/ -p 0.5 --cc_axis x --mask --surface --skeleton --xyzr --non_normalized > extract_xyzr.out
def main(argv):
    input_dir = ''
    output_dir = ''
    opts, args = getopt.getopt(argv, "hi:o:p:v:", ['help', 'input_dir=', 'input_suffix=', 'output_dir=', 'prob_threshold=', 
                                                   'label_value=', 'cc_axis=' , 'mask', 'surface','skeleton', 'xyzr', 
                                                   'non_normalized', 'dilate'])
    mask = False
    surface = False
    skeleton = False
    xyzr = False
    normalized = True
    prob_threshold = None
    label_value = None
    suffix = 'nii.gz'
    cc_axis = None
    dilate = False
    if len(opts) == 0:
        logger.error('unknow options, usage: pmap2xyzr.py -i <input_dir> --input_suffix <input_suffix=nii.gz> -o <output_dir> -p <prob_threshold = None> -v <label_value = None> --cc_axis <cc_axis=null> --mask --surface --skeleton --xyzr --non_normalized --dilate')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: pmap2xyzr.py -i <input_dir> --input_suffix <input_suffix=nii.gz> -o <output_dir> -p <prob_threshold = None> -v <label_value = None> --cc_axis <cc_axis=null> --mask --surface --skeleton --xyzr --non_normalized --dilate')
            sys.exit()
        elif opt in ("-i", '--input_dir'):
            input_dir = arg
        elif opt in ('--input_suffix'):
            suffix = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ("-p", '--prob_threshold'):
            prob_threshold = float(arg)
        elif opt in ('-v', '--label_value'):
            label_value = int(arg)
        elif opt in ('--cc_axis',):
            cc_axis = arg
        elif opt in ('--mask',):
            mask = True
        elif opt in ('--surface',):
            surface = True
        elif opt in ('--skeleton',):
            skeleton = True
        elif opt in ('--xyzr',):
            xyzr = True
        elif opt in ('--non_normalized',):
            normalized = False
        elif opt in ('--dilate',):
            dilate = True
        else:
            logger.error('unknow option, usage: pmap2xyzr.py -i <input_dir> --input_suffix <input_suffix=nii.gz> -o <output_dir> -p <prob_threshold = None> -v <label_value = None> --cc_axis <cc_axis=null> --mask --surface --skeleton --xyzr --non_normalized --dilate')
            sys.exit()
    assert label_value is not None or prob_threshold is not None, "At least one of [prob_threshold, label_value] should not be None."
    input_files = []
    filenames = []
    if os.path.isdir(output_dir):
        rkdirs(output_dir)
    input_files = sorted(glob.glob(os.path.join(input_dir, "*" + suffix)))
    filenames = [os.path.splitext(os.path.splitext(os.path.split(ifile)[1])[0])[0] for ifile in input_files]


    mask_file_list = [None for _ in range(len(input_files))]
    surface_file_list = [None for _ in range(len(input_files))]
    skeleton_file_list = [None for _ in range(len(input_files))]
    xyzr_file_list = [None for _ in range(len(input_files))]
    if mask:
        os.makedirs(output_dir+'/mask')
        mask_file_list = [ output_dir+'/mask/'+ filename  for filename in filenames]
    if surface:
        os.makedirs(output_dir+'/surface')
        surface_file_list = [ output_dir+'/surface/'+ filename  for filename in filenames]
    if skeleton:
        os.makedirs(output_dir+'/skeleton')
        skeleton_file_list = [ output_dir+'/skeleton/'+ filename  for filename in filenames]
    if xyzr:
        os.makedirs(output_dir+'/xyzr')
        xyzr_file_list = [ output_dir+'/xyzr/'+ filename for filename in filenames]

    for ifile, mask_file, surface_file, skeleton_file, xyzr_file in zip(input_files, mask_file_list, surface_file_list, skeleton_file_list, xyzr_file_list):
        f(ifile, mask_file, surface_file, skeleton_file, xyzr_file, normalized, prob_threshold = prob_threshold, label_value = label_value, cc_axis=cc_axis, dilate=dilate)
if __name__ == "__main__":
    main(sys.argv[1:])
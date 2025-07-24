from flemme.utils import load_itk
from flemme.logger import get_logger
import sys, getopt
import os
from glob import glob
import numpy as np
logger = get_logger('scripts::get_info_from_label')

def get_sample_info(label):
    unique_labels, counts = np.unique(label, return_counts=True)
    return label.shape, unique_labels, counts

def main(argv):
    label_dir = None 
    suffix='.nii.gz'
    opts, _ = getopt.getopt(argv, "hd:", ['help', 'label_dir=', 'suffix='])
    if len(opts) == 0:
        logger.info('unknow options, usage: get_info_from_label.py -d <label_dir> --suffix <suffix=.nii.gz>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: get_info_from_label.py -d <label_dir> --suffix <suffix=.nii.gz>')
            sys.exit()
        if opt in ('-d', '--label_dir'):
            label_dir = arg
        if opt in ('--suffix'):
            suffix = arg
    if label_dir is None:
        logger.info('label_dir is required.')
        exit(1)

    label_paths = sorted(glob(os.path.join(label_dir + '/' , '*'+suffix)))
    img_shapes = []
    label_to_count = {}
    logger.info(f'Getting info from {len(label_paths)} samples from {label_dir}.')
    for lp in label_paths:
        l = load_itk(lp)
        shape, labels, label_counts = get_sample_info(l)
        # logger.info(f"image_shape: {shape}")
        img_shapes.append(shape)
        for label, count in zip(labels, label_counts):
            if label in label_to_count:
                label_to_count[label] += count
            else:
                label_to_count[label] = count
    img_shapes = np.array(img_shapes)
    for l in label_to_count:
        label_to_count[l] = label_to_count[l] / len(img_shapes)
    logger.info('Max image size: {}'.format(img_shapes.max(0)))
    logger.info('Min image size: {}'.format(img_shapes.min(0)))
    logger.info('Average image size: {}'.format(img_shapes.mean(0)))
    logger.info('Average label count:{}'.format(label_to_count))

if __name__ == "__main__":
    main(sys.argv[1:])

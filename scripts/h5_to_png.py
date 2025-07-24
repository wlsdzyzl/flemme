# from hd5 to png

import h5py
import os
from flemme.utils import normalize_img, save_img, rkdirs
from flemme.logger import get_logger
from glob import glob
import sys, getopt
logger = get_logger('scripts::h5_to_png')
def f(h5_files, keys, output_dirs):
    center = None
    scaling = None
    for sub_id in range(len(h5_files)):
        h5_file = h5_files[sub_id]
        file = h5py.File(h5_files[sub_id], 'r')
        key = keys[sub_id]
        output_dir = output_dirs[sub_id]

        assert key in file.keys(), '{} not in {}'.format(key, file.keys())
        img_array, (center, scaling) = \
            normalize_img(file[key],
                            channel_dim=None, 
                            percentile_range=[2, 98], 
                            center=center, scaling=scaling,
                            return_transform=True)
        # img_array = ndimage.median_filter(img_array, size=2)
        filename = os.path.splitext(os.path.basename(h5_file))[0]
        logger.info('converting {} to {} images ...'.format(h5_file, len(img_array)) )
        for idx, img in enumerate(img_array):
            img = (img* 255).astype('uint8')
            save_img(f'{output_dir}/{filename}_{idx:0>2}.png', img)


def main(argv):
    dataset_path = '.'
    sub_dirs = ['.']
    suffix = ['.h5']
    output_dir = None
    keys = ['raw']
    opts, _ = getopt.getopt(argv, "hp:o:k:", ['help', 'dataset_path=', 'sub_dirs=', 'suffix=', 'keys=', 'output_dir='])
    if len(opts) == 0:
        logger.info('unknow options, usage: h5_to_png.py -p <dataset_path> -o <output_dir=.> --sub_dirs <sub_dirs=\'.\'> --suffix <suffix=.h5> --keys <keys=\'\'> ')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: h5_to_png.py -p <dataset_path> -o <output_dir=.> --sub_dirs <sub_dirs=\'.\'> --suffix <suffix=.h5> --keys <keys=\'\'> ')
            sys.exit()
        if opt in ('-p', '--dataset_path'):
            dataset_path = arg
        elif opt in ('--suffix',):
            suffix = arg.split(',')
        elif opt in ('--sub_dirs'):
            sub_dirs = arg.split(',')
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('-k', '--keys'):
            keys = arg.split(',')
    if dataset_path is None:
        logger.error('dataset_path is required.')
        sys.exit()
    if output_dir is None:
        logger.error('output_dir is required.')
        sys.exit()
    if len(suffix) == 1:
        suffix = suffix * len(sub_dirs)
    if len(keys) == 1:
        keys = keys * len(sub_dirs)
    assert not output_dir == dataset_path, 'output_dir and dataset_path are the same.'
    assert len(keys) == len(sub_dirs) and len(suffix) == len(keys), 'length of sub_dirs, keys and suffix should be the same.'

    contained_files = []
    output_sub_dirs = [] 
    for sd, sf in zip(sub_dirs, suffix):
        output_sub_dirs.append(output_dir+'/'+sd)
        rkdirs(output_dir+'/'+sd)
        if len(contained_files) == 0:
            files = sorted(glob(os.path.join(dataset_path + '/' +sd, "*" + sf)))
            contained_files.append(files)
        else:
            contained_files.append([ file.replace(sub_dirs[0], sd).replace(suffix[0], sf) for file in contained_files[0]])
    for fid in range(len(contained_files[0])):
        h5_files = [contained_files[sub_id][fid] for sub_id in range(len(sub_dirs))]
        f(h5_files, keys, output_sub_dirs)

if __name__ == "__main__":
    main(sys.argv[1:])
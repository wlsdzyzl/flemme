from flemme.utils import *
import sys, getopt
import numpy as np 
import os
from glob import glob
from flemme.logger import get_logger
logger = get_logger('scripts::crop_by')
def f(wait_to_crop, output_files, margin, 
    background, crop_by = None, boundingbox = None):
    data_array, origin, spacing = load_itk(wait_to_crop[0], True)
    wait_to_crop_data = [data_array]
    wait_to_crop_data = wait_to_crop_data + \
        [ load_itk(wait_to_crop[i]) for i in range(1, len(wait_to_crop))]
    if crop_by is not None:
        logger.info('crop {} by {} with margin {}'.format(wait_to_crop, crop_by, margin))
        _, cropped_follows, (start_idx, _) = crop_boundingbox(data = load_itk(crop_by), 
            margin = margin, background = background, boundingbox = boundingbox, 
            follows = wait_to_crop_data)
    else:
        logger.info('crop {} by bounding box {} with margin {}'.format(wait_to_crop, boundingbox, margin))
        cropped_follows, (start_idx, _) = crop_boundingbox(margin = margin, 
            background = background, boundingbox = boundingbox, 
            follows = wait_to_crop_data)
    origin = origin + start_idx.astype(float) * spacing
    logger.info('crop from', str(wait_to_crop_data[0].shape), 'to', str(cropped_follows[0].shape))
    for op, od in zip(output_files, cropped_follows):
        save_itk(op, od, origin = origin, spacing = spacing)


def main(argv):
    dataset_path = None
    sub_dirs = ['.']
    suffix = ['']
    output_dir = None
    crop_by = None
    margin = (20, 20, 20)
    background = 0.0
    separately = False
    boundingbox = None
    opts, args = getopt.getopt(argv, "hp:o:m:b:", ['help', 'dataset_path=', 'sub_dirs=', 'suffix=', 'crop_by=', 'output_dir=',  'margin=', 'background=', 'separately', 'boundingbox='])
    
    if len(opts) == 0:
        logger.error('unknow options, usage: crop_by.py -p <dataset_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> --crop_by <crop_by=None> -o <output_dir=.> -m <margin = 20,20,20>  -b <background = 0.0> --boundingbox <boundingbox = None>.')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: crop_by.py -p <dataset_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> --crop_by <crop_by=None> -o <output_dir=.> -m <margin = 20,20,20>  -b <background = 0.0> --boundingbox <boundingbox = None>.')
            sys.exit()
        if opt in ('-p', '--dataset_path'):
            dataset_path = arg
        elif opt in ('--sub_dirs',):
            sub_dirs = arg.split(',')
        elif opt in ('--suffix',):
            suffix = arg.split(',')
        elif opt in ('--crop_by',):
            crop_by = arg
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ("-m", '--margin'):
            margin = tuple(int(s) for s in arg.split(','))
            if len(margin) != 3:
                logger.info('Error: The length of margin need to be 3. Example: -m 20,20,20')
                sys.exit() 
        elif opt in ('-b', '--background'):
            background = float(arg)        
        elif opt in ('--separately',):
            separately = True
        elif opt in ('--boundingbox',):
            boundingbox = tuple(int(s) for s in arg.split(','))
            if len(boundingbox) != 6:
                logger.info('Error: The length of boundingbox need to be 6. Example: --boundingbox 0,0,0,100,200,100')
                sys.exit() 
        else:
            logger.info('usage: crop_by.py -p <dataset_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> --crop_by <crop_by=None> -o <output_dir=.> -m <margin = 20,20,20>  -b <background = 0.0> --boundingbox <boundingbox = None>.')
            sys.exit()
    if len(suffix) == 1:
        suffix = suffix * len(sub_dirs)
    # logger.info(suffix, sub_dirs)
    assert len(suffix) == len(sub_dirs), 'sub_dirs and suffix should have the same length.'
    assert crop_by is not None or boundingbox is not None, \
            'At lease one of crop_by or boundingbox should not be None.'
    assert not output_dir == dataset_path, 'output_dir and dataset_path are the same.'
    out_sub_dirs = []
    for sub_dir in sub_dirs:
        out_sub_dir = os.path.join(output_dir, sub_dir)
        rkdirs(out_sub_dir)
        out_sub_dirs.append(out_sub_dir)
        
    
    contained_files = []
    for sd, sf in zip(sub_dirs, suffix):
        if len(contained_files) == 0:
            files = sorted(glob(os.path.join(dataset_path + '/' +sd, "*" + sf)))
            contained_files.append(files)
        else:
            contained_files.append([ file.replace(sub_dirs[0], sd).replace(suffix[0], sf) for file in contained_files[0]])
    
    crop_by_id = sub_dirs.index(crop_by) if crop_by is not None else -1
    if not separately:
        if crop_by_id >= 0:
            boundingboxes = []
            for il in contained_files[crop_by_id]:
                logger.info('read from {}'.format(il))
                label_array = load_itk(il)
                tmp_bb = get_boundingbox(label_array)
                logger.info('get bounding box:', tmp_bb)
                boundingboxes.append(tmp_bb)
            final_bb = get_boundingbox_from_list(boundingboxes = boundingboxes)
            logger.info('final bounding box:', final_bb)
        else:
            final_bb = np.array(boundingbox[0:3]), np.array(boundingbox[3:])
            logger.info('get bounding box from input:', final_bb)
    else:
        if boundingbox is not None:
            logger.info('Crop separately, input boundingbox is ignored.')
            
    for idx in range(len(contained_files[0])):
        wait_to_crop = [contained_files[sub_id][idx] for sub_id in range(len(contained_files))]
        output_files = [out_sub_dirs[sub_id] + '/' + os.path.basename(wait_to_crop[sub_id]) for sub_id in range(len(contained_files))]
        if not separately:
            f(wait_to_crop, output_files = output_files, margin = margin, 
                background = background, boundingbox = final_bb)
        else:
            f(wait_to_crop, output_files = output_files, margin = margin, 
                background = background, crop_by = wait_to_crop[crop_by_id])

if __name__ == "__main__":
    main(sys.argv[1:])
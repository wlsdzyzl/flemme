import sys, getopt
import os
import random
import shutil
import math
from glob import glob
from flemme.logger import get_logger
from flemme.utils import rkdirs
logger = get_logger('scripts.random_split_k_fold')
## tmp file
def main(argv):
    dataset_path = None
    output_dir = None
    sub_dirs = ['.']
    suffix = ['']
    kfold = 5
    opts, _ = getopt.getopt(argv, "hp:o:k:", ['help', 'dataset_path=', 'sub_dirs=', 'suffix=', 'output_dir=', 'kfold=', 'method=', 'separately'])
    method = shutil.copy
    mn = 'copy'
    separately = False
    ### move is faster, but with higher risk of losing data.
    if len(opts) == 0:
        logger.info('unknow options, usage: random_split_k_fold.py -p <dataset_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> -o <output_dir=.> -k <kfold=5> --method <method=copy>')
        sys.exit()
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            logger.info('usage: random_split_k_fold.py -p <dataset_path> --sub_dirs <sub_dirs=.> --suffix <suffix=.> -o <output_dir=.> -k <kfold=5> --method <method=copy>')
            sys.exit()
        if opt in ('-p', '--dataset_path'):
            dataset_path = arg
        elif opt in ('--sub_dirs',):
            sub_dirs = arg.split(',')
        elif opt in ('--suffix',):
            suffix = arg.split(',')
        elif opt in ("-o", '--output_dir'):
            output_dir = arg
        elif opt in ('-k', '--kfold'):
            k = int(arg)
        elif opt in ('--method',):
            mn = arg
            if mn == 'move':
                method = shutil.move
            elif not mn == 'copy':
                logger.info('Unknow operation.')
                exit(1)
        elif opt in ('--separately',):
            separately = True
    if dataset_path is None:
        logger.error('dataset_path is required.')
        sys.exit()
    if output_dir is None:
        logger.error('output_dir is required.')
        sys.exit()
    if len(suffix) == 1:
        suffix = suffix * len(sub_dirs)
    # logger.info(suffix, sub_dirs)
    assert len(suffix) == len(sub_dirs), 'sub_dirs and suffix should have the same length.'
    assert sum([s == '' for s in sub_dirs]) == 0, 'sub_dirs includes empty folder name.'
    if not separately:
        contained_files = []
        for sd, sf in zip(sub_dirs, suffix):
            if len(contained_files) == 0:
                files = sorted(glob(os.path.join(dataset_path + '/' +sd, "*" + sf)))
                contained_files.append(files)
            else:
                contained_files.append([ file.replace(sub_dirs[0], sd).replace(suffix[0], sf) for file in contained_files[0]])
        file_id = list(range(len(contained_files[0])))
        # Shuffle the data
        random.shuffle(file_id)
        fold_size = math.ceil(len(file_id) / kfold)
        fold_file_id = [file_id[i:i+fold_size] for i in range(0, len(file_id), fold_size)]
        ## move or copy data
        for k, ff_id in enumerate(fold_file_id):
            logger.info(f'creating {k+1}-th fold ...') 
            for sub_id, sub_dir in enumerate(sub_dirs):
                fold_dir = os.path.join(output_dir, f"fold{k+1}/{sub_dir}")
                rkdirs(fold_dir)

                for f_id in ff_id:
                    logger.info(f'{mn} {contained_files[sub_id][f_id]} to {fold_dir}'  ) 
                    method(contained_files[sub_id][f_id], os.path.join(fold_dir, os.path.basename(contained_files[sub_id][f_id])))
    else:
        contained_files = []
        logger.info('Spliting files in different folders separately ...')
        for sd, sf in zip(sub_dirs, suffix):
            files = sorted(glob(os.path.join(dataset_path + '/' +sd, "*" + sf)))
            contained_files.append(files)
        for sub_id, sub_dir in enumerate(sub_dirs):
            logger.info(f'Processing subdir {sub_dir}, which has {len(contained_files[sub_id])} files.')

            file_id = list(range(len(contained_files[sub_id])))
            # Shuffle the data
            random.shuffle(file_id)
            fold_size = math.ceil(len(file_id) / kfold)
            fold_file_id = [file_id[i:i+fold_size] for i in range(0, len(file_id), fold_size)]
            for k, ff_id in enumerate(fold_file_id):
                fold_dir = os.path.join(output_dir, f"fold{k+1}/{sub_dir}")
                rkdirs(fold_dir)
                for f_id in ff_id:
                    logger.info(f'{mn} {contained_files[sub_id][f_id]} to {fold_dir}'  ) 
                    method(contained_files[sub_id][f_id], os.path.join(fold_dir, os.path.basename(contained_files[sub_id][f_id])))
if __name__ == "__main__":
    main(sys.argv[1:])
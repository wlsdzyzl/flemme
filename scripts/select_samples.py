### this file is just a tool to select samples that fully supported your conclusion.
import sys, getopt
import os
import random
import shutil
import math
from glob import glob
from flemme.logger import get_logger
from flemme.utils import load_img, load_itk, load_npy
from flemme.metrics import get_metrics
import numpy as np
logger = get_logger('scripts')
## tmp file
#### python select_samples.py --result_path path/to/results/seg/CVC-ClinicDB --sub_dirs ResNet,ResNet_HSeg,UNet,UNet_HSeg,UNet_Atten,UNet_Atten_HSeg,SwinU,SwinU_HSeg,MambaU,MambaU_HSeg --suffix .png --target_path path/to/results/seg/CVC-ClinicDB/target --target_suffix _tar.png --conditions '0<1,2>3,4>5,6>7,8>9' --eval Dice
def main(argv):
    result_path = None
    target_path = None
    sub_dirs = ['.']
    suffix = ['']
    target_suffix = ''
    conditions = ['']
    eval_config = {'name':'mIoU'}
    opts, _ = getopt.getopt(argv, "h", ['help', 'result_path=', 'sub_dirs=', 'suffix=', 'target_path=', 'target_suffix=', 'conditions=', 'eval='])
    ### move is faster, but with higher risk for losing data.
    if len(opts) == 0:
        logger.info('unknow options, usage: selected_samples.py --result_path <result_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> --target_path <target_path=.> --target_suffix <target_suffix=\'\'> --conditions <conditions=> --eval <eval=>')
        sys.exit()
    for opt, arg in opts:
        # print(arg)
        if opt in ('-h', '--help'):
            logger.info('usage: selected_samples.py --result_path <result_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> --target_path <target_path=.> --target_suffix <target_suffix=\'\'> --conditions <conditions=> --eval <eval=>')
            sys.exit()
        if opt in ('--result_path',):
            result_path = arg
        elif opt in ('--sub_dirs',):
            sub_dirs = arg.split(',')
        elif opt in ('--suffix',):
            suffix = arg.split(',')
        elif opt in ('--target_path',):
            target_path = arg
        elif opt in ('--target_suffix',):
            target_suffix = arg
        elif opt in ('--conditions', ):
            
            conditions = arg.split(',')
        elif opt in ('--eval',):
            eval_config['name'] = arg
    if result_path is None:
        logger.error('result_path is required.')
        sys.exit()
    if target_path is None:
        logger.error('target_path is required.')
        sys.exit()
    eval_func = get_metrics(eval_config)
    if len(suffix) == 1:
        suffix = suffix * len(sub_dirs)
    # logger.info(suffix, sub_dirs)
    assert len(suffix) == len(sub_dirs), 'sub_dirs and suffix should have the same length.'
    assert sum([s == '' for s in sub_dirs]) == 0, 'sub_dirs includes empty folder name.'

    result_files = []
    for sd, sf in zip(sub_dirs, suffix):
        if len(result_files) == 0:
            files = sorted(glob(os.path.join(result_path + '/' +sd, "*" + sf)))
            result_files.append(files)
        else:
            result_files.append([ file.replace(sub_dirs[0], sd).replace(suffix[0], sf) for file in result_files[0]])

    target_files = [ file.replace(result_path + '/' + sub_dirs[0], target_path).replace(suffix[0], target_suffix) 
                        for file in result_files[0]]
    ### sample ids
    selected_samples = list(range(len(target_files)))
    for cond in conditions:
        assert '>' in cond or '<' in cond, 'conditions should be \'>\' or \'<\''
        if '>' in cond:
            result_ids = cond.split('>')
        else:
            result_ids = cond.split('<')[::-1]
        assert len(result_ids) == 2, 'There should be two results in each comparison.'
        result_ids = [int(r) for r in result_ids]
        new_selected_samples = []
        for sample_id in selected_samples:
            if 'nii.gz' in target_suffix:
                res1, _ = load_itk(result_files[result_ids[0]][sample_id])
                res2, _ = load_itk(result_files[result_ids[1]][sample_id])
                tar, _ = load_itk(target_files[sample_id])
            elif 'png' in target_suffix:
                res1 = np.array(load_img(result_files[result_ids[0]][sample_id]))
                res2 = np.array(load_img(result_files[result_ids[1]][sample_id]))
                tar = np.array(load_img(target_files[sample_id]))
                res1 = ((res1 / 255) > 0.5).astype(int)
                res2 = ((res2 / 255) > 0.5).astype(int)
                tar = ((tar / 255) > 0.5).astype(int)
            else:
                res1 = load_npy(result_files[result_ids[0]][sample_id])
                res2 = load_npy(result_files[result_ids[1]][sample_id])
                tar = load_npy(target_files[sample_id])
            if eval_func(tar, res1) >= eval_func(tar, res2):
                new_selected_samples.append(sample_id)
        selected_samples = new_selected_samples
        if len(selected_samples) == 0: break
    sample_name = [target_files[sid] for sid in selected_samples]
    print(f'selected {len(sample_name)} samples')
    print(sample_name)
            
if __name__ == "__main__":
    main(sys.argv[1:])
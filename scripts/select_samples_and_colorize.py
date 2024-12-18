### this file is just a tool to select samples that fully supported your conclusion.
import sys, getopt
import os
import random
import shutil
import math
from glob import glob
from flemme.logger import get_logger
from flemme.utils import rkdirs, get_boundingbox, save_img
from flemme.trainer_utils import colorize_img_by_label, save_data, get_load_function
from flemme.metrics import get_metrics
import numpy as np
logger = get_logger('scripts')
## tmp file
#### python select_samples.py --result_path path/to/results/seg/CVC-ClinicDB --sub_dirs ResNet,ResNet_HSeg,UNet,UNet_HSeg,UNet_Atten,UNet_Atten_HSeg,SwinU,SwinU_HSeg,MambaU,MambaU_HSeg --suffix .png --target_path path/to/results/seg/CVC-ClinicDB/target --target_suffix _tar.png --conditions '0<1,2>3,4>5,6>7,8>9' --eval Dice
def main(argv):
    result_path = None
    target_path = None
    input_path = None
    output_dir = None

    sub_dirs = ['.']
    target_sub_dirs = ['.']
    input_sub_dirs = ['.']

    suffix = ['']
    target_suffix = ['']
    input_suffix = ['']
    
    conditions = ['']
    eval_config = {'name':'mIoU'}
    minimum_ratio = 0.05
    ## for better visualization
    compute_middle_for_3d = False
    opts, _ = getopt.getopt(argv, "h", ['help', 'result_path=', 'sub_dirs=', 'suffix=', 'target_path=',  'target_sub_dirs=', 'target_suffix=', 'input_path=', 'output_dir=', 'input_sub_dirs=', 'input_suffix=', 'conditions=', 'eval=', 'compute_middle_for_3d', 'minimum_ratio='])
    ### move is faster, but with higher risk for losing data.
    if len(opts) == 0:
        logger.info('unknow options, usage: select_samples_and_colorize.py --result_path <result_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> --target_path <target_path=.> --target_sub_dirs <target_sub_dirs=.> --target_suffix <target_suffix=.> --input_path <input_path=None> --input_sub_dirs <input_sub_dirs=.> --input_suffix <input_suffix=.> --output_dir <output_dir=None> --conditions <conditions=> --eval <eval=> --minimum_ratio <minimum_ratio=0.05>')
        sys.exit()
    for opt, arg in opts:
        # print(arg)
        if opt in ('-h', '--help'):
            logger.info('usage: select_samples_and_colorize.py --result_path <result_path> --sub_dirs <sub_dirs=.> --suffix <suffix=\'\'> --target_path <target_path=.> --target_sub_dirs <target_sub_dirs=.> --target_suffix <target_suffix=\'\'>  --input_path <input_path=None> --input_sub_dirs <input_sub_dirs=.> --input_suffix <input_suffix=.> --output_dir <output_dir=None>  --conditions <conditions=> --eval <eval=> --minimum_ratio <minimum_ratio=0.05>')
            sys.exit()
        if opt in ('--result_path',):
            result_path = arg
        elif opt in ('--sub_dirs',):
            sub_dirs = arg.split(',')
        elif opt in ('--suffix',):
            suffix = arg.split(',')
        elif opt in ('--target_path',):
            target_path = arg
        elif opt in ('--target_sub_dirs',):
            target_sub_dirs = arg.split(',')
        elif opt in ('--target_suffix',):
            target_suffix = arg.split(',')
        elif opt in ('--input_path',):
            input_path = arg
        elif opt in ('--input_sub_dirs',):
            input_sub_dirs = arg.split(',')
        elif opt in ('--input_suffix',):
            input_suffix = arg.split(',')
        elif opt in ('--output_dir',):
            output_dir = arg
        elif opt in ('--conditions', ):
            conditions = arg.split(',')
        elif opt in ('--eval',):
            eval_config['name'] = arg
        elif opt in ('--compute_middle_for_3d',):
            compute_middle_for_3d = True
        elif opt in ('--minimum_ratio', ):
            minimum_ratio = float(arg)
    if result_path is None:
        logger.error('result_path is required.')
        sys.exit()
    if target_path is None:
        logger.error('target_path is required.')
        sys.exit()
    if input_path is None:
        logger.info('There is no input_path, selected samples will not be colorized.')
        sys.exit()
    eval_func = get_metrics(eval_config)
    if len(suffix) == 1:
        suffix = suffix * len(sub_dirs)
    
    if len(target_sub_dirs) == 1:
        target_sub_dirs = target_sub_dirs * len(sub_dirs)
    if len(target_suffix) == 1:
        target_suffix = target_suffix * len(sub_dirs)

    if len(input_sub_dirs) == 1:
        input_sub_dirs = input_sub_dirs * len(sub_dirs)
    if len(input_suffix) == 1:
        input_suffix = input_suffix * len(sub_dirs)
    # logger.info(suffix, sub_dirs)
    assert len(suffix) == len(sub_dirs), 'sub_dirs and suffix should have the same length.'
    assert len(sub_dirs) == len(target_sub_dirs), 'sub_dirs and target_sub_dirs should have the same length.'
    assert sum([s == '' for s in sub_dirs]) == 0, 'sub_dirs contains empty folder name.'
    assert sum([s == '' for s in target_sub_dirs]) == 0, 'target_sub_dirs contains empty folder name.'

    result_files = []
    for sd, sf in zip(sub_dirs, suffix):
        if len(result_files) == 0:
            files = sorted(glob(os.path.join(result_path + '/' +sd, "*" + sf)))
            result_files.append(files)
        else:
            result_files.append([ file.replace(sub_dirs[0], sd).replace(suffix[0], sf) for file in result_files[0]])

    target_files = [[file.replace(result_path + '/' + sub_dirs[idx], target_path + '/' + target_sub_dirs[idx]).replace(suffix[idx], target_suffix[idx]) for file in files ]
        for idx, files in enumerate(result_files)]

    # print(result_path + '/' + sub_dirs[0], target_path + '/' + target_sub_dirs[0])
    # print(target_files)
    ### sample ids
    load_data, label_type = get_load_function(suffix[0])
    load_target_data, _ = get_load_function(target_suffix[0])

    selected_samples = list(range(len(target_files[0])))
    selected_slices = [None for _ in selected_samples]
    for cond in conditions:
        print('processing the first condition:', cond)
        assert '>' in cond or '<' in cond, 'conditions should be \'>\' or \'<\''
        if '>' in cond:
            result_ids = cond.split('>')
        else:
            result_ids = cond.split('<')[::-1]
        assert len(result_ids) == 2, 'There should be two results in each comparison.'
        result_ids = [int(r) for r in result_ids]
        new_selected_samples = []
        new_selected_slices = []
        for idx, sample_id in enumerate(selected_samples):
            res1 = load_data(result_files[result_ids[0]][sample_id])
            res2 = load_data(result_files[result_ids[1]][sample_id])
            tar1 = load_target_data(target_files[result_ids[0]][sample_id])
            tar2 = load_target_data(target_files[result_ids[1]][sample_id])    
            if label_type == 'img':
                res1 = ((res1 / 255) > 0.5).astype(int)
                res2 = ((res2 / 255) > 0.5).astype(int)
                tar1 = ((tar1 / 255) > 0.5).astype(int)
                tar2 = ((tar2 / 255) > 0.5).astype(int)
            if label_type == 'vol':
                res1, res2, tar1, tar2 = res1[0], res2[0], tar1[0], tar2[0]
                if compute_middle_for_3d:
                    min_zxy, max_zxy = get_boundingbox(tar1, background = 0)
                    middle_z = int(min_zxy[0] + max_zxy[0]) // 2
                    res1, tar1 = res1[middle_z], tar1[middle_z]
                    
                    min_zxy, max_zxy = get_boundingbox(tar2, background = 0)
                    middle_z = int(min_zxy[0] + max_zxy[0]) // 2
                    res2, tar2 = res2[middle_z], tar2[middle_z]   
                ### compute each slice, make sure the inputs has the same shape
            if label_type == 'vol' and not compute_middle_for_3d:
                assert res1.shape[0] == res2.shape[0], 'If we evaluate each slice, the inputs should have the same length on z-dimension.'
                tmp_selected_slices = []
                old_selected_slices = selected_slices[idx]
                if old_selected_slices is None: 
                    z = res1.shape[0]
                    old_selected_slices = list(range(z))
                for slice_id in old_selected_slices:
                    sres1, sres2, star1, star2 = res1[slice_id], res2[slice_id], tar1[slice_id], tar2[slice_id]
                    # print(sres1.sum() / sres1.size)
                    if sres1.sum() / sres1.size <= minimum_ratio or \
                        sres2.sum() / sres2.size <= minimum_ratio or \
                        star1.sum() / star1.size <= minimum_ratio or \
                        star2.sum() / star2.size <= minimum_ratio: 
                        continue
                    score1, score2 = eval_func(star1, sres1), eval_func(star2, sres2)
                    if score1 >= score2:
                        tmp_selected_slices.append(slice_id)
                if len(tmp_selected_slices) > 0:
                    new_selected_samples.append(sample_id)
                    new_selected_slices.append(tmp_selected_slices)
            else:
                if res1.sum() / res1.size <= minimum_ratio or \
                    res2.sum() / res2.size <= minimum_ratio or \
                    tar1.sum() / tar1.size <= minimum_ratio or \
                    tar2.sum() / tar2.size <= minimum_ratio: 
                    continue
                score1, score2 = eval_func(tar1, res1), eval_func(tar2, res2)
                if score1 >= score2:
                    new_selected_samples.append(sample_id)

        selected_samples = new_selected_samples
        selected_slices = new_selected_slices
        if len(selected_samples) == 0: break
    sample_name = [os.path.basename(target_files[0][sid]) for sid in selected_samples]
    print(f'selected {len(sample_name)} samples from {len(target_files[0])} samples')
    print(sample_name)

    
    if input_path is not None and len(sample_name) > 0:
        assert output_dir is not None, 'output dir is not specified.'
        load_input_data, _ = get_load_function(input_suffix[0])
        input_files = [[file.replace(result_path + '/' + sub_dirs[idx], input_path + '/' + input_sub_dirs[idx]).replace(suffix[idx], input_suffix[idx]) for file in files ]
            for idx, files in enumerate(result_files)]
        for group_id in range(len(result_files)):
            output_subdir = os.path.join(output_dir, sub_dirs[group_id])
            rkdirs(output_subdir)
            print(f'colorizing selected samples for the {group_id}-th group ...')
            for idx, sample_id in enumerate(selected_samples):        
                res = load_data(result_files[group_id][sample_id])
                tar = load_target_data(target_files[group_id][sample_id])
                input_ = load_input_data(input_files[group_id][sample_id])
                if label_type == 'img':
                    res = ((res1 / 255) > 0.5).astype(int)
                    tar = ((tar1 / 255) > 0.5).astype(int)
                if label_type == 'vol':
                    res, tar, input_ = res[0], tar[0], input_[0]
                    if compute_middle_for_3d:
                        min_zxy, max_zxy = get_boundingbox(tar, background = 0)
                        middle_z = int(min_zxy[0] + max_zxy[0]) // 2
                        res, tar, input_ = res[middle_z], tar[middle_z], input_[middle_z]
                if label_type == 'vol' and not compute_middle_for_3d:
                    for slice_id in selected_slices[idx]:
                        sres, star, sinput = res[slice_id], tar[slice_id], input_[slice_id]
                        output_path = os.path.join(output_subdir, os.path.basename(result_files[group_id][sample_id]).replace(suffix[group_id], f'_{slice_id}.png'))
                        cdata, raw_img =  colorize_img_by_label(sres[None, :], sinput[None, :], gt = star[None, :])                        
                        save_img(output_path, (cdata * 255).astype('uint8') )
                        save_img(output_path + '.raw.png', (raw_img * 255).astype('uint8') )
                        ctar, _ = colorize_img_by_label(star[None, :], sinput[None, :], gt = star[None, :])
                        save_img(output_path + '.tar.png', (ctar * 255).astype('uint8') )
                else:
                    output_path = os.path.join(output_subdir, os.path.basename(result_files[group_id][sample_id]).replace(suffix[group_id], '.png'))
                    cdata, raw_img = colorize_img_by_label(res[None, :], input_[None, :], gt = tar[None, :])
                    save_img(output_path, (cdata * 255).astype('uint8') )
                    save_img(output_path + '.raw.png', (raw_img * 255).astype('uint8') )
                    ctar, _ = colorize_img_by_label(tar[None, :], input_[None, :], gt = tar[None, :])
                    save_img(output_path + '.tar.png', (ctar * 255).astype('uint8') )
if __name__ == "__main__":
    main(sys.argv[1:])


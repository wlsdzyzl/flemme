#### This file is a script, and the code style sucks. 
import os
import torch
import glob
from .trainer_utils import *
from flemme.logger import get_logger
from flemme.augment import get_transforms
from flemme.dataset import sub_dataloader_from_files
from .test import save_results
logger = get_logger('trainer.eval')

def eval_run(t):
    ### split x, y, path in later implementation.
    x, y, _, _, path = process_input(t)
    ### stack prediction into batch form
    res = {'input': x}
    res['target'] = y
    res['path'] = path
    return res
def evaluate(eval_config,
        create_loader_fn,
        run_fn):
    eval_config = load_config()
    #### create dataset and dataloader
    save_target = eval_config.get('save_target', False)
    save_input = eval_config.get('save_input', False)
    save_colorized = eval_config.get('save_colorized', False)
    eval_type = eval_config.get('eval_type', 'segmentation')
    if eval_type == 'segmentation': eval_type = 'seg'
    elif eval_type == 'reconstruction': eval_type = 'recon'
    elif eval_type == 'generation': eval_type = 'gen'
    assert eval_type in ['seg', 'recon', 'gen'], 'Currently we only support evaluation for segmentation, reconstruction and generation.'

    eval_batch_num = eval_config.get('eval_batch_num', float('inf'))
    is_conditional = eval_config.get('is_conditional', False)

    pickle_results = eval_config.get('pickle_results', False)
    pickle_path = eval_config.get('pickle_path', 'flemme-pickled')
    if pickle_results:
        mkdirs(pickle_path) 

    if save_target:
        logger.info('Save target for reconstruction and segmentation tasks.')
    if save_input:
        logger.info('Save input for reconstruction and segmentation tasks.')
    
    dataset_name = None
    loader_config = eval_config.get('loader', None) 
    split_files = eval_config.get('split_files', None)
    data_form = eval_config.get('data_form', 'img')
    
    if data_form == 'pcd': data_form = DataForm.PCD
    elif data_form == 'img': data_form = DataForm.IMG
    else:
        logger.error('Currently we only support "pcd", and "img" data form.')
        exit(1)
    channel_dim = -1 if data_form == DataForm.PCD else 0
    ## information about saved predictions
    pred_info = eval_config.get('prediction', None)
    assert pred_info is not None, 'There is no information about predictions.'
    pred_path = pred_info.get('path')
    pred_suffix = pred_info.get('suffix')    
    is_mesh = eval_config.get('is_mesh', False)
    load_data = get_load_function(pred_suffix, is_mesh)[0]
    pred_trans_list = pred_info.get('transforms', [])
    pred_transforms = get_transforms(pred_trans_list, data_form=data_form)

    ## create evaluators
    eval_metrics = eval_config.get('evaluation_metrics', None)
    assert eval_metrics is not None, 'Please specify the evaluation metrics.'
    eval_metrics = {eval_type: eval_metrics}
    evaluators = create_batch_evaluators(eval_metrics, data_form)

    if loader_config is not None:
        assert eval_type in ['seg', 'recon'], \
            'Currently we only support evaluation for segmentation and reconstruction with dataloader.'
        logger.info('Creating dataloader for evaluation ...')
        ## even if there are multiple datasets, the predictions are still stored in a single folder.
        # if 'data_path_list' in loader_config and len(loader_config['data_path_list']) > 1:
        #     logger.error('Currently we only support single dataset evaluation.')
        #     exit(1)
        results = []
        loader_config['mode'] = 'test'
        dataset_name = loader_config.get('dataset').get('name')
        loader = create_loader_fn(loader_config)
        data_loader = loader['data_loader']
        if split_files:
            if type(split_files) == dict:
                assert 'test' in split_files, 'test is not in split_files when perform testing.'
                split_files = split_files['test']
            data_loader = sub_dataloader_from_files(data_loader, split_files, 
                                    shuffle = loader_config.get('shuffle', False))
        input_suffix = loader_config['dataset'].get('data_suffix', 'png')
        if type(input_suffix) == tuple or type(input_suffix) == list:
            input_suffix = input_suffix[0]
        assert data_form == loader['data_form'], 'Unmatched data form between loader and configuration.'
        
        logger.info('Finish parsing dataset(s).')
        logger.info('Data sample (test) count: {}'.format(len(data_loader.dataset)))
        ### load targets and predictions
        iter_id = 0
        for t in tqdm(data_loader, desc="Loading predictions and targets ..."):
            res = run_fn(t)
            pred = []
            for p in res['path']:
                filename = os.path.basename(p).replace(input_suffix, pred_suffix)
                tmp = pred_transforms(load_data(os.path.join(pred_path, filename)))
                pred.append(tmp)  
            res[eval_type] = torch.tensor(np.stack(pred))
            iter_id += 1
            if len(results) < eval_batch_num:
                process_results(results=results, 
                        res = res,
                        data_form = data_form,
                        pickle_results=pickle_results,
                        pickle_path=pickle_path,
                        mode = 'eval')                 
            else: break
        ### saving results of reconstruction and segmentation
        if data_form in [DataForm.PCD, DataForm.IMG]:
            save_results(results=results, 
                            data_form=data_form,
                            channel_dim=channel_dim,
                            recon_dir=eval_config.get('recon_dir', None),
                            seg_dir=eval_config.get('seg_dir', None),
                            is_conditional=is_conditional,
                            dataset_name=dataset_name,
                            save_target=save_target,
                            save_input=save_input,
                            save_colorized=save_colorized)
        eval_res = evaluate_results(results, evaluators, verbose = True)
        if len(eval_res) > 0:
            for eval_type, eval in eval_res.items():
                logger.info(f'{eval_type} evaluation: {eval}')
    else:
        ### directly evaluate files in the target path and prediction path
        logger.info("There is no dataloader, we will directly evaluate files in the target path and prediction path.")
        sub_dirs = eval_config.get('sub_dirs', ['.'])
        eval_res_list = []
        sample_num_list = []
        target_info = eval_config.get('target', None)
        assert target_info is not None, 'There is no information about target data.'
        target_path = target_info.get('path')
        target_suffix = target_info.get('suffix')
        load_target_data = get_load_function(target_suffix, is_mesh)[0]
        target_trans_list = target_info.get('transforms', [])
        target_transforms = get_transforms(target_trans_list, data_form = data_form)

        input_info = eval_config.get('input', None)
        if input_info is not None:
            input_path = input_info.get('path')
            input_suffix = input_info.get('suffix')
            input_trans_list = input_info.get('transforms', [])
            input_transforms = get_transforms(input_trans_list, data_form = data_form)
            load_input_data = get_load_function(input_suffix)[0]
        for sd in sub_dirs:
            results = []
            target_files = sorted(glob.glob(os.path.join(target_path, sd + "/*" + target_suffix)))
            sample_num_list.append(len(target_files))
            if eval_type == 'gen':
                ### for generation, the target is actually the input.
                pred_files = sorted(glob.glob(os.path.join(pred_path, sd + "/*" + pred_suffix)))
                pred_data = []
                for p in pred_files:
                    pred_data.append(pred_transforms(load_data(p)))
                
                target_data = []
                for t in target_files:
                    target_data.append(target_transforms(load_target_data(t)))
                if not is_mesh:
                    pred_data = np.stack(pred_data)
                    target_data = np.stack(target_data)
                results = [ {eval_type: pred_data, 'input': target_data}]
            else:
                batch_size = eval_config.get('batch_size', 16)
                ## for reconstruction and segmentation, target and prediction should be one-to-one correspondended
                for i in range(0, len(target_files), batch_size):
                    batch_target_files = target_files[i:i+batch_size]
                    batch_target_data = [ target_transforms(load_target_data(t)) for t in batch_target_files]
                    batch_pred_files = [ os.path.join(pred_path, sd, os.path.basename(t).replace(target_suffix, pred_suffix)) 
                                            for t in batch_target_files]
                    batch_pred_data = [ pred_transforms(load_data(p)) for p in batch_pred_files]
                    if not is_mesh:
                        batch_target_data = np.stack(batch_target_data)
                        batch_pred_data = np.stack(batch_pred_data)
                    res = {'target': batch_target_data, 
                            eval_type: batch_pred_data, 
                            'path': batch_target_files,
                            'input': batch_target_data}
                    ### load input data for visualization.
                    if input_info is not None:
                        batch_input_files = [ os.path.join(input_path, sd, os.path.basename(t).replace(target_suffix, input_suffix)) 
                                                for t in batch_target_files]
                        batch_input_data = [ input_transforms(load_input_data(p)) for p in batch_input_files]
                        if not is_mesh:
                            batch_input_data = np.stack(batch_input_data)
                        res['input'] = batch_input_data
                    
                    if len(results) < eval_batch_num:
                        process_results(results=results, 
                                res = res,
                                data_form = data_form,
                                pickle_results=pickle_results,
                                pickle_path=pickle_path,
                                mode = 'eval', skip_to_numpy=True)
                    else: break
                    ### save results to different sub dirs of the original recon/seg dir
                    if data_form in [DataForm.PCD, DataForm.IMG]:
                        recon_dir = eval_config.get('recon_dir', None)
                        seg_dir = eval_config.get('seg_dir', None)
                        if recon_dir is not None: recon_dir = os.path.join(recon_dir, sd)
                        if seg_dir is not None: seg_dir = os.path.join(seg_dir, sd)
                        save_results(results=results, 
                            data_form=data_form,
                            channel_dim=channel_dim,
                            recon_dir=recon_dir,
                            seg_dir=seg_dir,
                            save_target=save_target,
                            save_input=save_input,
                            save_colorized=save_colorized)
            eval_res = evaluate_results(results, evaluators, verbose = True)
            eval_res_list.append(eval_res)
            if len(eval_res) > 0:
                logger.info(f'{eval_type} evaluation' + (f' for sub dir ({sd})' if sd != '.' else '') + f': {eval_res[eval_type]}')
        
        if len(sub_dirs) > 1:   
            ### average results for all sub dirs
            per_sub_dir_eval_res = {}
            for eval_res in eval_res_list:
                for eval_type, eval in eval_res.items():
                    if eval_type not in per_sub_dir_eval_res:
                        per_sub_dir_eval_res[eval_type] = {}
                    for metric, value in eval.items():
                        if metric not in per_sub_dir_eval_res[eval_type]:
                            per_sub_dir_eval_res[eval_type][metric] = 0.0
                        per_sub_dir_eval_res[eval_type][metric] += value / len(sub_dirs)
            logger.info(f'Per subdir {eval_type} evaluation: {per_sub_dir_eval_res[eval_type]}')
            ### average results for all samples
            if not eval_type == 'gen':
                all_sample_num = sum(sample_num_list)
                per_sample_eval_res = {}
                for sid, eval_res in enumerate(eval_res_list):
                    for eval_type, eval in eval_res.items():
                        if eval_type not in per_sample_eval_res:
                            per_sample_eval_res[eval_type] = {}
                        for metric, value in eval.items():
                            if metric not in per_sample_eval_res[eval_type]:
                                per_sample_eval_res[eval_type][metric] = 0.0
                            per_sample_eval_res[eval_type][metric] += value * sample_num_list[sid] / all_sample_num
                logger.info(f'Per sample {eval_type} evaluation: {per_sample_eval_res[eval_type]}')
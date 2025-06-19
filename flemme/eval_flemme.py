#### This file is a script, and the code style sucks. 
import os
import torch
from .trainer_utils import *
from flemme.dataset import create_loader
from flemme.logger import get_logger


device = "cuda" if torch.cuda.is_available() else "cpu"
logger = get_logger('eval_flemme')

def main():
    with torch.no_grad():
        eval_config = load_config()
            
        #### create dataset and dataloader
        loader_config = eval_config.get('loader', None)
        save_target = eval_config.get('save_target', False)
        save_input = eval_config.get('save_input', False)
        save_colorized = eval_config.get('save_colorized', True)
        eval_type = eval_config.get('eval_type', 'segmentation')
        if eval_type == 'segmentation': eval_type = 'seg'
        elif eval_type == 'reconstruction': eval_type = 'recon'
        assert eval_type in ['seg', 'recon'], 'Currently we only support evaluation for segmentation and reconstruction.'
        num_classes = eval_config.get('num_classes', 2)
        pred_info = eval_config.get('prediction', None)
        assert pred_info is not None, 'There is no information about predictions.'
        
        pred_path = pred_info.get('path')
        pred_suffix = pred_info.get('suffix')    
        

        load_data = get_load_function(pred_suffix)[0]

        verbose = eval_config.get('verbose', False)
        eval_batch_num = eval_config.get('eval_batch_num', float('inf'))
        is_supervised = eval_config.get('is_supervised', True)
        is_conditional = eval_config.get('is_conditional', False)
        if save_target:
            logger.info('Save target for reconstruction and segmentation tasks.')
        if save_input:
            logger.info('Save input for reconstruction and segmentation tasks.')
        
        assert loader_config is not None, 'there is no dataloader for evaluation.'
        if 'data_path_list' in loader_config and len(loader_config['data_path_list']) > 1:
            logger.error('Currently we only support single dataset evaluation.')
            exit(1)
        loader_config['mode'] = 'test'
        loader = create_loader(loader_config)
        
        data_loader = loader['data_loader']
        
        results = {'input':[], 'target':[], 'condition':[], 'latent':[], 
                    'recon':[], 'seg':[], 'cluster':[], 'path':[]}
        logger.info('Finish loading data.')
        iter_id = 0
        data_form = loader['data_form']
        channel_dim = -1 if data_form == DataForm.PCD else 0
        input_suffix = loader_config['dataset'].get('data_suffix', 'png')
        if type(input_suffix) == tuple or type(input_suffix) == list:
            input_suffix = input_suffix[0]
        for t in tqdm(data_loader, desc="Predicting"):
            ### split x, y, path in later implementation.
            x, y, c, path = process_input(t)
            # print(x.shape, y.shape, len(path))
            ### move data to cuda
            if verbose:
                logger.info(f'We are at iter {iter_id}/{len(data_loader)}.')
            iter_id += 1
            if len(results['input']) < eval_batch_num:
                append_results(results=results, x = x, y = y, 
                                        res = None,
                                        path = path,
                                        data_form = data_form,
                                        is_supervised=is_supervised,
                                        is_conditional=is_conditional)
                res = []
                if type(path[0]) == tuple:
                    path = path[0]
                for p in path:
                    filename = os.path.basename(p).replace(input_suffix, pred_suffix)
                    tmp = load_data(os.path.join(pred_path, filename))
                    if type(tmp) == tuple: tmp = tmp[0]
                    if eval_type == 'seg':
                        if num_classes > 2:
                            tmp = label_to_onehot(tmp, channel_dim = channel_dim, num_classes = num_classes)
                        else:
                            # binary segmentation
                            tmp = tmp[None, :]
                        # ## old results
                        # if num_classes > 2:
                        #     tmp = label_to_onehot(tmp[0], channel_dim = channel_dim, num_classes = num_classes)
                        # else:
                        #     tmp = (tmp[None, :] > 0).astype(float)
                        
                    res.append(tmp)  
                res = np.stack(res)
                results[eval_type].append(res)                          
            else: break
        results = compact_results(results, data_form = data_form)    
        eval_metrics = eval_config.get('evaluation_metrics', None)
        if eval_metrics is not None:
            logger.info('evaluating the prediction accuracy ...')   
            evaluators = create_batch_evaluators(eval_metrics, data_form)
            eval_res = evaluate_results(results, evaluators, data_form)
            if len(eval_res) > 0:
                for eval_type, eval in eval_res.items():
                    logger.info(f'{eval_type} evaluation: {eval}')
        
        ### saving results of reconstruction and segmentation
        recon_dir = eval_config.get('recon_dir', None)

        if recon_dir is not None and len(results['recon']) > 0:
            logger.info(f'Saving reconstruction results to {recon_dir} ...')
            if not os.path.exists(recon_dir):
                os.makedirs(recon_dir)
            for idx, recon in enumerate(results['recon']):
                origin_path = results['path'][idx]
                if origin_path != '':
                    filename = os.path.basename(origin_path).split('.')[0]
                else:
                    filename = 'sample_{:03d}'.format(idx)
                output_path = os.path.join(recon_dir, filename)
                save_data(recon, data_form=data_form, output_path=output_path)
                if save_target:
                    target = results['target'][idx]
                    save_data(target, data_form=data_form, output_path=output_path+'_tar')
                if save_input:
                    input_x = results['input'][idx]
                    save_data(input_x, data_form=data_form, output_path=output_path+'_input')
        ### save segmentation
        seg_dir = eval_config.get('seg_dir', None)
        if seg_dir is not None and len(results['seg']) > 0:
            logger.info(f'Saving segmentation results to {seg_dir} ...')
            if not os.path.exists(seg_dir):
                os.makedirs(seg_dir)
            for idx, (data, seg, tar) in enumerate(zip(results['input'], results['seg'], results['target'])):
                origin_path = results['path'][idx]
                # print(origin_path)
                if origin_path != '':
                    filename = os.path.basename(origin_path).split('.')[0]
                else:
                    filename = 'sample_{:03d}'.format(idx)
                output_path = os.path.join(seg_dir, filename)
                
                ### transfer onehot to normal label for non-binary segmentation
                seg = onehot_to_label(seg, channel_dim=channel_dim, keepdim=True) if seg.shape[channel_dim] > 1 else seg
                tar = onehot_to_label(tar, channel_dim=channel_dim, keepdim=True) if tar.shape[channel_dim] > 1 else tar
                save_data(seg.astype(int), data_form=data_form, output_path=output_path, segmentation = True)
                ### save input
                if save_input:
                    save_data(data, data_form=data_form, output_path=output_path+'_input')
                ##### save target
                if save_target:
                    save_data(tar.astype(int), data_form=data_form, output_path=output_path + '_tar', segmentation = True)
                
                ### save colorized results
                if save_colorized:
                    #### save colorized pcd
                    if data_form == DataForm.PCD:
                        color = colorize_by_label(seg[..., 0])
                        cdata = (data, color)
                        save_data(cdata, data_form=data_form, output_path=output_path + '_colorized')
                        ### save colorized target
                        if save_target:
                            color = colorize_by_label(tar[..., 0])
                            cdata = (data, color)
                            save_data(cdata, data_form=data_form, output_path=output_path + '_colorized_tar')
                    #### save colorized img
                    if data_form == DataForm.IMG:                
                        cdata, raw_img = colorize_img_by_label(seg, data, gt = tar)
                        save_data(cdata, data_form=data_form, output_path=output_path + '_colorized')
                        if save_target:
                            cdata, _ = colorize_img_by_label(tar, data, gt = tar)
                            save_data(cdata, data_form=data_form, output_path=output_path + '_colorized_tar')
                        if save_input:
                            save_data(raw_img, data_form=data_form, output_path=output_path + '_input')


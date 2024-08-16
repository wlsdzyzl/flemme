#### This file is a script, and the code style sucks. 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .trainer_utils import *
from flemme.model import create_model
from flemme.dataset import create_loader
from flemme.logger import get_logger
from flemme.sampler import create_sampler

## if we want to train pcd or image, 
## make sure that the image size from data loader and image size from the model parameters are identical
device = "cuda" if torch.cuda.is_available() else "cpu"
logger = get_logger('test_flemme')

def main():
    with torch.no_grad():
        test_config = load_config()
        mode = test_config.get('mode', 'test')
        assert mode == 'test', "Wrong configuration for testing!"

        model_config = test_config.get('model', None)
        assert model_config is not None, "Model is not specified."
            
        ## For reproducibility
        rand_seed = test_config.get('rand_seed', None)
        if rand_seed is not None:
            logger.info('Set random seed manually.')
            torch.manual_seed(rand_seed)
            np.random.seed(rand_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(rand_seed)
        if test_config.get('determinstic', False):
            logger.info('Use determinstic algorithms, note that this may leads to performance decreasing.')
            if rand_seed is None:
                logger.warning('No random seed is specified, use 0 as random seed.')
                torch.manual_seed(0)
                np.random.seed(0)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        #### create model
        model = create_model(model_config)
        is_conditional = model.is_conditional
        is_supervising = model.is_supervising
        if is_supervising:
            logger.info('Supervising model, we will using label as target.')
        elif is_conditional:
            logger.info('Conditional model, we will using label as condition.')
        model_path = test_config.get('model_path', None)
        assert model_path is not None, "Model path is not specified."
        ## load check point
        load_checkpoint(model_path, model)
        model = model.to(device)
        ## turn to evaluation
        model.eval()
        #### create dataset and dataloader
        loader_config = test_config.get('loader', None)
        eval_gen_config = test_config.get('eval_gen', None)
        eval_batch_num = test_config.get('eval_batch_num', float('inf'))
        save_target = test_config.get('save_target', False)
        save_input = test_config.get('save_input', False)
        save_colorized = test_config.get('save_colorized', True)
        verbose = test_config.get('verbose', False)
        if save_target:
            logger.info('Save target for reconstruction and segmentation tasks.')
        if save_input:
            logger.info('Save input for reconstruction and segmentation tasks.')
        if loader_config is not None:
            if not 'mode' in loader_config:
                loader_config['mode'] = mode
            loader = create_loader(loader_config)
            assert model.data_form == loader['data_form'], 'Inconsistent data forms for model and loader.'
            
            data_loader = loader['data_loader']
            
            results = {'input':[], 'target':[], 'condition':[], 'latent':[], 
                       'recon':[], 'seg':[], 'cluster':[], 'path':[]}
            logger.info('Finish loading data.')
            iter_id = 0
            for t in data_loader:
                ### split x, y, path in later implementation.
                x, y, path = t
                # print(x.shape, y.shape, len(path))
                ### move data to cuda
                x, y = x.to(device).float(), y.to(device)
                if not x.shape[1:] == tuple(model.get_input_shape()):
                    logger.error("Inconsistent sample shape between data and model")
                    exit(1)   
                ### here we want to generate raw image
                res = forward_pass(model, x, y)
                if verbose:
                    logger.info(f'We are at iter {iter_id}/{len(data_loader)}.')
                iter_id += 1
                if len(results['input']) < eval_batch_num:
                    append_results(results=results, x = x, y = y, 
                                            path = path, res = res, data_form = model.data_form,
                                            is_supervising=is_supervising,
                                            is_conditional=is_conditional)
                else: break
            results = compact_results(results)    
            eval_metrics = test_config.get('evaluation_metrics', None)
            if eval_metrics is not None:
                logger.info('evaluating the prediction accuracy ...')   
                evaluators = create_batch_evaluators(eval_metrics, model.data_form)
                eval_res = evaluate_results(results, evaluators)
                if len(eval_res) > 0:
                    for eval_type, eval in eval_res.items():
                        logger.info(f'{eval_type} evaluation: {eval}')
            
            ### saving results of reconstruction and segmentation
            recon_dir = test_config.get('recon_dir', None)

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
                    save_data(recon, data_form=model.data_form, output_path=output_path)
                    if save_target:
                        target = results['target'][idx]
                        save_data(target, data_form=model.data_form, output_path=output_path+'_tar')
                    if save_input:
                        input_x = results['input'][idx]
                        save_data(input_x, data_form=model.data_form, output_path=output_path+'_input')
            ### save segmentation
            seg_dir = test_config.get('seg_dir', None)
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
                    channel_dim = -1 if model.data_form == DataForm.PCD else 0
                    
                    ### transfer onehot to normal label for non-binary segmentation
                    seg = onehot_to_label(seg, channel_dim=channel_dim, keepdim=True) if seg.shape[channel_dim] > 1 else seg
                    tar = onehot_to_label(tar, channel_dim=channel_dim, keepdim=True) if tar.shape[channel_dim] > 1 else tar
                    save_data(seg.astype(int), data_form=model.data_form, output_path=output_path, segmentation = True)
                    ### save input
                    if save_input:
                        save_data(data, data_form=model.data_form, output_path=output_path+'_input')
                    ##### save target
                    if save_target:
                        save_data(tar.astype(int), data_form=model.data_form, output_path=output_path + '_tar', segmentation = True)
                    
                    ### save colorized results
                    if save_colorized:
                        #### save colorized pcd
                        if model.data_form == DataForm.PCD:
                            color = colorize_by_label(seg[..., 0])
                            cdata = (data, color)
                            save_data(cdata, data_form=model.data_form, output_path=output_path + '_colorized')
                            ### save colorized target
                            if save_target:
                                color = colorize_by_label(tar[..., 0])
                                cdata = (data, color)
                                save_data(cdata, data_form=model.data_form, output_path=output_path + '_colorized_tar')
                        #### save colorized img
                        if model.data_form == DataForm.IMG:                
                            cdata, raw_img = colorize_img_by_label(seg, data, gt = tar)
                            save_data(cdata, data_form=model.data_form, output_path=output_path + '_colorized')
                            if save_target:
                                cdata, _ = colorize_img_by_label(tar, data, gt = tar)
                                save_data(cdata, data_form=model.data_form, output_path=output_path + '_colorized_tar')
                            if save_input:
                                save_data(raw_img, data_form=model.data_form, output_path=output_path + '_input')
            cond = y[0]
        else:
            logger.warning('loader_config is None, reconstruction, segmentation, conditional generation and interpolation will be ignored.')
            cond = None

        if model.is_generative and eval_gen_config is not None:
            sampler = None
            sampler_config = eval_gen_config.get('sampler', None)
            if sampler_config is not None:
                sampler = create_sampler(model=model, device = device, sampler_config = sampler_config)
            else:
                logger.error("Sampler is not specified for generation.")
                exit(1)
            ######## generate new samples -------------------------------------------
            gen_dir = eval_gen_config.get('gen_dir', None)
            if gen_dir is not None:
                if not os.path.exists(gen_dir):
                    os.makedirs(gen_dir)
                #### if interpolation is set, save the interpolation results
                inter_config = eval_gen_config.get('interpolation', None)
                ### need loader
                if inter_config is not None:
                    logger.info('Generating new samples through Interpolations ...')
                    group_num = inter_config.get('group_num', 2)
                    corner_num = inter_config.get('corner_num', 2)
                    inter_num = inter_config.get('inter_num', 8)
                    if corner_num != 2 and corner_num != 4:
                        logger.warning('number of corners should be 2 or 4. Reset corner_num to 2.')
                        corner_num = 2
                    for gid in range(group_num):
                        ### interpolating the latent embeddings
                        if loader_config is not None:
                            corner_latents = torch.from_numpy(results['latent'][gid * corner_num: gid * corner_num + corner_num])
                        else:
                            corner_latents = None
                        if is_conditional:
                            x_bar = sampler.interpolate(corner_latents = corner_latents, corner_num=corner_num, inter_num=inter_num, cond = cond)
                        else:
                            x_bar = sampler.interpolate(corner_latents = corner_latents, corner_num=corner_num, inter_num=inter_num)
                        ## save the images.
                        for iid, _x_bar in enumerate(x_bar):
                            x_bar_np = _x_bar.cpu().detach().numpy()
                            output_path = os.path.join(gen_dir, 'gen_inter_group{:02d}_{:02d}'.format(gid, iid))
                            save_data(x_bar_np, model.data_form, output_path)
                        if model.data_form == DataForm.IMG and model.encoder.dim == 2:
                            x_bar = F.interpolate(x_bar, size=(32, 32))
                            save_path = os.path.join(gen_dir, 'gen_inter_group{:02d}.png'.format(gid))
                            inter_figure = combine_figures(figs=x_bar.cpu().detach().numpy(), row_length=inter_num + 2)
                            save_img(save_path, (inter_figure * 255).astype('uint8'))

                #### if random sample numer is not 0, save the generation from random noise.
                random_sample_num = eval_gen_config.get('random_sample_num', 8)
                if random_sample_num > 0:
                    logger.info('Gererating new samples from random noise ...')
                    ### ddpm
                    if is_conditional:                        
                        x_bar = sampler.generate_rand(n=random_sample_num, cond = cond)
                    else:
                        x_bar = sampler.generate_rand(n=random_sample_num)
                    ## save the images.
                    for iid, _x_bar in enumerate(x_bar):
                        x_bar_np = _x_bar.cpu().detach().numpy()
                        output_path = os.path.join(gen_dir, 'gen_rand_{:02d}'.format(iid))
                        save_data(x_bar_np, model.data_form, output_path)
                    if model.data_form == DataForm.IMG and \
                            len(model.get_input_shape()) == 3:
                        x_bar = F.interpolate(x_bar, size=(32, 32))
                        save_path = os.path.join(gen_dir, 'gen_rand.png')
                        inter_figure = combine_figures(figs=x_bar.cpu().detach().numpy(), row_length=int(random_sample_num ** 0.5))
                        save_img(save_path, (inter_figure * 255).astype('uint8'))

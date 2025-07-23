#### This file is a script, and the code style sucks. 
from .trainer_utils import *
import torch.nn.functional as F
import math
from flemme.model import create_model, ClM
from flemme.dataset import create_loader
from flemme.logger import get_logger
from flemme.sampler import create_sampler
from tqdm import tqdm


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
            logger.warning('Use determinstic algorithms, note that this may leads to performance decreasing.')
            logger.warning('You may need to set number of worker to 0 to avoid concurrent data loading.')
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
        is_supervised = model.is_supervised
        if is_supervised:
            logger.info('Supervising model, we will use label as target.')
        elif is_conditional:
            logger.info('Conditional model, label may be used as condition.')
        model_path = test_config.get('model_path', None)
        assert model_path is not None, "Model path is not specified."

        ignore_mismatched_keys = test_config.get('ignore_mismatched_keys', [])
        ## load check point
        load_checkpoint(model_path, model, ignore_mismatched_keys = ignore_mismatched_keys)

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
        channel_dim = -1 if model.data_form == DataForm.PCD else 0
        if save_target:
            logger.info('Save target for reconstruction and segmentation tasks.')
        if save_input:
            logger.info('Save input for reconstruction and segmentation tasks.')
        dataset_name = None
        if loader_config is not None:
            if not 'mode' in loader_config:
                loader_config['mode'] = mode
            dataset_name = loader_config.get('dataset').get('name')
            loader = create_loader(loader_config)
            assert model.data_form == loader['data_form'], 'Inconsistent data forms for model and loader.'
            
            data_loader = loader['data_loader']
            logger.info('Finish loading data.')

            results = []

            custom_save_results = test_config.get('custom_save_results', [])
            custom_res_names = []
            custom_res_dirs = []
            for res in custom_save_results:
                custom_res_names.append(res.get('name'))
                custom_res_dirs.append(res.get('dir'))

            pickle_results = test_config.get('pickle_results', False)
            pickle_path = test_config.get('pickle_path', 'pickled')
            if pickle_results:
                mkdirs(pickle_path)

            iter_id = 0
            for t in tqdm(data_loader, desc="Predicting"):
                ### split x, y, path in later implementation.
                x, y, c, slice_indices, path = process_input(t)
                x = x.to(device).float() 
                if y is not None: 
                    y = y.to(device)
                if c is not None:
                    c = c.to(device)
                ### move data to cuda
                if not x.shape[1:] == tuple(model.get_input_shape()):
                    logger.error("Inconsistent sample shape between data and model")
                    exit(1)   
                ### here we want to generate raw image
                res = forward_pass(model, x, y, c)
                iter_id += 1
                
                if len(results) < eval_batch_num:
                    process_results(results=results, x = x, y = y, c = c,
                                            path = path, 
                                            slice_indices = slice_indices,
                                            res = res, 
                                            data_form = model.data_form,
                                            is_supervised=is_supervised,
                                            is_conditional=is_conditional,
                                            additional_keys = custom_res_names,
                                            pickle_results=pickle_results, 
                                            pickle_path=pickle_path)
                else: break
            if slice_indices is not None:
                results = merge_patches_in_results(results=results, 
                                                pickle_results=pickle_results, pickle_path=pickle_path)
            # results = compact_results(results, data_form = model.data_form,
            #                         additional_keys = custom_res_names)    

            eval_metrics = test_config.get('evaluation_metrics', None)
            if eval_metrics is not None:
                logger.info('evaluating the prediction accuracy ...')   
                evaluators = create_batch_evaluators(eval_metrics, model.data_form)
                eval_res = evaluate_results(results, evaluators, data_form = model.data_form, verbose = True)
                if len(eval_res) > 0:
                    for eval_type, eval in eval_res.items():
                        logger.info(f'{eval_type} evaluation: {eval}')
            tsne_config = test_config.get('tsne_visualization', None)
            if tsne_config:
                label_names = tsne_config.get('label_names', None)
                tsne_path = tsne_config.pop('tsne_path', './tsne.png')
                ## one matrix to one vector
                one_to_one = tsne_config.pop('one_to_one', True)
                if type(label_names) == str:
                    from flemme.dataset.label_dict import get_label_cls
                    label_names = list(get_label_cls(label_names).values())
                    tsne_config['label_names'] = label_names
                latents = extract_results(results, 'latent')
                ### get the label
                labels = None
                ### classification model
                if isinstance(model, ClM):
                    labels = extract_results(results, 'target')
                else:
                    labels = extract_results(results, 'condition')
                if latents is None:
                    logger.warning('There is no latent for tsne visualization.')
                elif labels is None or not labels.ndim == 2:
                    logger.warning('There is no label (should be one-hot classification label) for tsne visualization.')
                else:
                    ### process latents and labels for tsne visualization
                    ## latent is not a vector
                    repeat = 1
                    if not latents.ndim == 2:
                        logger.info('Latents will be reshaped to vector embeddings for TSNE visualization.')
                        if one_to_one:
                            ### one matrix to one vector
                            latents = latents.reshape((latents.shape[0], -1))
                        else:
                            if model.feature_channel_dim == 1:
                                if latents.ndim == 3:
                                    latents = latents.transpose((0, 2, 1))
                                elif latents.ndim == 4:
                                    latents = latents.transpose((0, 2, 3, 1))
                                elif latents.ndim == 5:
                                    latents = latents.transpose((0, 2, 3, 4, 1))
                            repeat = math.prod(latents.shape[1:-1][:1])
                            latents = latents.reshape((latents.shape[0] * repeat, -1))
                    ### onehot to label
                    labels = onehot_to_label(labels, channel_dim=-1)
                    labels = np.repeat(labels, repeat)
                    ### tsne
                    tsne_fig = construct_tsne_vis(latents, labels = labels, **tsne_config)
                    save_img(tsne_path, tsne_fig)

            ### saving results of reconstruction and segmentation
            recon_dir = test_config.get('recon_dir', None)

            if recon_dir is not None:
                logger.info(f'Saving reconstruction results to {recon_dir} ...')
                mkdirs(recon_dir)
                sample_idx = 0
                for res_dict in tqdm(results):
                    res_dict = load_pickle(res_dict)
                    for idx, recon in enumerate(res_dict['recon']):
                        origin_path = res_dict['path'][idx]
                        if origin_path != '':
                            filename = os.path.basename(origin_path).split('.')[0]
                        else:
                            filename = 'sample_{:03d}'.format(sample_idx)
                        class_name = None
                        if is_conditional and ('ClassLabel' in dataset_name or 'Cls' in dataset_name):
                            if is_supervised:
                                class_name = origin_path.split('/')[-3]
                            else:
                                class_name = origin_path.split('/')[-2]
                        if class_name:
                            output_dir = os.path.join(recon_dir, class_name)
                            mkdirs(output_dir)
                            output_path = os.path.join(output_dir, filename)
                        else:
                            output_path = os.path.join(recon_dir, filename)
                        save_data(recon, data_form=model.data_form, output_path=output_path)
                        if save_target:
                            target = res_dict['target'][idx]
                            save_data(target, data_form=model.data_form, output_path=output_path+'_tar')
                        if save_input:
                            input_x = res_dict['input'][idx]
                            save_data(input_x, data_form=model.data_form, output_path=output_path+'_input')
                        sample_idx += 1
            ### save segmentation
            seg_dir = test_config.get('seg_dir', None)
            if seg_dir is not None:
                logger.info(f'Saving segmentation results to {seg_dir} ...')
                mkdirs(seg_dir)
                sample_idx = 0
                for res_dict in tqdm(results):
                    res_dict = load_pickle(res_dict)
                    for idx, (data, seg, tar) in enumerate(zip(res_dict['input'], res_dict['seg'], res_dict['target'])):
                        origin_path = res_dict['path'][idx]
                        if origin_path != '':
                            filename = os.path.basename(origin_path).split('.')[0]
                        else:
                            filename = 'sample_{:03d}'.format(sample_idx)
                        output_path = os.path.join(seg_dir, filename)
                        
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
                        sample_idx += 1

            for res_name, res_dir in zip(custom_res_names, custom_res_dirs):
                res_name = res.get('name')
                res_dir = res.get('dir', None)
                if res_dir is not None:
                    logger.info(f'Saving {res_name} results to {res_dir} ...')
                    mkdirs(res_dir)
                    sample_idx = 0
                    for res_dict in tqdm(results):
                        res_dict = load_pickle(res_dict)
                        for idx, res_data in enumerate(res_dict[res_name]):
                            origin_path = res_dict['path'][idx]
                            if origin_path != '':
                                filename = os.path.basename(origin_path).split('.')[0]
                            else:
                                filename = 'sample_{:03d}'.format(sample_idx)
                            output_path = os.path.join(res_dir, filename)
                            save_data(res_data, data_form=model.data_form, output_path=output_path)
                            sample_idx += 1
                        if save_target and len(results['target']) > 0:
                            target = results['target'][idx]
                            save_data(target, data_form=model.data_form, output_path=output_path+'_tar')
                        if save_input and len(results['input']) > 0:
                            input_x = results['input'][idx]
                            save_data(input_x, data_form=model.data_form, output_path=output_path+'_input')
            ## for generation with fixed condition
            input_x = x[0]
            if y is not None:
                input_y = y[0]
            if c is not None:
                input_c = c[0]
        else:
            logger.warning('loader_config is None, reconstruction, segmentation, conditional generation and interpolation will be ignored.')
            cond = None

        if model.is_generative and eval_gen_config is not None:
            sampler = None
            sampler_config = eval_gen_config.get('sampler', {'name':'NormalSampler'})
            if sampler_config:
                sampler = create_sampler(model=model, sampler_config = sampler_config)
            else:
                logger.error("Sampler is not specified for generation.")
                exit(1)
            ######## generate new samples -------------------------------------------
            gen_dir = eval_gen_config.get('gen_dir', None)
            if gen_dir is not None:
                mkdirs(gen_dir)
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
                            latents = extract_results(results, 'latent')
                            corner_latents = torch.from_numpy(latents[gid * corner_num: gid * corner_num + corner_num])
                        else:
                            corner_latents = None
                        if is_conditional or is_supervised:
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
                    if is_conditional and is_supervised:
                        ### this should be happened:
                        logger.error("Currently conditional and supervised generative model is not supported.")
                        exit(1)
                        # x_bar = sampler.generate_rand(n=random_sample_num, cond = input_c)
                    elif is_conditional:
                        x_bar = sampler.generate_rand(n=random_sample_num, cond = input_y)
                    elif is_supervised:                      
                        x_bar = sampler.generate_rand(n=random_sample_num, cond = input_x)
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

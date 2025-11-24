#### This file is a script, and the code style sucks. 
from .trainer_utils import *
from flemme.model import ClM
from flemme.logger import get_logger
from tqdm import tqdm
from glob import glob
from flemme.augment import get_transforms
## if we want to train pcd or image, 
## make sure that the image size from data loader and image size from the model parameters are identical
device = "cuda" if torch.cuda.is_available() else "cpu"
logger = get_logger('trainer.test')

def save_results(results, 
                 data_form,
                 channel_dim,
                 recon_dir = None, 
                 seg_dir = None, 
                 save_data_fn = save_data,
                 custom_res_names = [], 
                 custom_res_dirs = [],
                 is_conditional = False,
                 dataset_name = '',
                 save_target = False,
                 save_input = False,
                 save_colorized = False,
                 ):

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
                    class_name = origin_path.split('/')[-2]
                    output_dir = os.path.join(recon_dir, class_name)
                    mkdirs(output_dir)
                    output_path = os.path.join(output_dir, filename)
                else:
                    output_path = os.path.join(recon_dir, filename)
                save_data_fn(recon, data_form=data_form, output_path=output_path)
                if save_target:
                    target = res_dict['target'][idx]
                    save_data_fn(target, data_form=data_form, output_path=output_path+'_tar')
                if save_input:
                    input_x = res_dict['input'][idx]
                    save_data_fn(input_x, data_form=data_form, output_path=output_path+'_input')
                sample_idx += 1
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

                class_name = None
                if is_conditional and ('ClassLabel' in dataset_name or 'Cls' in dataset_name):
                    class_name = origin_path.split('/')[-2]
                    output_dir = os.path.join(seg_dir, class_name)
                    mkdirs(output_dir)
                    output_path = os.path.join(output_dir, filename)
                else:
                    output_path = os.path.join(seg_dir, filename)
                
                ### transfer onehot to normal label for non-binary segmentation
                seg = onehot_to_label(seg, channel_dim=channel_dim, keepdim=True) if seg.shape[channel_dim] > 1 else seg
                tar = onehot_to_label(tar, channel_dim=channel_dim, keepdim=True) if tar.shape[channel_dim] > 1 else tar
                save_data_fn(seg.astype(int), data_form=data_form, output_path=output_path, segmentation = True)
                ### save input
                if save_input:
                    save_data_fn(data, data_form=data_form, output_path=output_path+'_input')
                ##### save target
                if save_target:
                    save_data_fn(tar.astype(int), data_form=data_form, output_path=output_path + '_tar', segmentation = True)
                
                ### save colorized results
                if save_colorized:
                    #### save colorized pcd
                    if data_form == DataForm.PCD:
                        color = colorize_by_label(seg[..., 0])
                        cdata = (data, color)
                        save_data_fn(cdata, data_form=data_form, output_path=output_path + '_colorized')
                        ### save colorized target
                        if save_target:
                            color = colorize_by_label(tar[..., 0])
                            cdata = (data, color)
                            save_data_fn(cdata, data_form=data_form, output_path=output_path + '_colorized_tar')
                    #### save colorized img
                    if data_form == DataForm.IMG:                
                        cdata, raw_img = colorize_img_by_label(seg, data, gt = tar)
                        save_data_fn(cdata, data_form=data_form, output_path=output_path + '_colorized')
                        if save_target:
                            cdata, _ = colorize_img_by_label(tar, data, gt = tar)
                            save_data_fn(cdata, data_form=data_form, output_path=output_path + '_colorized_tar')
                        if save_input:
                            save_data_fn(raw_img, data_form=data_form, output_path=output_path + '_input')
                sample_idx += 1

    for res_name, res_dir in zip(custom_res_names, custom_res_dirs):
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

                    class_name = None
                    if is_conditional and ('ClassLabel' in dataset_name or 'Cls' in dataset_name):
                        class_name = origin_path.split('/')[-2]
                        output_dir = os.path.join(res_dir, class_name)
                        mkdirs(output_dir)
                        output_path = os.path.join(output_dir, filename)
                    else:
                        output_path = os.path.join(res_dir, filename)
                    save_data_fn(res_data, data_form=data_form, output_path=output_path)
                    sample_idx += 1
                if save_target and len(results['target']) > 0:
                    target = results['target'][idx]
                    save_data_fn(target, data_form=data_form, output_path=output_path+'_tar')
                if save_input and len(results['input']) > 0:
                    input_x = results['input'][idx]
                    save_data_fn(input_x, data_form=data_form, output_path=output_path+'_input')

def test_run(model, t):
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
    res['input'] = x
    res['path'] = path
    if model.is_supervised:
        res['target'] = y
    if model.is_supervised and model.is_conditional:
        res['condition'] = c
    elif model.is_conditional:
        res['condition'] = y
    if slice_indices:
        res['slice_indices'] = slice_indices
    return res
def test(test_config,
        create_model_fn,
        create_loader_fn,
        create_sampler_fn,
        run_fn,
        save_data_fn):
    with torch.no_grad():
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
        model_name = model_config.get('name')
        model = create_model_fn(model_config)
        is_conditional = model.is_conditional
        is_supervised = model.is_supervised
        if is_supervised:
            logger.info('Supervising model, we will use label as target.')
        elif is_conditional:
            logger.info('Conditional model, label may be used as condition.')
        model_path = test_config.get('model_path', None)
        if model_path is None:
            logger.info('You are testing a model that has never been trained before.')
        else:
            ignore_mismatched_keys = test_config.get('ignore_mismatched_keys', None)
            loaded = True
            if type(model_path) == str:
                if os.path.isfile(model_path):
                    load_checkpoint(model_path, model, 
                                    ignore_mismatched_keys = ignore_mismatched_keys)
                else:
                    loaded = False
            else:
                assert type(model_path) == dict, \
                    "model_path should be a model path (str) or pathes of sub-models (dict)."
                assert ignore_mismatched_keys is None or type(ignore_mismatched_keys) == dict, \
                    "ignore_mismatched_keys should be a dict when model_path is a dict."
                
                for key in model_path.keys():
                    hasattr(model, key), \
                        "Unknown submodel {} for model {}".format(key, model_name)
                    if os.path.isfile(model_path[key]):
                        load_checkpoint(model_path[key], getattr(model, key), 
                                        ignore_mismatched_keys=ignore_mismatched_keys.get(key, None) if ignore_mismatched_keys else None)
                    else:
                        loaded = False
            if not loaded:
                logger.error('Trained model doesn\'t exist. You are testing a model that has never been trained before.')
                exit(1)
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
            loader = create_loader_fn(loader_config)
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
            pickle_path = test_config.get('pickle_path', 'flemme-pickled')
            load_pickle_results = test_config.get('load_pickle_results', False)
            if load_pickle_results:
                logger.info('Loading tmp results from pickled files to skip predicting (You need to make sure the pickled files are what you want). ')
                results = sorted(glob(os.path.join(pickle_path, "*.pkl")))
                pickle_results = True
                assert len(results) == len(data_loader), "mismatched number of results and data loader, please check your pickled files."
            else:
                ### test pickled results might be used later
                if pickle_results:
                    rkdirs(pickle_path)
                iter_id = 0
                for t in tqdm(data_loader, desc="Predicting"):
                    res = run_fn(model, t)
                    iter_id += 1
                    if len(results) < eval_batch_num:
                        process_results(results=results, 
                                            res = res, 
                                            data_form = model.data_form,
                                            additional_keys = custom_res_names,
                                            pickle_results=pickle_results, 
                                            pickle_path=pickle_path,
                                            mode = mode)
                    else: break
                if 'slice_indices' in results:
                    results = merge_patches_in_results(results=results, 
                                                    pickle_results=pickle_results, pickle_path=pickle_path)
            ### evaluate the model outputs in batch form, saving these results is not necessary.
            eval_metrics = test_config.get('evaluation_metrics', None)
            if eval_metrics is not None:
                logger.info('evaluating the prediction accuracy ...')
                evaluators = create_batch_evaluators(eval_metrics, model.data_form)
                eval_res = evaluate_results(results, evaluators, data_form=model.data_form, verbose=True)
                if len(eval_res) > 0:
                    for eval_type, eval in eval_res.items():
                        logger.info(f'{eval_type} evaluation: {eval}')
            tsne_config = test_config.get('tsne_visualization', None)
            if tsne_config:
                label_names = tsne_config.get('label_names', None)
                tsne_path = tsne_config.pop('tsne_path', './tsne.png')
                logger.info(f'Use t-SNE algorithm to visualize the latents, the resulting figure will be saved to {tsne_path}.')
                ## one matrix to one vector
                one_to_multiple = tsne_config.pop('one_to_multiple', 1)
                noise_level = tsne_config.pop('noise_level', 0)
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
                    if not latents.ndim == 2:
                        logger.warning('Latents will be compact to vector embeddings for TSNE visualization.')
                        if model.feature_channel_dim == 1:
                            latents = latents.reshape(latents.shape[0], latents.shape[1], -1).mean(axis = -1)
                        else:
                            latents = latents.reshape(latents.shape[0], -1, latents.shape[-1]).mean(axis = 1)
                    latents = normalize(latents, channel_dim=-1)
                    labels = onehot_to_label(labels, channel_dim=-1)
                    ### We add noise to the latents so that we could have more points just for visualization.
                    if one_to_multiple > 1:
                        latents = np.repeat(latents, one_to_multiple, axis=0)
                        labels = np.repeat(labels, one_to_multiple)
                    if noise_level > 0:
                        latents = latents + np.random.normal(0, noise_level, latents.shape)
                    ### tsne visualization
                    tsne_fig = construct_tsne_vis(latents, labels = labels, **tsne_config)
                    save_img(tsne_path, tsne_fig)

            ### saving results of reconstruction and segmentation
            save_results(results=results, 
                         data_form=model.data_form,
                         save_data_fn=save_data_fn,
                         channel_dim=channel_dim,
                         recon_dir=test_config.get('recon_dir', None),
                         seg_dir=test_config.get('seg_dir', None),
                         custom_res_names=custom_res_names,
                         custom_res_dirs=custom_res_dirs,
                         is_conditional=is_conditional,
                         dataset_name=dataset_name,
                         save_target=save_target,
                         save_input=save_input,
                         save_colorized=save_colorized)

        if model.is_generative and eval_gen_config is not None:
            ## evaluation metrics for generation are computed separately from other tasks
            sampler = None
            sampler_config = eval_gen_config.get('sampler', {'name':'NormalSampler'})
            if sampler_config:
                sampler = create_sampler_fn(model=model, sampler_config = sampler_config)
            else:
                logger.error("Sampler is not specified for generation.")
                exit(1)
            ######## generate new samples -------------------------------------------
            gen_dir = eval_gen_config.get('gen_dir', None)
            if gen_dir is not None:
                ### condition for generation
                ### we will 
                condition = eval_gen_config.get('conditions', None)
                if type(condition) == dict:
                    condition_name, condition = condition.keys(), condition.values()
                elif type(condition) == list:
                    condition_name, condition = condition, condition
                else:
                    ## transfer to list
                    condition_name, condition = [condition, ], [condition, ]
                cond_trans_list = eval_gen_config.get('condition_transforms', [])
                condition_transforms = get_transforms(cond_trans_list, data_form = model.data_form)
                for cond_n, cond in zip(condition_name, condition):
                    cond = condition_transforms(cond).to(device) if not cond is None else cond
                    output_dir = os.path.join(gen_dir, cond_n) if not cond_n is None else gen_dir
                    mkdirs(output_dir)
                    inter_config = eval_gen_config.get('interpolation', None)
                    ### need loader
                    if inter_config is not None:
                        logger.info('Generating new samples through Interpolations {}...'.format('with condition {}'.format(cond_n) if not cond_n is None else ''))
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
                            x_bar = x_bar.cpu().detach().numpy()
                            for iid, _x_bar in enumerate(x_bar):
                                output_path = os.path.join(output_dir, 'gen_inter_group{:02d}_{:02d}'.format(gid, iid))
                                save_data_fn(_x_bar, model.data_form, output_path)
                            if model.data_form == DataForm.IMG and model.encoder.dim == 2:
                                save_path = os.path.join(output_dir, 'gen_inter_group{:02d}.png'.format(gid))
                                inter_figure = combine_figures(figs=x_bar, row_length=inter_num + 2)
                                save_img(save_path, (inter_figure * 255).astype('uint8'))

                    #### if random sample numer is not 0, save the generation from random noise.
                    random_sample_num = eval_gen_config.get('random_sample_num', 8)
                    if random_sample_num > 0:
                        logger.info('Generating new samples from random noise {}...'.format(f'for condition {cond_n} ' if not cond_n is None else ''))
                        ### ddpm
                        if is_conditional or is_supervised:
                            x_bar = sampler.generate_rand(n=random_sample_num, cond = cond)
                        else:
                            x_bar = sampler.generate_rand(n=random_sample_num)
                        x_bar = x_bar.cpu().detach().numpy()
                        ## save the images.
                        for iid, _x_bar in tqdm(enumerate(x_bar), desc='Saving', total=len(x_bar)):
                            output_path = os.path.join(output_dir, 'gen_rand_{:02d}'.format(iid))
                            save_data_fn(_x_bar, model.data_form, output_path)
                        if model.data_form == DataForm.IMG and \
                                len(model.get_input_shape()) == 3:
                            save_path = os.path.join(output_dir, 'gen_rand.png')
                            inter_figure = combine_figures(figs=x_bar, row_length=int(random_sample_num ** 0.5))
                            save_img(save_path, (inter_figure * 255).astype('uint8'))

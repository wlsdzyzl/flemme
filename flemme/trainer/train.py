from .trainer_utils import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from flemme.logger import get_logger
from flemme.dataset import random_split_dataloader
from datetime import datetime
from tqdm import tqdm


logger = get_logger('trainer.train')
## if we want to train pcd or image, 
## make sure that the image size from data loader and image size from the model parameters are identical
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_run(model, t, only_forward = False):
    processed_input = process_input(t)
    x, y, c = processed_input[0], processed_input[1], processed_input[2]
    x = x.to(device).float() 
    if y is not None: 
        y = y.to(device)
    if c is not None:
        c = c.to(device)
    if not x.shape[1:] == tuple(model.get_input_shape()):
        logger.error("Inconsistent sample shape between data and model: {} and {}".format(x.shape[1:], tuple(model.get_input_shape())))
        exit(1)  
    res = {'input': x, }
    if model.is_supervised:
        res['target'] = y
    if model.is_supervised and model.is_conditional:
        res['condition'] = c
    elif model.is_conditional:
        res['condition'] = y
    ### here we want to generate raw image
    if only_forward:
        ## model.forward
        tmp_res = forward_pass(model, x, y, c)
        res.update(tmp_res)
        return res 
    else:
        ## model.compute_loss
        losses, tmp_res = compute_loss(model, x, y, c)
        res.update(tmp_res)
        return losses, res

def train(train_config, 
        create_model_fn,
        create_loader_fn,
        create_sampler_fn,
        run_fn):
    mode = train_config.get('mode', 'train')
    assert mode == 'train', "Wrong configuration for training!"

    model_config = train_config.get('model', None)
    assert model_config is not None, "Model is not specified."

    ## For reproducibility
    rand_seed = train_config.get('rand_seed', None)
    if rand_seed is not None:
        logger.info('Set random seed manually.')
        torch.manual_seed(rand_seed)
        np.random.seed(rand_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(rand_seed)
    if train_config.get('determinstic', False):
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

    detect_anomaly = train_config.get('detect_anomaly', False)
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    loader_config = train_config.get('loader', None)
    assert loader_config is not None, "Data loader is not speficied"
    if not 'mode' in loader_config:
        loader_config['mode'] = mode

    val_loader_config = train_config.get('val_loader', None)
    val_split_ratio = train_config.get('val_split_ratio', 0.0)
    val_split_seed = train_config.get('val_split_seed', None)
    ckp_dir = train_config.get('check_point_dir', '.')
    #### use writer to visualize the training process
    writer = SummaryWriter(log_dir = os.path.join(ckp_dir, 'logs'))
    custom_write_results = train_config.get('custom_write_results', [])
    #### create model
    model_name = model_config.get('name')
    model = create_model_fn(model_config)

    logger.info("model info:\n{}".format(model))
    
    is_conditional = model.is_conditional
    is_supervised = model.is_supervised

    eval_metrics = train_config.get('evaluation_metrics', {})
    evaluators = create_batch_evaluators(eval_metrics, model.data_form)
    
    ### use eval to compute the score and select the best model
    score_metric = None
    score_metric_config = train_config.get('score_metric', None)
    if evaluators is not None and score_metric_config is not None:
        score_metric = score_metric_config.get('name')
        higher_is_better = score_metric_config.get('higher_is_better', True)
        assert sum([score_metric in e for e in evaluators.values()]) > 0, "Score metric is not in the evaluation metrics."
    
    if is_supervised:
        logger.info('Supervising model, we will use label as target.')
    elif is_conditional:
        logger.info('Conditional model, we will using label as condition.')

    loss_names = model.get_loss_name()
    logger.info('Using loss(es): {}'.format(loss_names))
    logger.info("Total number of model parameters: {}".format(sum(p.numel() for p in model.parameters())))
    #### create dataset and dataloader
    loader = create_loader_fn(loader_config)
    assert model.data_form == loader['data_form'], \
        'Inconsistent data forms for model and loader: {} and {}.'.format(model.data_form, loader['data_form'])

    data_loader = loader['data_loader']
    if len(data_loader.dataset) == 0:
        logger.error('No training sample was found.')
        exit(1)
    ### create val dataloader
    val_data_loader = None
    if val_loader_config is not None:
        if not 'mode' in val_loader_config:
            val_loader_config['mode'] = 'val'
        val_loader = create_loader_fn(val_loader_config)
        assert model.data_form == val_loader['data_form'], 'Inconsistent data forms for model and validation loader.'
        val_data_loader = val_loader['data_loader']
        if len(val_data_loader.dataset) == 0:
            logger.error('validation set was provided, but no sample was found.')
            exit(1)
    elif val_split_ratio > 0.0:
        logger.error('Validation set is created by splitting training set with ratio {}.'.format(val_split_ratio))
        data_loader, val_data_loader = random_split_dataloader(data_loader, 
                                                               [1 - val_split_ratio, val_split_ratio], 
                                                               shuffle=[loader_config.get('shuffle', True), False],
                                                               generator=torch.Generator().manual_seed(val_split_seed) if val_split_seed is not None else None)
    else:
        logger.warning('No validation set, use training loss for checkpoint saving, learning scheduling, and early stopping.')
        if early_stop_patience > 0:
            logger.error('No validation set, early stop strategy is not reliable.')
            exit(1)
    #### define the optimizer
    optim_config = train_config.get('optimizer', None)
    assert optim_config is not None, 'Optimizer need to be specified.'
    logger.info('optimizer config: \n{}'.format(optim_config))
    model = model.to(device)
    optimizer = create_optimizer(optim_config, model)
    max_epoch = train_config.get('max_epoch', 1000)
    warmup_epochs = train_config.get('warmup_epochs', 0)
    scheduler_config = train_config.get('lr_scheduler', None)
    eval_batch_num = train_config.get('eval_batch_num', 8)
    if eval_batch_num < 0:
        eval_batch_num = float('inf')
    write_sample_num = train_config.get('write_sample_num', 16)
    clip_grad = train_config.get('clip_grad', None)
    ## if scheduler is OneCycleLR, set as True
    ## otherwise, scheduler is called after a training epoch
    if scheduler_config is not None:
        if scheduler_config['name'] == "OneCycleLR":
            scheduler_config['total_steps'] = max_epoch - warmup_epochs
            scheduler_config['max_lr'] = optim_config.get('lr')
        if scheduler_config['name'] == "LinearLR":
            scheduler_config['total_iters'] = max_epoch - warmup_epochs
        lr_scheduler = create_scheduler(scheduler_config, optimizer)
        logger.info('using {}'.format(type(lr_scheduler)))
    else:
        lr_scheduler = None
        logger.info('Fixed learning rate without any lr_scheduler')

    save_after_epochs = train_config.get('save_after_epochs', 1)
    write_after_iters = train_config.get('write_after_iters', 64)
    
    min_loss_delta = train_config.get('min_loss_delta', 1e-5)
    min_score_delta = train_config.get('min_score_delta', 1e-5)
    early_stop_patience = train_config.get('early_stop_patience', -1)
    formatter = create_formatter(model.data_form)

    ######## training iteration
    iter_id = 0     
    best_loss = 1e10   
    best_score = -1e10
    start_epoch = 1
    patience_count = 0
    ## load model from check points
    pretrained = train_config.get('pretrained', None)
    resume = train_config.get('resume', False)
    if pretrained and resume:
        logger.error('Only one of pretrained and resume can be specified.')
        exit(1)
    elif pretrained is None and not resume:
        logger.info('Model will be trained from scratch.')
    elif pretrained:
        logger.info(f'Using pre-trained model: {pretrained}')
        pretrained_components = train_config.get('pretrained_components', None)
        ignore_mismatched_keys = train_config.get('ignore_mismatched_keys', None)
        if type(pretrained) == str:
            if os.path.isfile(pretrained):
                load_checkpoint(pretrained, model, 
                                ignore_mismatched_keys = ignore_mismatched_keys, 
                                specified_model_components = pretrained_components)
            else:
                logger.warning('Pretrained model doesn\'t exist. Model will be trained from scratch.')
        else:
            assert type(pretrained) == dict, \
                "pretrained should be a model path (str) or pathes of sub-models (dict)."
            assert pretrained_components is None or type(pretrained_components) == dict, \
                "pretrained_components should be a dict when pretrained is a dict."
            assert ignore_mismatched_keys is None or type(ignore_mismatched_keys) == dict, \
                "ignore_mismatched_keys should be a dict when pretrained is a dict."
            load_part = False
            for key in pretrained.keys():
                hasattr(model, key), \
                    "Unknown submodel {} for model {}".format(key, model_name)
                if os.path.isfile(pretrained[key]):
                    load_checkpoint(pretrained[key], getattr(model, key), 
                                    ignore_mismatched_keys=ignore_mismatched_keys.get(key, None) if ignore_mismatched_keys else None,
                                    specified_model_components=pretrained_components.get(key, None) if pretrained_components else None)
                    load_part = True
            if not load_part:
                logger.warning('Pretrained model doesn\'t exist. Model will be trained from scratch.')
    else:
        ## resume
        if isinstance(resume, str):
            resume_pth = resume
        else:
            resume_pth = os.path.join(ckp_dir, "ckp_last.pth")
        if os.path.isfile(resume_pth):
            logger.info(f'Resume from old pth: {resume_pth}')
            state = load_checkpoint(resume_pth, model, 
                optimizer=optimizer, scheduler=lr_scheduler)
            if state is not None:
                best_loss = state['best_loss']
                logger.info(f'previous best loss: {best_loss}')
                best_score = state['best_score']
                if score_metric:
                    logger.info(f'previous best score: {best_score}')
                start_epoch = state['epoch'] + 1
                iter_id = (start_epoch - 1) * len(data_loader)
            else:
                logger.warning('Cannot read the information needed for resume training, use loaded model as pre-trained')
        else:
            if resume == resume_pth:
                logger.warning('Input checkpoint doesn\'t exist.')
            logger.info('Model will be trained from scratch.')
    if warmup_epochs:
        warmup_start_scale = train_config.get('warmup_start_scale', 0.05)
        warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs * len(data_loader), 
                            warmup_start_scale)
        logger.info('Warmup for {} epochs ({} iterations) with start scale {}.'.format(warmup_epochs, 
                        warmup_epochs * len(data_loader), warmup_start_scale))

    sampler = None
    sampler_config = train_config.get('sampler', {'name': 'NormalSampler'})
    if model.is_generative and sampler_config:
        sampler = create_sampler_fn(model=model, sampler_config = sampler_config)
    
    pickle_results = train_config.get('pickle_results', False)
    pickle_path = train_config.get('pickle_path', 'flemme-pickled')
    ## pickled result during traning should never be used later
    if pickle_results:
        mkdirs(pickle_path)
    start_time = datetime.now()
    for epoch in range(start_epoch, max_epoch+1):
        ### training process
        model.train()
        results = []
        for t in data_loader:
            losses, res = run_fn(model, t)
            #### to numpy for evaluation
            if evaluators is not None and len(results) < eval_batch_num:
                if res is None: res = run_fn(model, t, only_forward = True)
                process_results(results=results, 
                                    res = res, 
                                    data_form = model.data_form, 
                                    pickle_results=pickle_results, 
                                    pickle_path=pickle_path,
                                    mode = mode)
            loss = sum(losses)
            optimizer.zero_grad()
            loss.backward()
            if detect_anomaly:
                check_nan_grad(model)
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            if epoch <= warmup_epochs:
                warmup_scheduler.step(iter_id + 1)
            optimizer.step()
            iter_id += 1
            if iter_id % write_after_iters == 0:
                ### write data into tensorboard
                logger.info('iter {:04d}, training loss: {}'.format(iter_id, loss.item()))
                for ln, lv in zip(loss_names, losses):
                    write_loss(writer=writer, loss_name=ln, loss = lv.item(), iter_id=iter_id)
                write_lr(writer=writer, optimizer=optimizer, iter_id=iter_id)
                if write_sample_num > 0 \
                    and (model.data_form == DataForm.IMG or model.data_form == DataForm.PCD \
                    or model.data_form == DataForm.VEC):
                    with torch.no_grad():
                        for rn in res:
                            if torch.is_tensor(res[rn]):
                                res[rn] = res[rn][:write_sample_num]
                        actual_write_sample_num = res['input'].shape[0]
                        if sampler is not None:
                            if is_conditional:
                                x_rand = sampler.generate_rand(n=actual_write_sample_num, cond = res['condition'])   
                            elif is_supervised:
                                x_rand = sampler.generate_rand(n=actual_write_sample_num, cond = res['input'])   
                            else:
                                x_rand = sampler.generate_rand(n=actual_write_sample_num)   
                            res['gen'] = x_rand

                    write_data(writer=writer, formatter=formatter, data_form = model.data_form, 
                                input_map=res, iter_id=epoch, prefix='train',
                                additional_keys = custom_write_results)
        ### evaluate after each epoch
        ### write evaluation
        if evaluators is not None:
            # results = compact_results(results, data_form = model.data_form)
            eval_res = evaluate_results(results, evaluators, data_form = model.data_form)

            if len(eval_res) > 0:
                for eval_type, eval in eval_res.items():
                    for eval_metric, eval_value in eval.items():
                        if eval_metric == score_metric:
                            eval_score = eval_value
                        write_eval(writer=writer, eval_metric=eval_type + '_eval/' + eval_metric, 
                                    eval_value=eval_value, iter_id=epoch, prefix='train')

        ### training epoch finished
        logger.info('epoch {:03d}/{:03d}, training loss: {}'.format(epoch, max_epoch, loss.item()))
        remaining_time = (datetime.now() - start_time) * (max_epoch - epoch) / (epoch + 1 - start_epoch)
        logger.info(f'The training is expected to be completed in {str(remaining_time)}.')

        if epoch > warmup_epochs and lr_scheduler is not None:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(loss)
            else:
                lr_scheduler.step()
        if epoch % save_after_epochs == 0:
            if val_data_loader is not None:
                ### validation epoch !!!!!
                ## set val_loss as loss
                model.eval()
                with torch.no_grad():
                    vresults = []
                    val_losses = torch.zeros(len(loss_names))
                    val_n = 0
                    for vt in tqdm(val_data_loader, desc='validating'):
                        vlosses, vres = run_fn(model, vt)
                        if evaluators is not None and len(vresults) < eval_batch_num:
                            if vres is None: vres = run_fn(model, vt, only_forward = True)
                            process_results(results=vresults, 
                                                res = vres, 
                                                data_form = model.data_form,
                                                is_supervised=is_supervised, 
                                                is_conditional=is_conditional,
                                                pickle_results=pickle_results, 
                                                pickle_path=pickle_path,
                                                mode = 'val')
                        sample_size = vres['input'].shape[0]
                        ## mean, sum, or None
                        val_losses += torch.Tensor([l.item() * sample_size if model.loss_reduction == 'mean' else l.sum().item() for l in vlosses])
                        val_n += sample_size
                    val_losses = val_losses / val_n
                    loss = val_losses.sum().item()
                    #### write validation loss
                    logger.info('epoch {:03d}/{:03d}, validation loss: {}'.format(epoch, max_epoch, loss))
                    for ln, lv in zip(loss_names, val_losses):
                        write_loss(writer=writer, loss_name=ln, loss = lv.item(), iter_id=epoch, prefix='val')
                    if write_sample_num > 0 \
                        and (model.data_form == DataForm.IMG or model.data_form == DataForm.PCD \
                        or model.data_form == DataForm.VEC):
                        for rn in vres:
                            vres[rn] = vres[rn][:write_sample_num]
                        write_data(writer=writer, formatter=formatter, data_form = model.data_form, 
                                input_map=vres, iter_id=epoch, prefix='val',
                                additional_keys = custom_write_results)
                    ### evaluation on val datasets
                    if evaluators is not None:
                        # vresults = compact_results(vresults, data_form = model.data_form)
                        eval_res = evaluate_results(vresults, evaluators, data_form = model.data_form)
                        if len(eval_res) > 0:
                            for eval_type, eval in eval_res.items():
                                for eval_metric, eval_value in eval.items():
                                    write_eval(writer=writer, eval_metric=eval_type + '_eval/' + eval_metric, 
                                                eval_value=eval_value, iter_id=epoch, prefix='val')
                                    if eval_metric == score_metric:
                                        eval_score = eval_value
                                        logger.info('epoch {:03d}/{:03d}, validation score: {}'.format(epoch, max_epoch, eval_score))

            ## if without validation, the loss and score are computed using training dataset
            is_best_loss, is_best_score = False, False
            if loss <= best_loss - min_loss_delta:
                best_loss = loss
                is_best_loss = True

            if score_metric:
                ### 'lower is better' score
                if not higher_is_better:
                    eval_score = 1 - eval_score
                if eval_score >= best_score + min_score_delta:
                    best_score = eval_score
                    is_best_score = True                
            logger.info('saving model to {}'.format(ckp_dir))
            save_checkpoint(ckp_dir, model, 
                            optimizer=optimizer, 
                            scheduler=lr_scheduler,
                            epoch=epoch,
                            best_loss=best_loss,
                            best_score=best_score,
                            is_best_loss=is_best_loss,
                            is_best_score=is_best_score)
            if early_stop_patience > 0 :
                if not is_best_loss and not is_best_score:
                    patience_count += 1
                    if patience_count >= early_stop_patience:
                        logger.info('Early stop training.')
                        break
                else:
                    patience_count = 0
    logger.info('Finish training.')
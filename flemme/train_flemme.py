from torch.utils.tensorboard import SummaryWriter
import os
import torch
# import torch.nn as nn
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from .trainer_utils import *
from flemme.model import create_model
from flemme.dataset import create_loader
from flemme.logger import get_logger
from flemme.sampler import create_sampler
from flemme.config import module_config
logger = get_logger('train_flemme')
## if we want to train pcd or image, 
## make sure that the image size from data loader and image size from the model parameters are identical
device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 1e-8

def main():
    train_config = load_config()
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
        logger.info('Use determinstic algorithms, note that this may leads to performance decreasing.')
        if rand_seed is None:
            logger.warning('No random seed is specified, use 0 as random seed.')
            torch.manual_seed(0)
            np.random.seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    loader_config = train_config.get('loader', None)
    assert loader_config is not None, "Data loader is not speficied"
    if not 'mode' in loader_config:
        loader_config['mode'] = mode

    val_loader_config = train_config.get('val_loader', None)

    ckp_dir = train_config.get('check_point_dir', '.')
    #### use writer to visualize the training process
    writer = SummaryWriter(log_dir = os.path.join(ckp_dir, 'logs'))

    #### create model
    model = create_model(model_config)

    logger.info("model info:\n{}".format(model))
    
    is_conditional = model.is_conditional
    is_supervising = model.is_supervising

    eval_metrics = train_config.get('evaluation_metrics', {})
    evaluators = create_batch_evaluators(eval_metrics, model.data_form)
    
    ### use eval to compute the score and select the best model
    score_metric = None
    score_metric_config = train_config.get('score_metric', None)
    if evaluators is not None and score_metric_config is not None:
        score_metric = score_metric_config.get('name')
        higher_is_better = score_metric_config.get('higher_is_better', True)
        assert sum([score_metric in e for e in evaluators.values()]) > 0, "Score metric is not in the evaluation metrics."
    
    if is_supervising:
        logger.info('Supervising model, we will using label as target.')
    elif is_conditional:
        logger.info('Conditional model, we will using label as condition.')

    loss_names = model.get_loss_name()
    logger.info('Using loss(es): {}'.format(loss_names))
    logger.info("Total number of model parameters: {}".format(sum(p.numel() for p in model.parameters())))
    #### create dataset and dataloader
    loader = create_loader(loader_config)
    assert model.data_form == loader['data_form'], \
        'Inconsistent data forms for model and loader: {} and {}.'.format(model.data_form, loader['data_form'])

    data_loader = loader['data_loader']
    if len(data_loader.dataset) == 0:
        logger.error('No training sample was found.')
        exit(1)
    ### create val dataloader
    if val_loader_config is not None:
        if not 'mode' in val_loader_config:
            val_loader_config['mode'] = 'val'
        val_loader = create_loader(val_loader_config)
        assert model.data_form == val_loader['data_form'], 'Inconsistent data forms for model and validation loader.'
        val_data_loader = val_loader['data_loader']
        if len(val_data_loader.dataset) == 0:
            logger.error('validation set was provided, but no sample was found.')
            exit(1)
    else:
        logger.warning('No validation set, use training loss for checkpoint saving and learning scheduling.')

    logger.debug('size of loader:', len(data_loader))

    #### define the optimizer
    optim_config = train_config.get('optimizer', None)
    assert optim_config is not None, 'Optimizer need to be specified.'
    logger.info('optimizer config: \n{}'.format(optim_config))
    model = model.to(device)
    optimizer = create_optimizer(optim_config, model)
    max_epoch = train_config.get('max_epoch', 1000)
    warmup_epoch = train_config.get('warmup_epoch', 1)
    scheduler_config = train_config.get('lr_scheduler', None)
    eval_batch_num = train_config.get('eval_batch_num', 8)
    write_sample_num= train_config.get('write_sample_num', 16)


    ## if scheduler is OneCycleLR, set as True
    ## otherwise, scheduler is called after a training epoch
    if scheduler_config is not None:
        if scheduler_config['name'] == "OneCycleLR":
            scheduler_config['total_steps'] = max_epoch - warmup_epoch
            scheduler_config['max_lr'] = optim_config.get('lr')
        if scheduler_config['name'] == "LinearLR":
            scheduler_config['total_iters'] = max_epoch - warmup_epoch
        lr_scheduler = create_scheduler(scheduler_config, optimizer)
        logger.info('using {}'.format(type(lr_scheduler)))
    else:
        lr_scheduler = None
        logger.info('Fixed learning rate without any lr_scheduler')

    save_after_epochs = train_config.get('save_after_epochs', 1)
    write_after_iters = train_config.get('write_after_iters', 64)
    formatter = create_formatter(model.data_form)

    ######## training iteration
    iter_id = 0     
    best_loss = 1e10   
    best_score = -1e10
    torch.autograd.set_detect_anomaly(True)
    start_epoch = 1
    ## load model from check points
    pretrained = train_config.get('pretrained', None)
    resume = train_config.get('resume', None)

    if pretrained is not None and resume is not None:
        logger.error('Only one of pretrained and resume can be specified.')
        exit(1)

    if pretrained is not None:
        logger.info(f'Using pre-trained model: {pretrained}')
        load_checkpoint(pretrained, model)

    if resume is not None:
        logger.info(f'Resume from last pth: {resume}')
        state = load_checkpoint(resume, model, optimizer=optimizer, scheduler=lr_scheduler)
        if state is not None:
            best_loss = state['best_loss']
            logger.info(f'previous best loss: {best_loss}')
            best_score = state['best_score']
            if score_metric:
                logger.info(f'previous best score: {best_score}')
            start_epoch = state['epoch'] + 1
            iter_id = (start_epoch - 1) * len(data_loader)
        else:
            logger.warning('Cannot read the information needed for resume, use loaded model as pre-trained')
    
    sampler = None
    sampler_config = train_config.get('sampler', None)
    if model.is_generative and sampler_config is not None:
        sampler = create_sampler(model=model, device = device, sampler_config = sampler_config)


    for epoch in range(start_epoch, max_epoch+1):
        ### training process
        model.train()
        results = {'input':[], 'target':[], 'condition':[], 'latent':[], 'recon':[], 'seg':[], 'cluster':[]}
        for t in data_loader:
            x, y, _ = t
            x, y = x.to(device).float(), y.to(device)
            
            if not x.shape[1:] == tuple(model.get_input_shape()):
                logger.error("Inconsistent sample shape between data and model: {} and {}".format(x.shape[1:], tuple(model.get_input_shape())))
                exit(1)                
            ### here we want to generate raw image
            losses, res = compute_loss(model, x, y)
            
            #### to numpy for evaluation
            if evaluators is not None and len(results['input']) < eval_batch_num:
                if res is None: res = forward_pass(model, x, y)
                append_results(results=results, x = x, y = y, 
                                        res = res, data_form = model.data_form, 
                                        is_supervising=is_supervising, 
                                        is_conditional=is_conditional)
            loss = sum(losses)
            optimizer.zero_grad()
            loss.backward()
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
                        x = x[:write_sample_num]
                        y = y[:write_sample_num]
                        if res is None: res = forward_pass(model, x, y)
                        else:
                            for rn in res:
                                if torch.is_tensor(res[rn]):
                                    res[rn] = res[rn][:write_sample_num]
                        if sampler is not None:
                            if is_conditional:
                                x_rand = sampler.generate_rand(n=x.shape[0], cond = y)   
                            else:
                                x_rand = sampler.generate_rand(n=x.shape[0])   
                            res['gen'] = x_rand
                        res['input'] = x
                        if is_supervising:
                            res['target'] = y
                        if is_conditional:
                            res['condition'] = y

                    write_data(writer=writer, formatter=formatter, data_form = model.data_form, 
                                input_map=res, iter_id=epoch, prefix='train')
        ### evaluate after each epoch
        ### write evaluation
        if evaluators is not None:
            results = compact_results(results)
            eval_res = evaluate_results(results, evaluators)

            if len(eval_res) > 0:
                for eval_type, eval in eval_res.items():
                    for eval_metric, eval_value in eval.items():
                        if eval_metric == score_metric:
                            eval_score = eval_value
                        write_eval(writer=writer, eval_metric=eval_type + '_eval/' + eval_metric, 
                                    eval_value=eval_value, iter_id=epoch, prefix='train')

        ### training epoch finished
        logger.info('epoch {:03d}/{:03d}, training loss: {}'.format(epoch, max_epoch, loss.item()))
        if epoch > warmup_epoch and lr_scheduler is not None:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(loss)
            else:
                lr_scheduler.step()
        if epoch % save_after_epochs == 0:
            if val_loader_config is not None:
                ### validation epoch !!!!!
                ## set val_loss as loss
                model.eval()
                with torch.no_grad():
                    vresults = {'input':[], 'target':[], 'condition':[], 'latent':[], 'recon':[], 'seg':[], 'cluster':[]}
                    val_losses = torch.zeros(len(loss_names))
                    val_n = 0
                    for vt in val_data_loader:
                        vx, vy, _ = vt
                        vx, vy = vx.to(device), vy.to(device)
                        vlosses, vres = compute_loss(model, vx, vy)

                        if evaluators is not None and len(vresults['input']) < eval_batch_num:
                            if vres is None: vres = forward_pass(model, x, y)
                            append_results(results=vresults, x = vx, y = vy, 
                                                res = vres, data_form = model.data_form,
                                                is_supervising=is_supervising, 
                                                is_conditional=is_conditional)
                        val_losses += torch.Tensor([l.item() * vx.shape[0] if model.loss_reduction == 'mean' else l.sum().item() for l in vlosses])
                        val_n += vx.shape[0]
                    val_losses = val_losses / val_n
                    loss = val_losses.sum().item()
                    #### write validation loss
                    logger.info('epoch {:03d}/{:03d}, validation loss: {}'.format(epoch, max_epoch, loss))
                    for ln, lv in zip(loss_names, vlosses):
                        write_loss(writer=writer, loss_name=ln, loss = lv.item(), iter_id=epoch, prefix='val')
                    if write_sample_num > 0 \
                        and (model.data_form == DataForm.IMG or model.data_form == DataForm.PCD \
                        or model.data_form == DataForm.VEC):
                        vx = vx[:write_sample_num]
                        vy = vy[:write_sample_num]
                        if vres is None: vres = forward_pass(model, vx, vy)
                        else:
                            for rn in vres:
                                vres[rn] = vres[rn][:write_sample_num]
                        vres['input'] = vx
                        if is_supervising:
                            vres['target'] = vy
                        if is_conditional:
                            vres['condition'] = vy
                        write_data(writer=writer, formatter=formatter, data_form = model.data_form, 
                                input_map=vres, iter_id=epoch, prefix='val')
                    ### evaluation on val datasets
                    if evaluators is not None:
                        vresults = compact_results(vresults)
                        eval_res = evaluate_results(vresults, evaluators)
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
            if loss < best_loss:
                best_loss = loss
                is_best_loss = True

            if score_metric:
                ### 'lower is better' score
                if not higher_is_better:
                    eval_score = 1 - eval_score
                if eval_score > best_score:
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
    logger.info('Finish training.')
            


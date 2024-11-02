import os
import numpy as np
import torch
import importlib
from torch import optim
from flemme.utils import *
from .metrics import get_metrics
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from flemme.logger import get_logger
from flemme.color_table import color_table
import shutil
logger = get_logger('trainer_utils')

def colorize_by_label(labels):
    if torch.is_tensor(labels): 
        _color_table = torch.Tensor(color_table).to(labels.device)
        labels = torch.clamp(labels.long(), 0, len(color_table)-1)
    else:
        _color_table = np.array(color_table)
        labels = np.clip(labels.astype(int), 0, len(color_table)-1)
        
    colors = _color_table[labels.flatten()].reshape(labels.shape + (-1,))
    return colors

### 2D image
def colorize_img_by_label(label, img, gt = None, background = 0, threshold = 0.5, alpha = 0.65):
    assert label.shape[0] == 1, \
        'Number of channels should be 1. Remember to translate one-hot encoding to normal labels.'
    if img.ndim == 4:
        middle_z = img.shape[1] // 2
        if gt is not None:
            ### compute the middle frame using gt
            min_zxy, max_zxy = get_boundingbox(gt[0], background = background)
            middle_z = int(min_zxy[0] + max_zxy[0]) // 2
            gt = gt[:, middle_z, ...]
        img = img[:, middle_z, ...]
        label = label[:, middle_z, ...]
            
    if img.shape[0] != 1 and img.shape[0] != 3:
        img = img.mean(axis = 0, keepdims=True) 
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    img = normalize_img(img)
    original_shape = img.shape
    img = img.reshape(3, -1)
    raw_img = img.copy()
    label = label.flatten()
    if background is not None:
        non_background_label = label[label != background]
        if len(non_background_label) > 0:
            non_background_label[non_background_label > background] = \
                            non_background_label[non_background_label > background] - 1
            non_background_colors = colorize_by_label(non_background_label).transpose()
            img[:, label != background] = img[:, label != background]* (1 - alpha)\
                + non_background_colors * alpha
    else:
        colors = colorize_by_label(label)
        img = img * (1 - alpha) + colors * alpha   
    if gt is not None:
        ### paint error parts with red
        gt = gt.flatten()
        img[0, label != gt] = raw_img[0, label != gt]* (1 - alpha) +  alpha*0.85
        img[1, label != gt] = raw_img[1, label != gt]* (1 - alpha)
        img[2, label != gt] = raw_img[2, label != gt]* (1 - alpha)
    
    return img.reshape(original_shape), raw_img.reshape(original_shape)

## Tensorboard Formatter for image, point cloud
class ImageTensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. 
    """

    def __init__(self):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        return self.process_batch(name, batch)
    ### we shouldn't abondon the channel information
    def process_batch(self, name, batch):
        ### check:
        assert batch.ndim == 4 or batch.ndim == 5, 'Only 2D (NCHW) and 3D (NCDHW) images are accepted for display'
        tag_template = '{}/batch_{}'

        tagged_images = []
        ### if image is not with form "gray" or "rgb"
        if batch.shape[1] != 1 and batch.shape[1] != 3:
            # logger.warning('Only 1-channel and 3-channel images are supported for visualization, convert the images to 1-channel images by choosing the middle channel.')
            ## choose the last 3 channel
            if batch.shape[1] < 3:
                batch = batch[:, [-1], ...]
            else:
                batch = batch[:, [-3, -2, -1], ...]
        if batch.ndim == 5:
            # batch of 3D images: NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice    
            batch = batch[:, :, slice_idx, ...]
        if batch.max() > 1 or batch.min() < 0:
            batch = normalize_img(batch, channel_dim = 1)
        ## we need to split image batch into images.
        for batch_idx in range(batch.shape[0]):
            tag = tag_template.format(name, batch_idx)
            img = batch[batch_idx, ...]
            tagged_images.append((tag, img))
        

        return tagged_images

class PcdTensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of point cloud (be it input/output to the network or the target
    point cloud) to a series of point clouds that can be displayed in tensorboard. 
    """

    def __init__(self):
        pass

    def __call__(self, name, batch):

        return self.process_batch(name, batch)
    ### we shouldn't abondon the channel information
    def process_batch(self, name, batch):
        assert batch.ndim == 3, 'pcd batches need to be organized as shape (N * Pn * Pd).'
        assert batch.shape[2] == 3 or batch.shape[2] == 6, 'Only support 3D Point cloud for visualization.'
        color = None
        if batch.shape[2] == 6:
            batch, color = torch.chunk(batch, 2, dim=-1)
            color = (color * 255).long()
        ## we can add pcd batch into tensorboard directly.
        return [(name, (batch, color)), ]

class Vec2PCDTensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of points to point cloud
    """

    def __init__(self):
        pass
    def __call__(self, name, batch):
        return self.process_batch(name, batch)
    ### we shouldn't abondon the channel information
    def process_batch(self, name, batch):
        assert batch.ndim == 2 and (batch.shape[1] == 3 or batch.shape[1] == 2), 'Only support the visualization of 3D vectors.'
        ## we can add pcd batch into tensorboard directly.
        if batch.shape[1] == 3:
            return [(name, batch[None, :]), ]
        else:
            plt.figure(figsize=(10, 8))
            plt.scatter(batch[:, 0].detach().cpu().numpy(), 
                        batch[:, 1].detach().cpu().numpy(), s = 10, alpha=0.3)
            return [(name, plt.gcf()),]  
def create_formatter(data_form):
    if data_form == DataForm.IMG:
        return ImageTensorboardFormatter()
    elif data_form == DataForm.PCD:
        return PcdTensorboardFormatter()
    elif data_form == DataForm.VEC:
        return Vec2PCDTensorboardFormatter()
def write_data(writer, formatter, data_form, iter_id, input_map, prefix='train'):
    #### if data form is pcd, get the color by labels
    if 'seg_logits' in input_map:
        if data_form == DataForm.PCD:
            #### pcd colorized by label (y is label, seg is predicted label)
            seg_color = colorize_by_label(onehot_to_label(input_map['seg_logits'], -1))
            target_color = colorize_by_label(onehot_to_label(input_map['target'], -1))
            input_map['seg'] = torch.cat((input_map['input'], seg_color), dim = -1)
            input_map['target'] = torch.cat((input_map['input'], target_color), dim = -1)
        elif data_form == DataForm.IMG and not 'seg' in input_map:
            input_map['seg'] = logits_to_onehot_label(input_map['seg_logits'], data_form)

    for name, batch in input_map.items():
        if not name in ['input', 'target', 'condition', 'recon', 'seg', 'gen']: continue
        if batch is None or not torch.is_tensor(batch): 
            logger.debug('results of {} is not a tensor.'.format(name))
            continue
        if batch.ndim < 2 or batch.ndim == 2 and not (batch.shape[1] == 3 or batch.shape[1] == 2):
            logger.debug('results of {} can not be visualized.'.format(name))
            continue
        for tag, data in formatter(name, batch):
            if data_form == DataForm.IMG:
                writer.add_image(prefix + '_' + tag, data, iter_id)
            elif data_form == DataForm.PCD:
                vertices, colors = data
                writer.add_mesh(prefix + '_'+ tag, vertices = vertices, colors = colors, global_step = iter_id)
            elif data_form == DataForm.VEC:
                if batch.shape[1] == 3:
                    writer.add_mesh(prefix + '_'+ tag, vertices = data, global_step = iter_id)
                else:
                    writer.add_figure(prefix + '_'+ tag, data, iter_id)
def write_figure(writer, tag, fig, iter_id):
    writer.add_figure(tag, fig, iter_id)


def write_lr(writer, optimizer, iter_id):
    for idx in range(len(optimizer.param_groups)):
        lr = optimizer.param_groups[idx]['lr']
        writer.add_scalar('learning_rate_'+str(idx), lr, iter_id)

def write_loss(writer, loss_name, loss, iter_id, prefix='train'):
    writer.add_scalar(f'{prefix}_{loss_name}_loss_avg', loss, iter_id)

def write_eval(writer, eval_metric, eval_value, iter_id, prefix='train'):
    writer.add_scalar(f'{prefix}_{eval_metric}_eval_avg', eval_value, iter_id)

def create_optimizer(optim_config, model):
    name = optim_config.pop('name', 'SGD')
    assert name in ['SGD', 'Adam', 'AdamW'], 'Unsupported optimizer.'
    optim_config['params'] = model.parameters()
    if name == 'SGD':
        return optim.SGD(**optim_config)
    elif name =='Adam':
        return optim.Adam(**optim_config)
    elif name =='AdamW':
        return optim.AdamW(**optim_config)

def create_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)
### get the TSNE embeddings
## x is a set of latent embeddings
def tsne(X, n_components = 2):
    return TSNE(n_components=n_components, learning_rate='auto', init='random').fit_transform(X)
#### wait to be implemented: save optimizer.
def save_checkpoint(ckp_dir, model, optimizer = None, 
                    scheduler = None, epoch = -1, best_loss = 1e10,
                    best_score = -1e10, is_best_loss = False, 
                    is_best_score = False):
    
    path = "{}/ckp_last.pth".format(ckp_dir)
    if not os.path.exists(ckp_dir):
        os.makedirs(ckp_dir)
    
    state_dict = {}
    state_dict['trained_model'] = model.state_dict()
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()
    if epoch > 0:
        state_dict['epoch'] = epoch
        state_dict['best_loss'] = best_loss
        state_dict['best_score'] = best_score

    torch.save(state_dict, path)
    if is_best_loss:
        shutil.copyfile(path, "{}/ckp_best_loss.pth".format(ckp_dir))
    if is_best_score:
        shutil.copyfile(path, "{}/ckp_best_score.pth".format(ckp_dir))
    
def load_checkpoint(ckp_path, model, optimizer = None, scheduler = None, ignore_mismatched_keys = []):
    logger.info('load model from {}'.format(ckp_path))
    state_dict = torch.load(ckp_path, map_location='cpu')
    if 'trained_model' in state_dict:
        trained_model_state_dict = state_dict.pop('trained_model')
        if len(ignore_mismatched_keys) > 0:
            logger.info(f'ignore keys while loading model: {ignore_mismatched_keys}')
            model_state_dict = model.state_dict()
            for k in trained_model_state_dict:
                ignored = sum([ imk in k for imk in ignore_mismatched_keys]) > 0
                if ignored:
                    trained_model_state_dict[k] = model_state_dict[k]
        model.load_state_dict(trained_model_state_dict)
        if optimizer is not None:
            optimizer.load_state_dict(state_dict.pop('optimizer'))
        if scheduler is not None:
            scheduler.load_state_dict(state_dict.pop('scheduler'))    
        return state_dict
    else:
        model.load_state_dict(state_dict)

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False
def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def append_results(results, x, y, res, data_form, path = None,  is_supervised = False, is_conditional = False):

        results['input'].append(x.cpu().detach().numpy())
        if y is not None:
            if is_supervised:
                results['target'].append(y.cpu().detach().numpy())
            if is_conditional:
                results['condition'].append(y.cpu().detach().numpy())
            else:
                ### y is not used during the model training.
                pass 
                
        ## should be a list of strings or none
        ## only for test
        if path is not None:
            if type(path[0]) == tuple:
                results['path'] += list(path[0])
            else:
                results['path'] += path
        if res:
            if 'cluster_logits' in res:
                res['cluster'] = logits_to_onehot_label(res['cluster_logits'], data_form)
            if 'seg_logits' in res:
                res['seg'] = logits_to_onehot_label(res['seg_logits'], data_form)
            if 'latent' in res:
                results['latent'].append(res['latent'].cpu().detach().numpy())
            if 'recon' in res:
                results['recon'].append(res['recon'].cpu().detach().numpy())
            if 'seg' in res:
                results['seg'].append(res['seg'].cpu().detach().numpy())
            if 'cluster' in res:
                results['cluster'].append(res['cluster'].cpu().detach().numpy())
            if 'cluster_centers' in res:
                results['cluster_centers'] = res['cluster_centers'].cpu().detach().numpy()
def compact_results(results):
    results['input'] = np.concatenate(results['input'])
    if len(results['target']) > 0:
        results['target'] = np.concatenate(results['target'])
    if len(results['condition']) > 0:
        results['condition'] = np.concatenate(results['condition'])        
    if len(results['latent']) > 0:
        results['latent'] = np.concatenate(results['latent'])
    if len(results['recon']) > 0:
        results['recon'] = np.concatenate(results['recon'])       
    if len(results['seg']) > 0:
        results['seg'] = np.concatenate(results['seg'])
    if len(results['cluster']) > 0:
        results['cluster'] = np.concatenate(results['cluster'])
    return results

def create_evaluator(eval_configs, data_form):

    if len(eval_configs) ==0:
        return None
    evaluator = {}
    for e_config in eval_configs:
        name = e_config.get('name')
        e = get_metrics(e_config, data_form=data_form)
        if e is not None:
            evaluator[name] = e
    return evaluator

def create_batch_evaluators(eval_metrics, data_form):
    if len(eval_metrics) == 0:
        return None
    r_eval = create_evaluator(eval_metrics.get('recon', []), data_form)
    s_eval = create_evaluator(eval_metrics.get('seg', []), data_form)
    c_eval = create_evaluator(eval_metrics.get('cluster', []), data_form)
    evaluators = {}

    if r_eval is not None: 
        evaluators['recon'] = r_eval
    if s_eval is not None: 
        evaluators['seg'] = s_eval
    if c_eval is not None: 
        evaluators['cluster'] = c_eval
    return evaluators

def evaluate_results(results, evaluators):
    sample_num = len(results['input'])
    eval_res = {}
    for eval_type in evaluators:
        if len(results[eval_type]) == 0:
            logger.warning(f'This model doesn\'t predict {eval_type}')
        else:
            eval_res[eval_type] = {}
            if eval_type == 'cluster':
                for (eval_metric, eval_func) in evaluators[eval_type].items():
                    eval_res[eval_type][eval_metric] = eval_func(results['cluster'], results['target'])
            else:
                for (eval_metric, eval_func) in evaluators[eval_type].items():
                    eval_res[eval_type][eval_metric] = 0.0
                    if eval_type == 'recon':
                        #### supervised
                        if len(results['target']) > 0:
                            zipped = zip(results['recon'], results['target'])
                        ### unsupervised
                        else:
                            zipped = zip(results['recon'], results['input'])
                    elif eval_type == 'seg':
                        zipped = zip(results['seg'], results['target'])
                    for pred, target in zipped:
                        eval_res[eval_type][eval_metric] += eval_func(pred, target)
                    # print(eval_metric, eval_func)
                    # print(eval_res)
                    eval_res[eval_type][eval_metric] /= sample_num
    return eval_res

def compute_loss(model, x, y):
    if model.is_supervised:
        losses, res = model.compute_loss(x, y = y)
    elif model.is_conditional:
        # losses, res = model.compute_loss(x, y, epoch<=2)
        losses, res = model.compute_loss(x, c = y)
    else:
        losses, res = model.compute_loss(x)
    return losses, res
def forward_pass(model, x, y):
    if model.is_conditional:
        res = model(x, c = y)
    else:
        res = model(x)
    return res

#### save tsne visualization
def construct_tsne_vis(embeddings, labels = None, cluster_centers = None, vis_dim = 2):
    c_end = 0
    if cluster_centers is not None:
        embeddings = np.concatenate([cluster_centers, embeddings])
        c_end = cluster_centers.shape[0]
        if labels is not None:
            labels = np.concatenate([np.arange(c_end), labels])
    kwargs = {}
    center_kwargs = {}
    if labels is not None:
        kwargs = {'c':labels[c_end:], 
            'cmap':plt.cm.get_cmap('jet', labels.max() + 1)}
        center_kwargs = {'c':labels[:c_end], 
            'cmap':plt.cm.get_cmap('jet', labels.max() + 1)}
    if embeddings.shape[1] > vis_dim:
        tsne = TSNE(n_components = vis_dim, random_state=42)
        compressed_vec = tsne.fit_transform(embeddings)
    else:
        compressed_vec = embeddings
    if vis_dim == 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(compressed_vec[c_end:, 0], compressed_vec[c_end:, 1], s = 10, alpha=0.3, **kwargs)
        if c_end > 0:
            plt.scatter(compressed_vec[:c_end, 0], compressed_vec[:c_end, 1], s = 50, marker='*',  **center_kwargs)
        plt.colorbar()
    elif vis_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(compressed_vec[c_end:, 0], compressed_vec[c_end:, 1], 
                   compressed_vec[c_end:, 2], s = 10, **kwargs)
        if c_end > 0:
            ax.scatter(compressed_vec[:c_end, 0], compressed_vec[:c_end, 1], 
                       compressed_vec[:c_end, 2], s = 50, marker='*', **center_kwargs)
    plt.title('t-SNE Visualization of Embedding')
    return plt.gcf()        

### figs in batch form
def combine_figures(figs, row_length, size = (32, 32)):
    col_length = int(len(figs) / row_length)
    # display an n*n 2D manifold of digits
    if not isinstance(size, list) and not isinstance(size, tuple) :
        size = (size, size)
    figure = np.zeros((figs.shape[1], col_length * size[0], row_length * size[1]))
    for i in range(col_length):
        for j in range(row_length):
            f = normalize_img(figs[i * row_length + j])

            ## size f
            figure[:,
                i * size[0] : (i + 1) * size[0],
                j * size[1] : (j + 1) * size[1],
            ] = f
    return figure

## we save 2D image to png, 3D image to nii.gz, point cloud to ply.
def save_data(output, data_form, output_path, segmentation = False):    
    if data_form == DataForm.IMG:
        # remove channel dimension
        if len(output.shape) == 3:
            ### non-binary segmentation will be saved as npy
            if segmentation:
                output = output[0]
                save_npy(output_path+'.npy', output)
            ### binary segmentation will be saved as png
            else:
                if output.max() > 1 or output.min() < 0:
                    output = normalize_img(output)
                save_img(output_path+'.png', (output * 255).astype('uint8'))
        ## CDHW
        elif len(output.shape) == 4:
            output = output[0]
            save_itk(output_path+'.nii.gz', output)
    elif data_form == DataForm.PCD:
        if segmentation:
            np.savetxt(output_path+'.seg', output)
        else:
            save_ply(output_path+'.ply', output)
    else:
        raise NotImplementedError

def get_load_function(suffix):
    load_data = None
    data_type = 'img'
    if suffix.endswith('png') or suffix.endswith('jpg') or suffix.endswith('tif'):
        load_data = load_img_as_numpy
    elif suffix.endswith('nii.gz') :
        load_data = load_itk
        data_type = 'vol'
    elif suffix.endswith('npy'):
        load_data = load_npy
        data_type = 'npy'
    if module_config['point-cloud']:
        if suffix.endswith('xyz') or suffix.endswith('ply'):
            data_type = 'pcd'
            load_data = load_pcd
        elif suffix.endswith('txt') or suffix.endswith('seg'):
            load_data = np.loadtxt
            data_type = 'txt'
    if load_data is None:
        logger.error('Unknown data type.')
        exit(1)
    return load_data, data_type
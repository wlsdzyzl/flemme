import importlib
from flemme.utils import *
from torch import optim
from flemme.metrics import get_metrics
from matplotlib import pyplot as plt
from flemme.color_table import get_color_table
from sklearn.manifold import TSNE
from flemme.logger import get_logger
from flemme.color_table import color_table
from functools import partial
import shutil
import joblib
from tqdm import tqdm

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

### expand cropped image to square image for visualization 
def expand_to_square_img(img, margin = 10):
    assert img.ndim == 3, "Only support to expand 2D image"
    length = max(img.shape[1:]) + margin * 2
    expanded_img = np.zeros((img.shape[0], ) + (length, ) * (img.ndim - 1), dtype = img.dtype)
    slice_x = slice((length - img.shape[1]) // 2, (length - img.shape[1]) // 2 + img.shape[1])
    slice_y = slice((length - img.shape[2]) // 2, (length - img.shape[2]) // 2 + img.shape[2])
    slice_c = slice(0, img.shape[1])
    expanded_img[(slice_c, slice_x, slice_y)] = img
    return expanded_img
    
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
        # assert batch.ndim == 4 or batch.ndim == 5, 'Only 2D (NCHW) and 3D (NCDHW) images are accepted for display'
        if not (batch.ndim == 4 or batch.ndim == 5): return [('0', None), ]
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
        # assert batch.ndim == 3, 'pcd batches need to be organized as shape (N * Pn * Pd).'
        # assert batch.shape[2] > 3, 'Only support 3D Point cloud for visualization.'

        if not (batch.ndim == 3 and batch.shape[2] >= 3):
            return [('0', None), ]

        if not (batch.shape[2] == 3 or batch.shape[2] == 6):
            batch = batch[:, :, 0:3]
        
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
        # assert batch.ndim == 2 and batch.shape[1] > 2, 'Only support the visualization of 2D or 3D vectors.'
        if not (batch.ndim == 2 and batch.shape[1] >= 2):
            return [('0', None), ]
        ## we can add pcd batch into tensorboard directly.
        if batch.shape[1] == 3:
            return [(name, batch[None, :]), ]
        else:
            batch = batch[:, 0:2]
            plt.figure(figsize=(10, 8))
            plt.scatter(batch[:, 0].detach().cpu().numpy(), 
                        batch[:, 1].detach().cpu().numpy(), s = 10, alpha=0.3)
            return [(name, plt.gcf()),]  

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps = 100, start_scale = 0.05):
        super().__init__()
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.start_scale = start_scale
        self.each_step_scale = (1.0 - start_scale) / warmup_steps 
    def step(self, step_num):
        if step_num <= self.warmup_steps:
            # Linear warm-up
            scale = step_num * self.each_step_scale + self.start_scale
            for group in self.optimizer.param_groups:
                group['lr'] = group['initial_lr'] * scale
                
def create_formatter(data_form):
    if data_form == DataForm.IMG:
        return ImageTensorboardFormatter()
    elif data_form == DataForm.PCD:
        return PcdTensorboardFormatter()
    elif data_form == DataForm.VEC:
        return Vec2PCDTensorboardFormatter()
def write_data(writer, formatter, data_form, iter_id, input_map, prefix='train', additional_keys = []):
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
        if not name in ['input', 'target', 'condition', 'recon', 'recon_dpm', 'seg', 'gen'] + additional_keys: continue
        if batch is None or not torch.is_tensor(batch): 
            logger.debug('results of {} is not a tensor.'.format(name))
            continue
        # if batch.ndim < 2 or batch.ndim == 2 and batch.shape[1] != 2 and batch.shape[1] != 3:
        #     logger.debug('results of {} cannot be visualized.'.format(name))
        #     continue
        for tag, data in formatter(name, batch):
            if data == None: continue
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

def process_results(results, res, data_form, path = None,  
        slice_indices = None,
        is_supervised = False, 
        is_conditional = False, 
        additional_keys = [],
        pickle_results = False, pickle_path = "pickled",
        mode = 'train'):
    if not data_form == DataForm.GRAPH:
        res_dict = {
            'input': res['input'].cpu().detach().numpy(),
        }
        if 'condition' in res: 
            res_dict['condition'] = res['condition'].cpu().detach().numpy()
        if 'target' in res:
            res_dict['target'] = res['target'].cpu().detach().numpy()
        if 'path' in res:
            res_dict['path'] = res['path']
        if 'slice_indices' in res:
            res_dict['slice_indices'] = res['slice_indices']
        if 'cluster_logits' in res:
            res['cluster'] = logits_to_onehot_label(res['cluster_logits'], data_form)
        if 'cls_logits' in res:
            res['cls'] = logits_to_onehot_label(res['cls_logits'], data_form)
            res_dict['cls_logits'] = res['cls_logits'].cpu().detach().numpy()
        if 'seg_logits' in res:
            res['seg'] = logits_to_onehot_label(res['seg_logits'], data_form)
            res_dict['seg_logits'] = res['seg_logits'].cpu().detach().numpy()
        if 'latent' in res:
            res_dict['latent'] = res['latent'].cpu().detach().numpy()
        if 'recon' in res:
            res_dict['recon'] = res['recon'].cpu().detach().numpy()
        if 'seg' in res:
            res_dict['seg'] = res['seg'].cpu().detach().numpy()
        if 'cluster' in res:
            res_dict['cluster'] = res['cluster'].cpu().detach().numpy()
        if 'cls' in res:
            res_dict['cls'] = res['cls'].cpu().detach().numpy()
        if 'cluster_centers' in res:
            res_dict['cluster_centers'] = res['cluster_centers'].cpu().detach().numpy()
        for k in additional_keys:
            res_dict[k] = res[k].cpu().detach().numpy()
        if pickle_results:
            filename = os.path.join(pickle_path, f'{mode}_batch_{len(results)}.pkl')
            with open(filename, 'wb') as file:
                joblib.dump(res_dict, file)
            results.append(filename)
        else:
            results.append(res_dict)
    else:
        ## graph_related operation is not implemented
        raise NotImplementedError
def load_pickle(res_dict):
    if type(res_dict) == str:
        ### pickle path
        with open(res_dict, "rb") as file:
            res_dict = joblib.load(file)
    return res_dict
def extract_results(results, key):
    compacted_res = []
    for res_dict in results:
        res_dict = load_pickle(res_dict)
        if key in res_dict:
            compacted_res.append(res_dict[key])
        else:
            return None
    return np.concatenate(compacted_res)


def merge_patches_in_results(results, pickle_results = False, pickle_path = 'flemme-pickled'):
    ### merge patches to whole volumes
    ### only perform in test phase
    merged_path = None
    merged_patch_indices = []
    merged_patch_targets = []
    merged_patch_inputs = []
    merged_patch_segs = []
    merged_patch_seg_logits = []
    merged_shapes = [0, 0, 0]
    new_results = []
    def patch_to_volume():
        input_c = merged_patch_inputs[0].shape[0]
        label_c = merged_patch_targets[0].shape[0]
        tmp_input = np.zeros([input_c, ] + merged_shapes)
        weight_input = np.zeros([input_c, ] + merged_shapes)
        tmp_target = np.zeros([label_c, ] + merged_shapes)
        tmp_seg = np.zeros([label_c, ] + merged_shapes)
        tmp_seg_logits = np.zeros([label_c, ] + merged_shapes)
        weight_label = np.zeros([label_c, ] + merged_shapes)

        for patch_id, si in enumerate(merged_patch_indices):
            isi = (slice(0, input_c),) + si
            lsi = (slice(0, label_c),) + si
            tmp_input[isi] += merged_patch_inputs[patch_id]
            weight_input[isi] += 1
            tmp_target[lsi] += merged_patch_targets[patch_id]
            tmp_seg[lsi] += merged_patch_segs[patch_id]
            tmp_seg_logits[lsi] += merged_patch_seg_logits[patch_id]
            weight_label[lsi] += 1
        new_res_dict = {'input': (tmp_input / weight_input)[None, ...], 
                'target': (tmp_target / weight_label)[None, ...], 
                'seg': (tmp_seg / weight_label)[None], 
                'seg_logits': (tmp_seg_logits / weight_label)[None],
                'path': merged_path}
        if pickle_results:
            filename = os.path.join(pickle_path, f'test_merged_{len(new_results)}.pkl')
            with open(filename, 'wb') as file:
                joblib.dump(new_res_dict, file)
            new_results.append(filename)
        else:
            new_results.append(new_res_dict)
    for res_dict in tqdm(results, desc="merging patches"):
        res_dict = load_pickle(res_dict)
        for patch_id in range(len(res_dict['slice_indices'])):
            current_path = res_dict['path'][patch_id]
            if not current_path == merged_path:
                if merged_path is not None:
                    patch_to_volume()
                    merged_path = current_path 
                    merged_patch_indices = []
                    merged_patch_targets = []
                    merged_patch_inputs = []
                    merged_patch_segs = []
                    merged_patch_seg_logits = []
                    merged_shapes = [0, 0, 0]
                merged_path = current_path

            merged_patch_indices.append(res_dict['slice_indices'][patch_id])
            merged_patch_targets.append(res_dict['target'][patch_id])
            merged_patch_inputs.append(res_dict['input'][patch_id])
            merged_patch_segs.append(res_dict['seg'][patch_id])
            merged_patch_seg_logits.append(res_dict['seg_logits'][patch_id])
            tmp_shape = [idx.stop for idx in res_dict['slice_indices'][patch_id]]
            merged_shapes = [max(m, t) for m, t in zip(merged_shapes, tmp_shape)]
    patch_to_volume()
    return new_results

def create_evaluator(eval_configs, data_form, classification = False):

    if len(eval_configs) ==0:
        return None
    evaluator = {}
    for e_config in eval_configs:
        name = e_config.get('name')
        e = get_metrics(e_config, data_form=data_form, classification = classification)
        if e is not None:
            evaluator[name] = e
    return evaluator

def create_batch_evaluators(eval_metrics, data_form):
    if len(eval_metrics) == 0:
        return None
    evaluators = {}
    r_eval = create_evaluator(eval_metrics.get('recon', []), data_form)
    s_eval = create_evaluator(eval_metrics.get('seg', []), data_form)
    c_eval = create_evaluator(eval_metrics.get('cluster', []), data_form)
    cls_eval = create_evaluator(eval_metrics.get('cls', []), data_form, classification = True)

    if r_eval is not None: 
        evaluators['recon'] = r_eval
    if s_eval is not None: 
        evaluators['seg'] = s_eval
    if c_eval is not None: 
        evaluators['cluster'] = c_eval
    if cls_eval is not None:
        evaluators['cls'] = cls_eval
    return evaluators

    
## results is a batch of inputs, each batch can be read through pickle from external storage.
def evaluate_results(results, evaluators, data_form, verbose = False):
    eval_res = {}
    sample_num = 0
    if data_form == DataForm.GRAPH:
        raise NotImplementedError
    if verbose:
        results = tqdm(results, desc=f"evaluating")
    for res_dict in results:
        res_dict = load_pickle(res_dict)
        batch_size = len(res_dict['input'])
        sample_num += batch_size
        for eval_type in evaluators:
            if len(res_dict[eval_type]) == 0:
                logger.warning(f'This model doesn\'t predict {eval_type}')
                # exit(1)
            else:
                if eval_type not in eval_res:
                    eval_res[eval_type] = {}
                for (eval_metric, eval_func) in evaluators[eval_type].items():
                    if eval_type == 'cluster':
                        if eval_metric in eval_res[eval_type]:
                            eval_res[eval_type][eval_metric] += eval_func(res_dict['cluster'], res_dict['target']) * batch_size
                        else:
                            eval_res[eval_type][eval_metric] = eval_func(res_dict['cluster'], res_dict['target']) * batch_size
                    elif eval_type == 'cls':
                        if 'Soft' in eval_metric or 'TopK' in eval_metric:
                            if eval_metric in eval_res[eval_type]:
                                eval_res[eval_type][eval_metric] += eval_func(res_dict['cls_logits'], res_dict['target']) * batch_size
                            else:
                                eval_res[eval_type][eval_metric] = eval_func(res_dict['cls_logits'], res_dict['target']) * batch_size
                        else:
                            if eval_metric in eval_res[eval_type]:
                                eval_res[eval_type][eval_metric] += eval_func(res_dict['cls'], res_dict['target']) * batch_size
                            else:
                                eval_res[eval_type][eval_metric] = eval_func(res_dict['cls'], res_dict['target']) * batch_size
                    else:
                        tmp_res = []
                        if eval_type == 'recon':
                            #### supervised
                            if len(res_dict['target']) > 0:
                                zipped = zip(res_dict['recon'], res_dict['target'])
                            ### unsupervised
                            else:
                                zipped = zip(res_dict['recon'], res_dict['input'])
                        elif eval_type == 'seg':
                            if 'Soft' in eval_metric or 'TopK' in eval_metric:
                                zipped = zip(res_dict['seg_logits'], res_dict['target'])
                            else:
                                zipped = zip(res_dict['seg'], res_dict['target'])
                        for pred, target in zipped:
                            tmp_res.append(eval_func(pred, target))
                        tmp_res = sum(tmp_res) 
                        ## accuracy for different class
                        if isinstance(tmp_res, np.ndarray):
                            tmp_res = tmp_res.mean()
                        if eval_metric in eval_res[eval_type]:
                            eval_res[eval_type][eval_metric] += tmp_res
                        else:
                            eval_res[eval_type][eval_metric] = tmp_res
    for eval_type in eval_res:
        for eval_metric in eval_res[eval_type]:
            eval_res[eval_type][eval_metric] /= sample_num
    return eval_res


def process_input(t):
    x, y, c, si, p = None, None, None, None, None
    if len(t) == 2:
        x, p = t
    if len(t) == 3:
        x, y, p = t
    if len(t) == 4:
        if type(t[2]) == list and \
            type(t[2][0]) == tuple and \
            type(t[2][0][0]) == slice:
            x, y, si, p = t
        else:
            x, y, c, p = t
    ## patch
    # if len(t) == 5:
    #     x, y, c, si, p = t
    return x, y, c, si, p
def compute_loss(model, x, y, c, **kwargs):
    
    if model.is_supervised and model.is_conditional:
        losses, res = model.compute_loss(x, y = y, c = c, **kwargs)
    elif model.is_supervised:
        losses, res = model.compute_loss(x, y = y, **kwargs)
    ## using y as condition
    elif model.is_conditional:
        # losses, res = model.compute_loss(x, y, epoch<=2)
        losses, res = model.compute_loss(x, c = y, **kwargs)
    else:
        losses, res = model.compute_loss(x, **kwargs)
    return losses, res
def forward_pass(model, x, y, c, **kwargs):
    if model.is_supervised and model.is_conditional:
        res = model(x, c = c, **kwargs)
    elif model.is_conditional:
        res = model(x, c = y, **kwargs)
    else:
        res = model(x, **kwargs)
    return res

#### save tsne visualization
def construct_tsne_vis(embeddings, labels, vis_dim = 2, label_names = None, top_n = 10, 
    title = 't-SNE Visualization', size = 10, alpha = 0.3, 
    random_state = 42, perplexity = 30, remove_ticks = False, 
    color_map = 'jet', legend = True, **kwargs):
    if top_n > 0:
        unique_elements, counts = np.unique(labels, return_counts=True)
        # print(unique_elements, counts, labels)
        unique_elements = unique_elements[np.argsort(-counts)][:top_n]
        indices = (labels == unique_elements[0])
        for i in range(1, len(unique_elements)):
            indices = np.logical_or(indices, labels == unique_elements[i])
        embeddings = embeddings[indices]
        labels = labels[indices]
        unique_elements, inverse = np.unique(labels, return_inverse=True)
        labels = inverse.reshape(labels.shape)
        if label_names:
            label_names = [label_names[u] for u in unique_elements]
        # else:
        #     label_names = [u for u in unique_elements]

    if embeddings.shape[1] > vis_dim:
        tsne = TSNE(n_components = vis_dim, 
                random_state=random_state, 
                perplexity = perplexity, **kwargs)
        compressed_vec = tsne.fit_transform(embeddings)
    else:
        compressed_vec = embeddings

    label_count = labels.max() + 1
    ctable = get_color_table(color_map, label_count)
    if vis_dim == 2:
        fig, ax = plt.subplots()
        for l in range(label_count):
            tmp_vec = compressed_vec[labels == l]
            color = ctable[l % len(ctable)]
            ln = label_names[l] if label_names else ln
            ax.scatter(tmp_vec[:, 0], tmp_vec[:, 1], s = size, alpha=alpha, color = color, label = ln)
        if remove_ticks:
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            for tick in ax.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
        if legend:
            ax.legend(loc='best') 
    elif vis_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for l in range(label_count):
            tmp_vec = compressed_vec[labels == l]
            color = ctable[l % len(ctable)]
            ln = label_names[l] if label_names else ln
            ax.scatter(tmp_vec[:, 0], tmp_vec[:, 1], 
                    tmp_vec[:, 2], s = size, alpha = alpha, color = color, label = ln)
        if remove_ticks:
            for tick in ax.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            for tick in ax.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            for tick in ax.zaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)        
        if legend:
            ax.legend(loc='best') 
    plt.title(title)
    plt.tight_layout()
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
    ## segmentation label will be saved without channel dim
    ## because label_to_onehot function will expand the cahnnel dim
    if data_form == DataForm.IMG:
        if len(output.shape) == 3:
            ### segmentation will be saved as npy
            if segmentation:
                output = output.squeeze(0)
                save_npy(output_path+'.npy', output)
            ### reconstruction will be saved as png
            else:
                if output.max() > 1 or output.min() < 0:
                    output = normalize_img(output)
                save_img(output_path+'.png', (output * 255).astype('uint8'))
        ## CDHW
        elif len(output.shape) == 4:
            if segmentation:
                output = output.squeeze(0)
            save_itk(output_path+'.nii.gz', output)

    elif data_form == DataForm.PCD:
        if segmentation:
            np.savetxt(output_path+'.seg', output)
        elif output.shape[-1] == 3:
            save_ply(output_path+'.ply', output)
        ### not regular point cloud
        else:
            save_npy(output_path + '.npy', output)
    else:
        raise NotImplementedError

def get_load_function(suffix, transpose = False):
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

# def append_results(results, x, y, c, res, data_form, path = None,  
#         slice_indices = None,
#         is_supervised = False, 
#         is_conditional = False, 
#         additional_keys = []):
#     if not data_form == DataForm.GRAPH:
#         results['input'].append(x.cpu().detach().numpy())
#         if y is not None:
#             if is_supervised:
#                 results['target'].append(y.cpu().detach().numpy())
#             elif is_conditional:
#                 results['condition'].append(y.cpu().detach().numpy())
#         if c is not None and is_conditional and is_supervised:
#             results['condition'].append(c.cpu().detach().numpy())
                
#         ## should be a list of strings or none
#         ## only for test
#         if path is not None:
#             # if type(path[0]) == tuple:
#             #     results['path'] += list(path[0])
#             # else:
#             results['path'] += path
#         if slice_indices is not None:
#             results['slice_indices'] += slice_indices
#         if res:
#             if 'cluster_logits' in res:
#                 res['cluster'] = logits_to_onehot_label(res['cluster_logits'], data_form)
#             if 'cls_logits' in res:
#                 res['cls'] = logits_to_onehot_label(res['cls_logits'], data_form)
#                 results['cls_logits'].append(res['cls_logits'].cpu().detach().numpy())
#             if 'seg_logits' in res:
#                 res['seg'] = logits_to_onehot_label(res['seg_logits'], data_form)
#                 results['seg_logits'].append(res['seg_logits'].cpu().detach().numpy())
#             if 'latent' in res:
#                 results['latent'].append(res['latent'].cpu().detach().numpy())
#             if 'recon' in res:
#                 results['recon'].append(res['recon'].cpu().detach().numpy())
#             if 'seg' in res:
#                 results['seg'].append(res['seg'].cpu().detach().numpy())
#             if 'cluster' in res:
#                 results['cluster'].append(res['cluster'].cpu().detach().numpy())
#             if 'cls' in res:
#                 results['cls'].append(res['cls'].cpu().detach().numpy())
#             if 'cluster_centers' in res:
#                 results['cluster_centers'] = res['cluster_centers'].cpu().detach().numpy()
#             for k in additional_keys:
#                 if not k in results:
#                     results[k] = []
#                 if k in res:
#                     results[k].append(res[k].cpu().detach().numpy())
#     else:
#         results['input'].append(x)
#         if res:
#             if 'latent' in res:
#                 results['latent'].append(res['latent'].cpu().detach().numpy())
#             if 'recon' in res:
#                 results['recon'].append(res['recon'])
            
# def compact_results(results, data_form, additional_keys = []):
#     if not data_form == DataForm.GRAPH:
#         results['input'] = np.concatenate(results['input'])
#         if len(results['target']) > 0:
#             results['target'] = np.concatenate(results['target'])
#         if len(results['condition']) > 0:
#             results['condition'] = np.concatenate(results['condition'])        
#         if len(results['latent']) > 0:
#             results['latent'] = np.concatenate(results['latent'])
#         if len(results['recon']) > 0:
#             results['recon'] = np.concatenate(results['recon'])       
#         if len(results['seg']) > 0:
#             results['seg'] = np.concatenate(results['seg'])
#         if len(results['cluster']) > 0:
#             results['cluster'] = np.concatenate(results['cluster'])
#         if len(results['cls']) > 0:
#             results['cls'] = np.concatenate(results['cls'])
#         if len(results['cls_logits']) > 0:
#             results['cls_logits'] = np.concatenate(results['cls_logits'])
#         if len(results['seg_logits']) > 0:
#             results['seg_logits'] = np.concatenate(results['seg_logits'])
#         for k in additional_keys:
#             if len(results[k]) > 0:
#                 results[k] = np.concatenate(results[k])
#         ### merge patches to whole volumes
#         ### only perform in test phase
#         if 'slice_indices' in results and len(results['slice_indices']) > 0:
#             merged_paths = []
#             merged_patch_indices = []
#             merged_patch_targets = []
#             merged_patch_inputs = []
#             merged_patch_segs = []
#             merged_patch_seg_logits = []
#             merged_shapes = []
#             for patch_id in range(len(results['slice_indices'])):
#                 current_path = results['path'][patch_id]
#                 if len(merged_paths) == 0 or not current_path == merged_paths[-1]:
#                     merged_paths.append(current_path)
#                     merged_patch_indices.append([results['slice_indices'][patch_id], ])
#                     merged_patch_targets.append([results['target'][patch_id], ])
#                     merged_patch_inputs.append([results['input'][patch_id], ])
#                     merged_patch_segs.append([results['seg'][patch_id], ])
#                     merged_patch_seg_logits.append([results['seg_logits'][patch_id], ])
#                     merged_shapes.append([idx.stop for idx in results['slice_indices'][patch_id]])
#                 else:
#                     merged_patch_indices[-1].append(results['slice_indices'][patch_id])
#                     merged_patch_targets[-1].append(results['target'][patch_id])
#                     merged_patch_inputs[-1].append(results['input'][patch_id])
#                     merged_patch_segs[-1].append(results['seg'][patch_id])
#                     merged_patch_seg_logits[-1].append(results['seg_logits'][patch_id])
#                     tmp_shape = [idx.stop for idx in results['slice_indices'][patch_id]]
#                     merged_shapes[-1] = [max(m, t) for m, t in zip(merged_shapes[-1], tmp_shape)]

#             inputs = []
#             targets = []
#             segs = []
#             seg_logits = []
#             input_c = results['input'][0].shape[0]
#             label_c = results['target'][0].shape[0]
#             for i in range(len(merged_paths)):
#                 
#                 tmp_input = np.zeros([input_c, ] + merged_shapes[i])
#                 weight_input = np.zeros([input_c, ] + merged_shapes[i])
#                 tmp_target = np.zeros([label_c, ] + merged_shapes[i])
#                 tmp_seg = np.zeros([label_c, ] + merged_shapes[i])
#                 tmp_seg_logits = np.zeros([label_c, ] + merged_shapes[i])
#                 weight_label = np.zeros([label_c, ] + merged_shapes[i])

#                 for patch_id, si in enumerate(merged_patch_indices[i]):
#                     
#                     isi = (slice(0, input_c),) + si
#                     lsi = (slice(0, label_c),) + si
#                     tmp_input[isi] += merged_patch_inputs[i][patch_id]
#                     weight_input[isi] += 1
#                     tmp_target[lsi] += merged_patch_targets[i][patch_id]
#                     tmp_seg[lsi] += merged_patch_segs[i][patch_id]
#                     tmp_seg_logits[lsi] += merged_patch_seg_logits[i][patch_id]
#                     weight_label[lsi] += 1

#                 inputs.append(tmp_input / weight_input)
#                 targets.append(tmp_target / weight_label)
#                 segs.append(tmp_seg / weight_label)
#                 seg_logits.append(tmp_seg_logits / weight_label)

#             results['input'] = inputs
#             results['target'] = targets
#             results['seg'] = segs
#             results['seg_logits'] = seg_logits
#     return results

# ## evaluate the whole results which are stored in system memory
# def evaluate_results_fast(results, evaluators, data_form, verbose = False):

#     sample_num = len(results['input'])
#     if data_form == DataForm.GRAPH:
#         raise NotImplementedError
    
#     eval_res = {}
#     for eval_type in evaluators:
#         if len(results[eval_type]) == 0:
#             logger.warning(f'This model doesn\'t predict {eval_type}')
#         else:
#             eval_res[eval_type] = {}
#             if eval_type == 'cluster':
#                 for (eval_metric, eval_func) in evaluators[eval_type].items():
#                     eval_res[eval_type][eval_metric] = eval_func(results['cluster'], results['target'])
#             elif eval_type == 'cls':
#                 for (eval_metric, eval_func) in evaluators[eval_type].items():
#                     if 'Soft' in eval_metric or 'TopK' in eval_metric:
#                         eval_res[eval_type][eval_metric] = eval_func(results['cls_logits'], results['target'])
#                     else:
#                         eval_res[eval_type][eval_metric] = eval_func(results['cls'], results['target'])
#             else:
#                 for (eval_metric, eval_func) in evaluators[eval_type].items():
#                     tmp_res = []
#                     if eval_type == 'recon':
#                         #### supervised
#                         if len(results['target']) > 0:
#                             zipped = zip(results['recon'], results['target'])
#                         ### unsupervised
#                         else:
#                             zipped = zip(results['recon'], results['input'])
#                     elif eval_type == 'seg':
#                         if 'Soft' in eval_metric or 'TopK' in eval_metric:
#                             zipped = zip(results['seg_logits'], results['target'])
#                         else:
#                             zipped = zip(results['seg'], results['target'])
#                     if verbose:
#                         zipped = tqdm(zipped, total = len(results['input']) , desc=f"evaluating using {eval_metric}")
#                     for pred, target in zipped:
#                         tmp_res.append(eval_func(pred, target))
#                     tmp_res = sum(tmp_res) / sample_num
                    
#                     if isinstance(tmp_res, np.ndarray):
#                         tmp_res = tmp_res.mean()
#                     eval_res[eval_type][eval_metric] = tmp_res
#     return eval_res

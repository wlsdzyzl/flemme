from flemme.config import module_config
from . import img_transforms
from . import vol_transforms
from flemme.utils import DataForm, get_class, move_to_front, contains_one_of
from flemme.logger import get_logger
from torchvision.transforms import Compose

if module_config['point-cloud']:
    from . import pcd_transforms
if module_config['graph']:
    from . import graph_transforms

logger = get_logger('flemme.augment')
### label transforms table
seg_label_transform_table = {
    'img':['rotate', 'rotation', 'elastic', 'flip', 'crop', 'resize', 'totensor'],
    'pcd':['fixed', 'shuffle', 'totensor'],
    ## wait to be implemented
    'graph': ['fixed','totensor']
}
random_transforms = [
    'ElasticDeform', 'GaussianBlur', 'FixedPoints', 'ShufflePoints' ]
### ToTensor is not in the dataset.transforms
def get_transforms(trans_config_list, data_form = DataForm.IMG, img_dim = 2):
    transforms = []
    contain_fixed_points = False
    if len(trans_config_list) <= 0:
        return Compose(transforms)
    ### transforms
    if data_form == DataForm.IMG:
        if img_dim == 2:
            module = img_transforms
        elif img_dim == 3:
            module = vol_transforms
        else:
            logger.error(f'unsupported image dimension{img_dim}')
            exit(1)
    elif data_form == DataForm.PCD:
        module = pcd_transforms
        contain_fixed_points = move_to_front(trans_config_list, func = 
                        lambda a: a['name'] == 'FixedPoints')
    elif data_form == DataForm.GRAPH:
        module = graph_transforms
    else:
        raise NotImplementedError
    
    for tc in trans_config_list:
        trans_config = tc.copy()
        trans_name = trans_config.pop('name')
        trans_class = get_class(trans_name, module)
        if trans_class is not None:
            transforms.append(trans_class(**trans_config))
        else:
            logger.error(f"Unsupported transforms:{trans_name} for module {module}")
            exit(1)
    transforms = Compose(transforms)
    transforms.fixed_points = contain_fixed_points
    return transforms
def check_random_transforms(data_trans_config_list, label_trans_config_list):
    data_random_transform_list = [ t['name']  for t in data_trans_config_list 
        if t['name'] in random_transforms or 'random' in t['name'].lower() or 'noise' in t['name'].lower()]
    label_random_transform_list = [ t['name']  for t in label_trans_config_list 
        if t['name'] in random_transforms or 'random' in t['name'].lower() or 'noise' in t['name'].lower()]
    if len(data_random_transform_list) < len(label_random_transform_list):
        logger.warning('Label transforms introduce more random operations, are you sure this is what you want?')
    for drt, lrt in zip(data_random_transform_list, label_random_transform_list):
        if drt != lrt:
            logger.error('Transforms that introduce random operations shoule have the same order for data and label.')
            logger.error(f'random transforms for data: {data_random_transform_list}')
            logger.error(f'random transforms for label: {label_random_transform_list}')
            exit(1)
def check_consistent_transforms(data_trans_config_list, label_trans_config_list, data_form):
    if data_form == DataForm.IMG:
        selector = seg_label_transform_table['img']
    elif data_form == DataForm.PCD:
        selector = seg_label_transform_table['pcd']
    elif data_form == DataForm.GRAPH:
        selector = seg_label_transform_table['graph']

    data_consistent_transform_list = [ t['name']  for t in data_trans_config_list 
        if contains_one_of(t['name'].lower(), selector)]
    label_consistent_transform_list = [ t['name']  for t in label_trans_config_list 
        if contains_one_of(t['name'].lower(), selector)]
        
    if data_consistent_transform_list != label_consistent_transform_list:
        logger.error('Transforms that change the shape or order of data should be consistent.')
        logger.error(f'Consistent transforms for data: {data_consistent_transform_list}')
        logger.error(f'Consistent transforms for label: {label_consistent_transform_list}')
        exit(1)
def select_label_transforms(trans_config_list, data_form):
    label_config_list = []
    selector = []
    if data_form == DataForm.IMG:
        selector = seg_label_transform_table['img']
    elif data_form == DataForm.PCD:
        selector = seg_label_transform_table['pcd']
    elif data_form == DataForm.GRAPH:
        selector = seg_label_transform_table['graph']
    
    for t in trans_config_list:
        if contains_one_of(t['name'].lower(), selector):
            ### for 3D image, the name is elastic deform instead of elastic transform (torch)
            if t['name'] == 'ElasticDeform':
                t['spline_order'] = 0
            label_config_list.append(t)
    return label_config_list
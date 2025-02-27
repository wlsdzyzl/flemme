from .medshapenet import *
from .shapenet import *
from flemme.logger import get_logger
logger = get_logger('label_dict')
def get_cls_label(name):
    if name.lower() == 'medshapenet':
        return medshapenet_cls_label
    elif name.lower() == 'shapenet':
        return shapenet_cls_label
    else:
        logger.error(f'Unknown classification label for dataset {name}')
    return None
def get_label_cls(name):
    if name.lower() == 'medshapenet':
        return coarse_label_to_organ
    elif name.lower() == 'shapenet':
        return shapenet_label_cls
    else:
        logger.error(f'Unknown classification label for dataset {name}')
    return None
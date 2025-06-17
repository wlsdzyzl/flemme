from .medpoints import *
from .shapenet import *
from .imagecas import *
from flemme.logger import get_logger
logger = get_logger('label_dict')
def get_cls_label(name):
    if name.lower() == 'medpoints':
        return medpoints_cls_label
    elif name.lower() == 'shapenet':
        return shapenet_cls_label
    elif name.lower() == 'imagecas':
        return imagecas_cls_label
    else:
        logger.error(f'Unknown classification label for dataset {name}')
    return None
def get_label_cls(name):
    if name.lower() == 'medpoints':
        return coarse_label_to_organ
    elif name.lower() == 'shapenet':
        return shapenet_label_cls
    elif name.lower() == 'imagecas':
        return imagecas_label_cls
    else:
        logger.error(f'Unknown classification label for dataset {name}')
    return None
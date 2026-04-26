from .medpoints import *
from .shapenet13 import *
from .shapenet16 import *
from .imagecas import *
from .medsdf import *
from flemme.logger import get_logger
logger = get_logger('label_dict')
def get_cls_label(name):
    if name.lower() == 'medpoints':
        return medpoints_cls_label
    elif name.lower() == 'medsdf':
        return medsdf_cls_label
    elif name.lower() == 'shapenet13':
        return shapenet13_cls_label
    elif name.lower() == 'shapenet16':
        return shapenet16_cls_label
    elif name.lower() == 'imagecas':
        return imagecas_cls_label
    else:
        logger.error(f'Unknown classification label for dataset {name}')
    return None
def get_label_cls(name):
    if name.lower() == 'medpoints':
        return coarse_label_to_organ
    elif name.lower() == 'medsdf':
        return medsdf_label_cls
    elif name.lower() == 'shapenet13':
        return shapenet13_label_cls
    elif name.lower() == 'shapenet16':
        return shapenet16_label_cls
    elif name.lower() == 'imagecas':
        return imagecas_label_cls
    else:
        logger.error(f'Unknown classification label for dataset {name}')
    return None
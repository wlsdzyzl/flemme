from .medshapenet import medshapenet_cls_label
from .shapenet import shapenet_cls_label
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
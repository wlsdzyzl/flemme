from flemme.config import module_config
from .img import *
from .vol_patch import PatchImgSegDataset, MultiModalityPatchImgSegDataset
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data._utils.collate import default_collate
from flemme.utils import DataForm
from torchvision.transforms import Compose
from flemme.augment import get_transforms, select_label_transforms, check_random_transforms
from flemme.logger import get_logger
from .label_dict import get_cls_label



logger = get_logger('dataset')
img_dataset_dict = {
    'ImgDataset': ImgDataset,
    'ImgClsDataset': ImgClsDataset,
    'ImgSegDataset': ImgSegDataset,
    'ImgReconDataset': ImgReconDataset,
    'MultiModalityImgSegDataset': MultiModalityImgSegDataset,
    'PatchImgSegDataset': PatchImgSegDataset,
    'MultiModalityPatchImgSegDataset': MultiModalityPatchImgSegDataset,
    'MNIST': MNISTWrapper,
    'CIFAR10': CIFAR10Wrapper,
    'CelebA': CelebAWrapper,
    }
pcd_dataset_dict = {}
vec_dataset_dict = {}
graph_dataset_dict = {}

process_label_datasets = [ImgSegDataset, MultiModalityImgSegDataset, 
                        PatchImgSegDataset, MultiModalityPatchImgSegDataset]
process_target_datasets = [ImgReconDataset, ]

if module_config['point-cloud']:
    from .pcd import *
    from .point import PointDataset, ToyDataset
    pcd_dataset_dict = {
        'PcdDataset': PcdDataset,
        'PcdClsDataset': PcdClsDataset,
        'PcdSegDataset': PcdSegDataset,
        'PcdReconDataset': PcdReconDataset,
        'PcdReconWithClassLabelDataset': PcdReconWithClassLabelDataset,
        }
    vec_dataset_dict = {
        'PointDataset': PointDataset,
        'ToyDataset': ToyDataset,
    }
    
    process_label_datasets.append(PcdSegDataset)
    process_target_datasets += [PcdReconDataset, PcdReconWithClassLabelDataset]
if module_config['graph']:
    from .graph import *
    from torch_geometric.loader import DataLoader as GraphLoader
    graph_dataset_dict = {'GraphDataset': GraphDataset,
        'GraphShapeNet': GraphShapeNetWrapper,}

def custom_collate(batch):
    inputs = zip(*batch)
    inputs = tuple( list(i) if type(i[0]) == tuple and type(i[0][0]) == slice else None if i[0] is None else default_collate(i) for i in inputs)
    return inputs

def create_loader(loader_config):
    """
    Returns dictionary containing the loaders (torch.utils.data.DataLoader).
    """

    
    loader = {}
    mode = loader_config.get('mode', 'train')
    logger.info('Creating {} set loaders...'.format(mode))
    # get dataset class
    dataset_config = loader_config.get('dataset', None)
    assert dataset_config is not None, 'Cannot get the dataset configuration!'
    data_form = DataForm.IMG
    dataset_cls_str = dataset_config.get('name', None)
    process_label = False
    process_target = False
    if dataset_cls_str is None:
        dataset_cls_str = 'ImgSegDataset'
        logger.warning(f"Cannot find dataset class in the config. Using default '{dataset_cls_str}'.")

    data_path_list = loader_config.get('data_path_list', None)
    if data_path_list is not None:
        assert type(data_path_list) == list, \
        "If data_path is provided in loader config, it should be a list."
    data_suffix_list = loader_config.get('data_suffix_list', None)
    if data_suffix_list is not None:
        assert type(data_suffix_list) == list and len(data_suffix_list) == len(data_path_list), \
            "If suffix is provided in loader config, it should be a list and has the equal length to the data_path_list."
    dataset_dict = img_dataset_dict | pcd_dataset_dict | vec_dataset_dict | graph_dataset_dict
    assert dataset_cls_str in dataset_dict, f'Unsupported dataset class: {dataset_cls_str}'
    
    dataset_class = dataset_dict[dataset_cls_str]

    if dataset_cls_str in img_dataset_dict:
        data_form = DataForm.IMG
    elif dataset_cls_str in pcd_dataset_dict:
        data_form = DataForm.PCD
    elif dataset_cls_str in vec_dataset_dict:
        data_form = DataForm.VEC
    elif dataset_cls_str in graph_dataset_dict:
        data_form = DataForm.GRAPH

    if dataset_class in process_label_datasets:
        process_label = True 
    if dataset_class in process_target_datasets:
        process_target = True
    img_dim = None
    if data_form == DataForm.IMG:
        img_dim = dataset_config.get('dim', 2)
    loader['data_form'] = data_form

    #### set transforms
    ## for 2D image, we use official implementation from torch_vision
    ## for 3D point cloud, we use custom transforms

    trans_config_list = loader_config.get('data_transforms', [])
    data_transforms = get_transforms(trans_config_list, data_form, img_dim=img_dim)
    data_transforms = Compose(data_transforms)
    dataset_config['data_transform'] = data_transforms


    ### label_transform and label
    ### None means it has the same suffix with data
    label_suffix_list = loader_config.get('label_suffix_list', None)
    if label_suffix_list is not None:
        assert type(label_suffix_list) == list and len(label_suffix_list) == len(data_path_list), \
            "If label_suffix_list is provided in loader config, it should be a list and has the equal length to the data_path_list."


    label_trans_config_list = loader_config.get('label_transforms', None)
    if process_label or label_trans_config_list is not None:
        
        if label_trans_config_list is None:
            logger.info('There is no specified transforms for labels, we would generate the necessary transforms for labels automatically.')
            label_trans_config_list = \
                select_label_transforms(trans_config_list, data_form)

        ### check random operations are in the same order.
        check_random_transforms(trans_config_list, label_trans_config_list)

        label_transforms = get_transforms(label_trans_config_list, data_form, img_dim=img_dim)
        label_transforms = Compose(label_transforms)
        dataset_config['label_transform'] = label_transforms

    ### None means it has the same suffix with data
    target_suffix_list = loader_config.get('target_suffix_list', None)
    if target_suffix_list is not None:
        assert type(target_suffix_list) == list and len(target_suffix_list) == len(data_path_list), \
            "If target_suffix_list is provided in loader config, it should be a list and has the equal length to the data_path_list."

    target_trans_config_list = loader_config.get('target_transforms', None)
    if process_target or target_trans_config_list is not None:
        
        if target_trans_config_list is None:
            logger.info('There is no specified transforms for targets, we would generate the necessary transforms for targets automatically.')
            target_trans_config_list = \
                select_label_transforms(trans_config_list, data_form)

        ### check random operations are in the same order.
        check_random_transforms(trans_config_list, target_trans_config_list)

        target_transforms = get_transforms(target_trans_config_list, data_form, img_dim=img_dim)
        target_transforms = Compose(target_transforms)
        dataset_config['target_transform'] = target_transforms

    
    num_workers = loader_config.get('num_workers', 1)
    logger.info(f'Number of workers for {mode} dataloader: {num_workers}')
    batch_size = loader_config.get('batch_size', 1)
    drop_last = loader_config.get('drop_last', False)

    ##### multi-gpu is not supported yet
    # # when multiple gpus are available
    # if torch.cuda.device_count() > 1 and not config['device'].type == 'cpu':
    #     logger.info(
    #         f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}')
    #     batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for {mode} loader: {batch_size}')
    

    ### for classification dataset
    cls_label = dataset_config.get('cls_label', None)
    if cls_label is not None and not type(cls_label) == dict:
        dataset_config['cls_label'] = get_cls_label(cls_label)
    shuffle = loader_config.get('shuffle', None)
    # if shuffle is not set, we use the mode to determine whether the datasets need to be shuffled
    if shuffle is None:
        if mode == 'train':
            shuffle = True
        else:
            shuffle = False
    
    dataset_config['mode'] = mode
    datasets = []
    if data_path_list is not None and len(data_path_list) > 0:
        for idx, tmp_path in enumerate(data_path_list):
            dataset_config['data_path'] = tmp_path
            if data_suffix_list is not None:
                dataset_config['data_suffix'] = data_suffix_list[idx]
            if label_suffix_list is not None:
                dataset_config['label_suffix'] = label_suffix_list[idx]
            if target_suffix_list is not None:
                dataset_config['target_suffix'] = target_suffix_list[idx]
            datasets.append(dataset_class(**dataset_config))
    else:
        ### single dataset, all related information is contained in dataset configuration
        datasets.append(dataset_class(**dataset_config))
    if data_form == DataForm.GRAPH:
        loader['data_loader'] = GraphLoader(ConcatDataset(datasets), 
            batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, drop_last = drop_last)
    else:
        loader['data_loader'] = DataLoader(ConcatDataset(datasets), 
            batch_size = batch_size, shuffle = shuffle, 
            num_workers = num_workers, drop_last = drop_last,
            collate_fn = custom_collate)
    return loader
from flemme.config import module_config
from .img import *
from torch.utils.data import DataLoader, ConcatDataset
from flemme.utils import DataForm
from torchvision.transforms import Compose
from flemme.augment import get_transforms, select_label_transforms, check_random_transforms
from flemme.logger import get_logger
if module_config['point-cloud']:
    from .pcd import *
    from .point import PointDataset, ToyDataset
if module_config['graph']:
    from .graph import *
    from torch_geometric.loader import DataLoader as GraphLoader
logger = get_logger('dataset')



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
    if dataset_cls_str == 'ImgDataset':
        dataset_class = ImgDataset
    elif dataset_cls_str == 'ImgSegDataset' or dataset_cls_str == 'ImgReconDataset':
        dataset_class = ImgSegDataset
        process_label = True
    elif dataset_cls_str == 'MultiModalityImgSegDataset':
        dataset_class = MultiModalityImgSegDataset
        process_label = True
    elif dataset_cls_str == 'MNIST':
        dataset_class = MNISTWrapper
    elif dataset_cls_str == 'CIFAR10':
        dataset_class = CIFAR10Wrapper
    elif dataset_cls_str == 'CelebA':
        dataset_class = CelebAWrapper
    elif dataset_cls_str == 'PCDDataset':
        dataset_class = PCDDataset
        data_form = DataForm.PCD
    elif dataset_cls_str == 'PCDSegDataset':
        dataset_class = PCDSegDataset
        data_form = DataForm.PCD
        process_label = True
    elif dataset_cls_str == 'PointDataset':
        dataset_class = PointDataset
        data_form = DataForm.VEC
    elif dataset_cls_str == 'ToyDataset':
        dataset_class = ToyDataset
        data_form = DataForm.VEC
    elif dataset_cls_str == 'GraphDataset':
        dataset_class = GraphDataset
        data_form = DataForm.GRAPH
    elif dataset_cls_str == 'GraphShapeNet':
        dataset_class = GraphShapeNetWrapper
        data_form = DataForm.GRAPH
    else:
        raise RuntimeError(f'Unsupported dataset class: {dataset_cls_str}')
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
    if process_label:
        label_trans_config_list = loader_config.get('label_transforms', None)
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
        label_suffix_list = loader_config.get('label_suffix_list', None)
        if label_suffix_list is not None:
            assert type(label_suffix_list) == list and len(label_suffix_list) == len(data_path_list), \
                "If label_suffix_list is provided in loader config, it should be a list and has the equal length to the data_path_list."

    
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
            if process_label and label_suffix_list is not None:
                dataset_config['label_suffix'] = label_suffix_list[idx]
            datasets.append(dataset_class(**dataset_config))
    else:
        ### single dataset, all related information is contained in dataset configuration
        datasets.append(dataset_class(**dataset_config))
    if data_form == DataForm.GRAPH:
        loader['data_loader'] = GraphLoader(ConcatDataset(datasets), 
            batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, drop_last = drop_last)
    else:
        loader['data_loader'] = DataLoader(ConcatDataset(datasets), 
            batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, drop_last = drop_last)
    return loader
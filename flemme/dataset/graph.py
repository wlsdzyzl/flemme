import numpy as np

import torch
from torch_geometric.data import Dataset, download_url
from torch_geometric.datasets import ShapeNet, ModelNet
import os
import glob
from flemme.utils import load_graph
from flemme.logger import get_logger
from torch_geometric.data import Batch, Data
#### Pure graph dataset
### current only support point cloud and mesh
logger = get_logger('graph_dataset')
class GraphDataset(Dataset):
    def __init__(self, data_path, 
                  vertex_features = None, data_transform = None, 
                  pre_transform=None, pre_filter=None,
                  mode = 'train', data_dir = '', 
                  data_suffix = '.ply', processed_dir = '.', **kwargs):
        super().__init__(data_path, transform, pre_transform, pre_filter)
        logger.info("loading data from the directory: {}".format(data_path))
        self.graph_path_list = sorted(glob(os.path.join(data_path+'/' + data_dir, "*" + data_suffix)))
        self.mode = mode
        self.data_transform = data_transform
        self.data_path = data_path
        self.vertex_features = None
    def process(self):
        for idx, graph_path in enumerate(self.graph_path_list):
            # Read data from `raw_path`.
            data = load_graph(graph_path, vertex_features = vertex_features)
            if type(data) == tuple:
                vertices, edges = data
                pos = torch.from_numpy(vertices[:, 0:3])
                ## read node features
                x = torch.from_numpy(vertices[:, 3:]) if vertices.shape[1] > 3 else None
                edge_index = torch.from_numpy(edges) if edges is not None else None
                data = Data(x = x, pos = pos, edge_index = edge_index)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, os.path.join(self.processed_dir, f'gdata_{idx}.pt'))
    def len(self):
        return len(self.graph_path_list)
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'gdata_{idx}.pt'))
        return data

class GraphShapeNetWrapper(ShapeNet):
    def __init__(self, data_path, categories= None, 
                  include_normals = None, 
                  data_transform = None, 
                  pre_transform=None, 
                  pre_filter=None,
                  mode = 'train', name = '10', 
                  **kwargs):
        super().__init__(root = data_path, split = mode, 
          categories = categories,
          include_normals = include_normals,
          transform = data_transform, 
          pre_transform = pre_transform, 
          pre_filter = pre_filter)
        self.data_path = data_path
        self.mode = mode
class GraphModelNetWrapper(ModelNet):
    def __init__(self, data_path, data_transform = None, 
                  pre_transform=None, pre_filter=None,
                  mode = 'train', name = '10', 
                  **kwargs):
        super().__init__(root = data_path, train = (mode == 'train'), 
          transform = data_transform, pre_transform = pre_transform, 
          pre_filter = pre_filter)
        self.data_path = data_path
        self.mode = mode
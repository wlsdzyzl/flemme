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
                  data_transform = None, 
                  pre_transform=None, pre_filter=None,
                  mode = 'train', data_dir = '', 
                  data_suffix = '.ply', 
                  processed_dir = 'processed',
                  vertex_features = None, **kwargs):
        logger.info("loading data from the directory: {}".format(data_path))
        self.mode = mode
        self.data_path = data_path
        root = os.path.join(data_path, data_dir)
        processed_dir = os.path.join(data_path, processed_dir)
        self.vertex_features = vertex_features
        self.data_suffix = data_suffix
        self.graph_path_list = sorted(glob(os.path.join(root, "*" + self.data_suffix)))
        self.processed_path_list = [os.path.join(processed_dir, f'gdata_{idx}.pt') for idx in range(len(self.graph_path_list))]

        super().__init__(os.path.join(data_path, data_dir), 
            transform = data_transform, 
            pre_transform = pre_transform, 
            pre_filter = pre_filter)
        
    @property
    def processed_paths(self):
        return self.processed_path_list

    def process(self):
        for idx, graph_path in enumerate(self.graph_path_list):
            # Read data from `raw_path`.
            data = load_graph(graph_path, vertex_features = self.vertex_features)
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
            torch.save(data, self.processed_path_list[idx])
    def len(self):
        return len(self.graph_path_list)
    def get(self, idx):
        data = torch.load(self.processed_path_list[idx])
        return data

class GraphShapeNetWrapper(ShapeNet):
    def __init__(self, data_path, categories= None, 
                  include_normals = None, 
                  data_transform = None, 
                  pre_transform=None, 
                  pre_filter=None,
                  mode = 'train',
                  **kwargs):
        super().__init__(root = data_path, split = mode, 
          categories = categories,
          include_normals = include_normals,
          transform = data_transform, 
          pre_transform = pre_transform, 
          pre_filter = pre_filter)
        self.data_path = data_path
        self.mode = mode
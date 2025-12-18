import numpy as np
import random

import os
from glob import glob
from flemme.utils import load_pcd, save_pcd, rkdirs
import numpy as np

def partialize(points, num_remove = None, remove_ratio = 0.1):
    if not num_remove:
        num_remove = max(int(remove_ratio * len(points)), 1)
    center_point = points[random.randint(0, len(points) - 1), ]
    distances = np.linalg.norm(points - center_point, axis=1)
    knn_indices = np.argsort(distances)[:num_remove]
    remaining_points = np.delete(points, knn_indices, axis=0)
    return remaining_points

remove_ratio = 0.2
dataset_path = '/media/wlsdzyzl/DATA/datasets/pcd/MedPointS/'
kfolds = glob(os.path.join(dataset_path, 'classification/*'))

for current_fold in kfolds:
    sub_dirs = glob(current_fold+'/*')
    sub_files = []
    for sub_dir in sub_dirs:
        rkdirs(sub_dir.replace('classification', 'completion'))
        sub_files = sub_files + glob(sub_dir + '/*.ply' )  
    for pfile in sub_files:
        print(f'generating partial points for {pfile} ...')
        pcd = load_pcd(pfile)
        ppcd = partialize(pcd, remove_ratio=0.2)
        save_pcd(pfile.replace('classification', 'completion'), ppcd)

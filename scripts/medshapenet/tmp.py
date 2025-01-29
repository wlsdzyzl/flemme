import shutil
from glob import glob
import os
from flemme.utils import rkdirs
dataset_path = '/media/wlsdzyzl/DATA/datasets/pcd/MedShapeNetPCD/completion'
# target_path = '/media/wlsdzyzl/DATA/datasets/pcd/MedShapeNetPCD/completion'
organ_dirs =  glob(dataset_path +'/*')
kfolds = glob(dataset_path + '/brain/*')
for o in organ_dirs:
    for kf in kfolds:
        base_organ = os.path.basename(o)
        base_kf = os.path.basename(kf)
        source_dir = os.path.join(o, base_kf, '*')
        target_dir = os.path.join(dataset_path, base_kf, base_organ)
        rkdirs(target_dir)
        for sd in glob(source_dir):
            print(f'from {sd} to {target_dir}')
            shutil.move(sd, target_dir)
        # exit(1)
        # rkdirs(target_dir)
        # shutil.move(source_dir, target_dir)
    

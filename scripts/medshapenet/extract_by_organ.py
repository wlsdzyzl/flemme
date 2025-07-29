from flemme.utils import mkdirs
from flemme.logger import get_logger
import os
logger = get_logger('scripts::extract_by_organ')
## 'bladder'
organ_list = ['gallbladder', 'bladder', 'brain', 'heart', 'liver', 'pancreas', 'skull', 
              'spleen', 'colon', 'stomach', 'duodenum', 'uterus', 'trachea', 
              'bronchie']
all_stl_files = []
medshapenet_path = '/media/wlsdzyzl/DATA1/datasets/pcd/MedShapeNet/MedShapeNetDataset.txt'
output_path = '/media/wlsdzyzl/DATA1/datasets/pcd/MedShapeNet'
with open(medshapenet_path, 'r') as f:
    all_stl_files = f.readlines()

organ_files = {}
for organ in organ_list:
    organ_files[organ] = [p for p in all_stl_files if ('_' + organ+'.stl' in p) or ('_' + organ +'.nii.g_1.stl' in p)]

for organ, files in organ_files.items():
    logger.info(f"Organ: {organ}, Number of files: {len(files)}")
    mkdirs(output_path + '/' + organ)
    with open(f'{output_path}/{organ}/wget.txt', 'w') as f:
        for file in files:
            f.write(file)

# ### use wget to download the files
# command = "wget --content-disposition --trust-server-names -i {} -P {} --no-show-progress"
# for organ in organ_list:
#     logger.info(f"Downloading files for {organ} to {output_path}/{organ}")
#     cmd = command.format(f'{output_path}/{organ}/wget.txt', f'{output_path}/{organ}')
#     logger.info(cmd)
#     os.system(cmd)
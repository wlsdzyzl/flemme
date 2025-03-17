import os
from glob import glob
from flemme.utils import load_pcd, save_pcd, rkdirs
import numpy as np
from flemme.dataset.label_dict.MedPointS import coarse_labels, coarse_label_to_organ

dataset_path = '/media/wlsdzyzl/DATA/datasets/pcd/MedPointS'
selected_number_paths = glob(os.path.join(dataset_path, 'by_number/*')) 
sample_num = 16384
broken_stls = []
unknown_labels = []
organ_counts = {}
for _, on in coarse_label_to_organ.items():
    dir_path = os.path.join(dataset_path, f'classification/{on}')
    rkdirs(dir_path)
for p in selected_number_paths:
    if p[-1] == '/': p = p[:-1]
    print(f"Processing {p} ...")
    organ_paths = glob(f'{p}/*')
    person_organ = {}
    for op in organ_paths:
        # print(f"loading stl from {op}")
        try:
            tmp_pcd = load_pcd(op)
        except Exception as e:
            broken_stls.append(op)
        else:
            if len(tmp_pcd) == 0:
                broken_stls.append(op)
                continue
            if len(tmp_pcd) > sample_num:
                tmp_pcd = tmp_pcd[ np.random.choice(len(tmp_pcd), sample_num, replace = False)]

            #### get the label
            oname = os.path.basename(op).split('.', maxsplit=1)[0].split('_', maxsplit=1)[1].replace('_', '')
            if not oname in coarse_labels:
                print('unknown label:', oname)
                unknown_labels.append(oname)
    
            ## from label to the organ name
            tmp_clabel = coarse_labels[oname]
            
            if not tmp_clabel:
                continue
            organ_name = coarse_label_to_organ[tmp_clabel]
            if not organ_name in person_organ:
                person_organ[organ_name] = []
            person_organ[organ_name].append(tmp_pcd)            
    ### combine to organ
    for on in person_organ:
        if not on in organ_counts:
            organ_counts[on] = 1
        else:
            organ_counts[on] += 1
        og_pcd = np.concatenate(person_organ[on], axis = 0)
        if len(og_pcd) > sample_num:
            choice = np.random.choice(len(og_pcd), sample_num, replace = False)
            og_pcd = og_pcd[choice]
        print(f'saving sampled point cloud of {p}\'s {on}')
        ### save_pcd and labels
        filename = p.split('/')[-1]
        save_pcd(os.path.join(dataset_path, f'classification/{on}/{filename}.ply'), og_pcd)


### Exception checking
print('broken_stls:')
print(broken_stls)
print('-------------')
print('unknown_labels:')
print(unknown_labels)
print('-------------')
print('organ_counts:')
print(organ_counts)
import os
from glob import glob
from flemme.utils import load_pcd, save_pcd
import numpy as np
from flemme.dataset.label_dict.MedPointS import fine_labels, coarse_labels
from flemme.color_table import color_table
dataset_path = '/media/wlsdzyzl/DATA/datasets/pcd/MedShapeNet'
output_path = '/data/guoqingzhang/datasets/MedPointS'
selected_number_paths = glob(os.path.join(dataset_path, 'by_number/*')) 
sample_num = 65536
broken_stls = []
unknown_labels = []
label_counts = {}
for p in selected_number_paths:
    if p[-1] == '/': p = p[:-1]
    print(f"Processing {p} ...")
    organ_paths = glob(f'{p}/*')
    pcd = []
    pcd_flabel = []
    pcd_clabel = []
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
            if not oname in fine_labels:
                print('unknown label:', oname)
                unknown_labels.append(oname)
            if not oname in label_counts:
                label_counts[oname] = 1
            else:
                label_counts[oname] += 1
            tmp_flabel = fine_labels[oname]
            if tmp_flabel == 0:
                continue
            pcd.append(tmp_pcd)
            pcd_flabel = pcd_flabel + [tmp_flabel,] * len(tmp_pcd)

            ## coarse label
            tmp_clabel = coarse_labels[oname]
            pcd_clabel = pcd_clabel + [tmp_clabel,] * len(tmp_pcd)

    pcd = np.concatenate(pcd, axis = 0)
    pcd_flabel, pcd_clabel = np.array(pcd_flabel), np.array(pcd_clabel)
    if len(pcd) > sample_num:
        choice = np.random.choice(len(pcd), sample_num, replace = False)
        pcd = pcd[choice]
        pcd_flabel = pcd_flabel[choice]
        pcd_clabel = pcd_clabel[choice]
    print(f'saving sampled point cloud and labels of {p}')
    ### save_pcd and labels
    filename = p.split('/')[-1]
    save_pcd(os.path.join(output_path, f'segmentation/pcd/{filename}.ply'), pcd)
    np.savetxt(os.path.join(output_path, f'segmentation/fine_label/{filename}.seg'), pcd_flabel)
    np.savetxt(os.path.join(output_path, f'segmentation/coarse_label/{filename}.seg'), pcd_clabel)
    ### save colorized pcds
    pcd_color = color_table[(pcd_clabel - 1) % len(color_table)]
    save_pcd(os.path.join(output_path, f'segmentation/colorized_pcd/{filename}.ply'), (pcd, pcd_color))

### Exception checking
print('broken_stls:')
print(broken_stls)
print('-------------')
print('unknown_labels:')
print(unknown_labels)
print('-------------')
print('label_counts:')
print(label_counts)
import os
from glob import glob
from flemme.utils import load_pcd, save_pcd
import numpy as np
from label_map import fine_labels, coarse_labels
path = '/media/wlsdzyzl/DATA/datasets/pcd/MedShapeNet/'
selected_number_paths = glob(os.path.join(path, 'by_number/*'))
sample_num = 16384
broken_stls = []
unknown_labels = []
for p in selected_number_paths:
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
            print(f"loading stl from {op}")
            print(e)
            broken_stls.append(op)
        # else:
        #     if len(tmp_pcd) > sample_num:
        #         tmp_pcd = tmp_pcd[ np.random.choice(len(tmp_pcd), sample_num, replace = False)]
        #     pcd.append(tmp_pcd)
        ## fine label
        oname = os.path.basename(op).split('.', maxsplit=1)[0].split('_', maxsplit=1)[1].replace('_', '')
        if not oname in fine_labels:
            print('unknown label:', oname)
            unknown_labels.append(oname)
    #     tmp_flabel = fine_labels[oname]
    #     pcd_flabel = pcd_flabel + [tmp_flabel,] * len(tmp_pcd)

    #     ## coarse label
    #     tmp_clabel = coarse_labels[oname]
    #     pcd_clabel = pcd_clabel + [tmp_clabel,] * len(tmp_pcd)

    # pcd = np.concatenate(pcd, axis = 0)
    # pcd_flabel, pcd_clabel = np.array(pcd_flabel), np.array(pcd_clabel)
    # if len(pcd) > sample_num:
    #     choice = np.random.choice(len(pcd), sample_num, replace = False)
    #     pcd = pcd[choice]
    #     pcd_flabel = pcd_flabel[choice]
    #     pcd_clabel = pcd_clabel[choice]
    # print(f'saving sampled point cloud and labels of {p}')
    # ### save_pcd and labels
    # save_pcd(os.path.join(p.replace('by_number', 'segmentation/pcd'), '.ply'), pcd)
    # np.savetxt(os.path.join(p.replace('by_number', 'segmentation/fine_label'), '.seg'), pcd_flabel)
    # np.savetxt(os.path.join(p.replace('by_number', 'segmentation/coarse_label'), '.seg'), pcd_clabel)

### Exception checking
print('broken_stls:')
print(broken_stls)
print('-------------')
print('unknown_labels:')
print(unknown_labels)
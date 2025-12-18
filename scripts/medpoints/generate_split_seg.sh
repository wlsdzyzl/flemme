#!/bin/bash
python -u generate_seg_dataset.py
python ../random_split_k_fold.py -p /media/wlsdzyzl/DATA/datasets/pcd/MedPointS/segmentation/ --sub_dirs coarse_label,colorized_pcd,fine_label,pcd --suffix .seg,.ply,.seg,.ply -o /media/wlsdzyzl/DATA/datasets/pcd/MedPointS/segmentation/

rm -rf /media/wlsdzyzl/DATA/datasets/pcd/MedPointS/segmentation/*pcd
rm -rf /media/wlsdzyzl/DATA/datasets/pcd/MedPointS/segmentation/*label
zip -0 -r /media/wlsdzyzl/DATA/datasets/pcd/MedPointS.zip /media/wlsdzyzl/DATA/datasets/pcd/MedPointS/ 
#!/bin/bash
python -u generate_seg_dataset.py
python ../random_split_k_fold.py -p /data/guoqingzhang/datasets/MedPointS/segmentation/ --sub_dirs coarse_label,colorized_pcd,fine_label,pcd --suffix .seg,.ply,.seg,.ply -o /data/guoqingzhang/datasets/MedPointS/segmentation/

rm -rf /data/guoqingzhang/datasets/MedPointS/segmentation/*pcd
rm -rf /data/guoqingzhang/datasets/MedPointS/segmentation/*label
zip -0 -r /data/guoqingzhang/datasets/MedPointS.zip /data/guoqingzhang/datasets/MedPointS/ 
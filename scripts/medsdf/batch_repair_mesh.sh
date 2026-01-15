#!/bin/bash
python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/
python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/
python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/
python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/
python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/
python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/
python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/ours/
python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/ --smoothing
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/ -o /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/ --smoothing -r
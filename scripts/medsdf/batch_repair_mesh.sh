#!/bin/bash
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/ours/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/ --smoothing
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/ --smoothing -r
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_512/ --smoothing -r
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh -r
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh -r
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh -r
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/CoW_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC_LA -o /data/guoqingzhang/vcg-results/CoW_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC_LA_repaired --smoothing --ncc 1
python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/CoW_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC -o /data/guoqingzhang/vcg-results/CoW_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC_repaired --ncc 1
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/CoW_vessel_diff/gen/VessDiff_Mesh -o /data/guoqingzhang/vcg-results/CoW_vessel_diff/gen/VessDiff_Mesh_repaired --ncc 1

# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC_LA -o /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC_LA_repaired --smoothing --ncc 2
python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC -o /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/LDM_EDM_SkCNN_with_SKC_repaired --ncc 2
# python ../repair_mesh.py -i /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/VessDiff_Mesh -o /data/guoqingzhang/vcg-results/imageCAS_vessel_diff/gen/VessDiff_Mesh_repaired --ncc 2
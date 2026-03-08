#!/bin/bash
python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_bladder_1770311526/out.npy -o ~/diffusion-point-cloud/results/gen_ply/bladder

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_brain_1770311505/out.npy -o ~/diffusion-point-cloud/results/gen_ply/brain

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_colon_1770311547/out.npy -o ~/diffusion-point-cloud/results/gen_ply/colon

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_coronary_artery_left_d_1770311569/out.npy -o ~/diffusion-point-cloud/results/gen_ply/coronary_artery_left_d

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_coronary_artery_right_d_1770311590/out.npy -o ~/diffusion-point-cloud/results/gen_ply/coronary_artery_right_d

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_duodenum_1770311611/out.npy -o ~/diffusion-point-cloud/results/gen_ply/duodenum

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_gallbladder_1763372792/out.npy -o ~/diffusion-point-cloud/results/gen_ply/gallbladder

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_liver_1770311653/out.npy -o ~/diffusion-point-cloud/results/gen_ply/liver

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_pancreas_1770311675/out.npy -o ~/diffusion-point-cloud/results/gen_ply/pancreas

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_skull_1770311696/out.npy -o ~/diffusion-point-cloud/results/gen_ply/skull

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_spleen_1770311718/out.npy -o ~/diffusion-point-cloud/results/gen_ply/spleen

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_stomach_1770311739/out.npy -o ~/diffusion-point-cloud/results/gen_ply/stomach

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_trachea_1770311760/out.npy -o ~/diffusion-point-cloud/results/gen_ply/trachea

python npy2ply_gen.py --result_file ~/diffusion-point-cloud/results/GEN_Ours_uterus_1770311782/out.npy -o ~/diffusion-point-cloud/results/gen_ply/uterus

# python npy2ply_recon.py -p /media/wlsdzyzl/DATA/datasets/pcd/MedSDF -o /home/wlsdzyzl/project/diffusion-point-cloud/results/recon_ply --test_sub_dirs fold5 --result_file /home/wlsdzyzl/project/diffusion-point-cloud/results/AE_Ours_bladder_brain_colon_coronary_artery_left_d_coronary_artery_right_d_duodenum_gallbladder_liver_pancreas_skull_spleen_stomach_trachea_uterus_1763963244/out.npy --pcd_dir raw


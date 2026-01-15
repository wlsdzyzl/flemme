#!/bin/bash
#SBATCH --job-name=render
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

echo bladder
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/bladder/097446_bladder.ply -o bladder_ref.png --size 1600,1200 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_207.ply -o bladder_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_179.ply -o bladder_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_447.ply -o bladder_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_253.ply -o bladder_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_3.ply -o bladder_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_300.ply -o bladder_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_57.ply -o bladder_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_489.ply -o bladder_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_119.ply -o bladder_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_471.ply -o bladder_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_57.ply -o bladder_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_300.ply -o bladder_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_489.ply -o bladder_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_140.ply -o bladder_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_471.ply -o bladder_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-87.ply -o bladder_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-299.ply -o bladder_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-358.ply -o bladder_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-485.ply -o bladder_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-188.ply -o bladder_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/bladder/gen_rand_0_192.ply -o bladder_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/bladder/gen_rand_0_207.ply -o bladder_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/bladder/gen_rand_0_162.ply -o bladder_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/bladder/gen_rand_0_415.ply -o bladder_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/bladder/gen_rand_0_51.ply -o bladder_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

echo brain
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/brain/006919_brain.ply -o brain_ref.png --size 1600,1200 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_43.ply -o brain_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_234.ply -o brain_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_118.ply -o brain_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_245.ply -o brain_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_290.ply -o brain_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_437.ply -o brain_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_298.ply -o brain_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_416.ply -o brain_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_49.ply -o brain_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_494.ply -o brain_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_11.ply -o brain_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_99.ply -o brain_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_273.ply -o brain_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_285.ply -o brain_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_492.ply -o brain_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-469.ply -o brain_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-425.ply -o brain_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-315.ply -o brain_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-475.ply -o brain_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-22.ply -o brain_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/brain/gen_rand_0_400.ply -o brain_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/brain/gen_rand_0_66.ply -o brain_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/brain/gen_rand_0_385.ply -o brain_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/brain/gen_rand_0_69.ply -o brain_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/brain/gen_rand_0_150.ply -o brain_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

echo colon
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/colon/s0074_colon.ply -o colon_ref.png --size 1600,1200--xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_350.ply -o colon_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_400.ply -o colon_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_144.ply -o colon_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_281.ply -o colon_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_245.ply -o colon_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_237.ply -o colon_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_316.ply -o colon_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_246.ply -o colon_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_51.ply -o colon_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_98.ply -o colon_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_179.ply -o colon_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_51.ply -o colon_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_290.ply -o colon_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_72.ply -o colon_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_180.ply -o colon_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-177.ply -o colon_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-434.ply -o colon_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-92.ply -o colon_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-331.ply -o colon_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-490.ply -o colon_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/colon/gen_rand_0_232.ply -o colon_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/colon/gen_rand_0_492.ply -o colon_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/colon/gen_rand_0_402.ply -o colon_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/colon/gen_rand_0_371.ply -o colon_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/colon/gen_rand_0_271.ply -o colon_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

echo coronary_artery_left_d
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d/10814698_0.ply -o coronary_artery_left_d_ref.png --size 1600,1200 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_73.ply -o coronary_artery_left_d_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_185.ply -o coronary_artery_left_d_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_96.ply -o coronary_artery_left_d_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_250.ply -o coronary_artery_left_d_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_490.ply -o coronary_artery_left_d_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_87.ply -o coronary_artery_left_d_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_360.ply -o coronary_artery_left_d_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_52.ply -o coronary_artery_left_d_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_294.ply -o coronary_artery_left_d_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_141.ply -o coronary_artery_left_d_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_33.ply -o coronary_artery_left_d_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_337.ply -o coronary_artery_left_d_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_182.ply -o coronary_artery_left_d_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_274.ply -o coronary_artery_left_d_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_164.ply -o coronary_artery_left_d_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-262.ply -o coronary_artery_left_d_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-266.ply -o coronary_artery_left_d_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-432.ply -o coronary_artery_left_d_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-186.ply -o coronary_artery_left_d_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-256.ply -o coronary_artery_left_d_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_left_d/gen_rand_0_194.ply -o coronary_artery_left_d_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_left_d/gen_rand_0_140.ply -o coronary_artery_left_d_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_left_d/gen_rand_0_461.ply -o coronary_artery_left_d_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_left_d/gen_rand_0_329.ply -o coronary_artery_left_d_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_left_d/gen_rand_0_88.ply -o coronary_artery_left_d_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

echo coronary_artery_right_d
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d/11019652_1.ply -o coronary_artery_right_d_ref.png --size 1600,1200 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_291.ply -o coronary_artery_right_d_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_25.ply -o coronary_artery_right_d_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_495.ply -o coronary_artery_right_d_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_115.ply -o coronary_artery_right_d_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_198.ply -o coronary_artery_right_d_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_211.ply -o coronary_artery_right_d_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_481.ply -o coronary_artery_right_d_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_28.ply -o coronary_artery_right_d_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_220.ply -o coronary_artery_right_d_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_379.ply -o coronary_artery_right_d_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_417.ply -o coronary_artery_right_d_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_52.ply -o coronary_artery_right_d_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_75.ply -o coronary_artery_right_d_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_317.ply -o coronary_artery_right_d_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_38.ply -o coronary_artery_right_d_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-336.ply -o coronary_artery_right_d_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-72.ply -o coronary_artery_right_d_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-166.ply -o coronary_artery_right_d_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-209.ply -o coronary_artery_right_d_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-35.ply -o coronary_artery_right_d_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_right_d/gen_rand_0_289.ply -o coronary_artery_right_d_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_right_d/gen_rand_0_115.ply -o coronary_artery_right_d_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_right_d/gen_rand_0_130.ply -o coronary_artery_right_d_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_right_d/gen_rand_0_272.ply -o coronary_artery_right_d_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/coronary_artery_right_d/gen_rand_0_67.ply -o coronary_artery_right_d_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

echo duodenum
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/duodenum/045963_duodenum.ply -o duodenum_ref.png --size 1600,1200 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_447.ply -o duodenum_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_100.ply -o duodenum_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_391.ply -o duodenum_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_301.ply -o duodenum_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_496.ply -o duodenum_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_06.ply -o duodenum_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_450.ply -o duodenum_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_214.ply -o duodenum_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_384.ply -o duodenum_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_149.ply -o duodenum_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_193.ply -o duodenum_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_486.ply -o duodenum_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_174.ply -o duodenum_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_29.ply -o duodenum_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_319.ply -o duodenum_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-87.ply -o duodenum_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-381.ply -o duodenum_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-478.ply -o duodenum_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-54.ply -o duodenum_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-365.ply -o duodenum_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/duodenum/gen_rand_0_220.ply -o duodenum_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/duodenum/gen_rand_0_353.ply -o duodenum_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/duodenum/gen_rand_0_146.ply -o duodenum_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/duodenum/gen_rand_0_340.ply -o duodenum_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/duodenum/gen_rand_0_306.ply -o duodenum_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

echo gallbladder
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/gallbladder/008517_gallbladder.ply -o gallbladder_ref.png --size 1600,1200 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_350.ply -o gallbladder_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_402.ply -o gallbladder_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_205.ply -o gallbladder_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_132.ply -o gallbladder_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_87.ply -o gallbladder_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_248.ply -o gallbladder_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_124.ply -o gallbladder_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_29.ply -o gallbladder_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_333.ply -o gallbladder_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_384.ply -o gallbladder_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_170.ply -o gallbladder_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_386.ply -o gallbladder_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_59.ply -o gallbladder_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_497.ply -o gallbladder_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_114.ply -o gallbladder_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-428.ply -o gallbladder_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-46.ply -o gallbladder_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-365.ply -o gallbladder_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-229.ply -o gallbladder_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-425.ply -o gallbladder_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/gallbladder/gen_rand_0_488.ply -o gallbladder_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/gallbladder/gen_rand_0_410.ply -o gallbladder_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/gallbladder/gen_rand_0_208.ply -o gallbladder_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/gallbladder/gen_rand_0_211.ply -o gallbladder_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/gallbladder/gen_rand_0_42.ply -o gallbladder_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

echo liver
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/liver/047501_liver.ply -o liver_ref.png --size 1600,1200 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_225.ply -o liver_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_265.ply -o liver_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_484.ply -o liver_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_148.ply -o liver_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_313.ply -o liver_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_123.ply -o liver_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_255.ply -o liver_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_296.ply -o liver_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_338.ply -o liver_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_146.ply -o liver_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_196.ply -o liver_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_288.ply -o liver_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_461.ply -o liver_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_456.ply -o liver_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_39.ply -o liver_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-358.ply -o liver_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-485.ply -o liver_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-327.ply -o liver_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-292.ply -o liver_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-390.ply -o liver_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/liver/gen_rand_0_338.ply -o liver_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/liver/gen_rand_0_323.ply -o liver_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/liver/gen_rand_0_121.ply -o liver_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/liver/gen_rand_0_358.ply -o liver_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/liver/gen_rand_0_486.ply -o liver_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

echo pancreas
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/pancreas/s0213_pancreas.ply -o pancreas_ref.png --size 1600,1200--xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_237.ply -o pancreas_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_186.ply -o pancreas_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_54.ply -o pancreas_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_99.ply -o pancreas_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_390.ply -o pancreas_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_440.ply -o pancreas_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_202.ply -o pancreas_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_230.ply -o pancreas_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_267.ply -o pancreas_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_03.ply -o pancreas_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_46.ply -o pancreas_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_49.ply -o pancreas_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_278.ply -o pancreas_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_375.ply -o pancreas_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_172.ply -o pancreas_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-86.ply -o pancreas_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-91.ply -o pancreas_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-185.ply -o pancreas_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-437.ply -o pancreas_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-290.ply -o pancreas_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/pancreas/gen_rand_0_294.ply -o pancreas_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/pancreas/gen_rand_0_242.ply -o pancreas_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/pancreas/gen_rand_0_272.ply -o pancreas_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/pancreas/gen_rand_0_278.ply -o pancreas_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/pancreas/gen_rand_0_436.ply -o pancreas_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

echo spleen
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/spleen/s0429_spleen.ply -o spleen_ref.png --size 1600,1200 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_270.ply -o spleen_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_137.ply -o spleen_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_125.ply -o spleen_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_220.ply -o spleen_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_444.ply -o spleen_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_179.ply -o spleen_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_60.ply -o spleen_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_98.ply -o spleen_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_324.ply -o spleen_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_334.ply -o spleen_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_185.ply -o spleen_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_65.ply -o spleen_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_372.ply -o spleen_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_429.ply -o spleen_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_377.ply -o spleen_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-87.ply -o spleen_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-179.ply -o spleen_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-20.ply -o spleen_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-402.ply -o spleen_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-23.ply -o spleen_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/spleen/gen_rand_0_183.ply -o spleen_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/spleen/gen_rand_0_485.ply -o spleen_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/spleen/gen_rand_0_123.ply -o spleen_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/spleen/gen_rand_0_03.ply -o spleen_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/spleen/gen_rand_0_61.ply -o spleen_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

echo stomach
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/stomach/s0578_stomach.ply -o stomach_ref.png --size 1600,1200 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_31.ply -o stomach_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_158.ply -o stomach_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_212.ply -o stomach_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_433.ply -o stomach_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_223.ply -o stomach_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_77.ply -o stomach_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_48.ply -o stomach_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_183.ply -o stomach_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_171.ply -o stomach_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_91.ply -o stomach_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_194.ply -o stomach_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_28.ply -o stomach_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_08.ply -o stomach_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_372.ply -o stomach_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_419.ply -o stomach_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-455.ply -o stomach_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-413.ply -o stomach_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-110.ply -o stomach_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-175.ply -o stomach_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-486.ply -o stomach_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/stomach/gen_rand_0_289.ply -o stomach_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/stomach/gen_rand_0_115.ply -o stomach_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/stomach/gen_rand_0_130.ply -o stomach_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/stomach/gen_rand_0_272.ply -o stomach_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/stomach/gen_rand_0_67.ply -o stomach_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

echo trachea
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/trachea/051928_trachea.ply -o trachea_ref.png --size 1600,1200 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_238.ply -o trachea_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_113.ply -o trachea_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_356.ply -o trachea_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_381.ply -o trachea_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_126.ply -o trachea_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_464.ply -o trachea_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_175.ply -o trachea_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_144.ply -o trachea_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_134.ply -o trachea_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_358.ply -o trachea_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_343.ply -o trachea_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_487.ply -o trachea_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_40.ply -o trachea_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_362.ply -o trachea_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_115.ply -o trachea_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-304.ply -o trachea_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-123.ply -o trachea_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-94.ply -o trachea_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-494.ply -o trachea_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-247.ply -o trachea_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/trachea/gen_rand_0_356.ply -o trachea_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/trachea/gen_rand_0_303.ply -o trachea_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/trachea/gen_rand_0_76.ply -o trachea_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/trachea/gen_rand_0_233.ply -o trachea_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/trachea/gen_rand_0_427.ply -o trachea_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

echo uterus
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/gen/uterus/075415_uterus.ply -o uterus_ref.png --size 1600,1200 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_183.ply -o uterus_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_319.ply -o uterus_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_171.ply -o uterus_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_178.ply -o uterus_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_208.ply -o uterus_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_362.ply -o uterus_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_364.ply -o uterus_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_150.ply -o uterus_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_118.ply -o uterus_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_29.ply -o uterus_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_365.ply -o uterus_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_37.ply -o uterus_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_237.ply -o uterus_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_358.ply -o uterus_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_343.ply -o uterus_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-114.ply -o uterus_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-395.ply -o uterus_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-376.ply -o uterus_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-284.ply -o uterus_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-250.ply -o uterus_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/uterus/gen_rand_0_177.ply -o uterus_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/uterus/gen_rand_0_175.ply -o uterus_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/uterus/gen_rand_0_72.ply -o uterus_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/uterus/gen_rand_0_37.ply -o uterus_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la_repaired/uterus/gen_rand_0_15.ply -o uterus_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
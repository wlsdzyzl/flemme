#!/bin/bash
#SBATCH --job-name=render
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

# echo bladder
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/bladder/097446_bladder.ply -o ./tmp_img/gen/bladder_ref.png --size 1600,1200 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_207.ply -o ./tmp_img/gen/bladder_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_179.ply -o ./tmp_img/gen/bladder_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_447.ply -o ./tmp_img/gen/bladder_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_253.ply -o ./tmp_img/gen/bladder_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/bladder/gen_rand_3.ply -o ./tmp_img/gen/bladder_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_300.ply -o ./tmp_img/gen/bladder_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_57.ply -o ./tmp_img/gen/bladder_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_489.ply -o ./tmp_img/gen/bladder_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_119.ply -o ./tmp_img/gen/bladder_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/bladder/gen_rand_471.ply -o ./tmp_img/gen/bladder_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_57.ply -o ./tmp_img/gen/bladder_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_300.ply -o ./tmp_img/gen/bladder_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_489.ply -o ./tmp_img/gen/bladder_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_140.ply -o ./tmp_img/gen/bladder_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/bladder/gen_rand_471.ply -o ./tmp_img/gen/bladder_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-87.ply -o ./tmp_img/gen/bladder_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-299.ply -o ./tmp_img/gen/bladder_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-358.ply -o ./tmp_img/gen/bladder_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-485.ply -o ./tmp_img/gen/bladder_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/bladder/00000001-188.ply -o ./tmp_img/gen/bladder_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/bladder/gen_rand_0_283.ply -o ./tmp_img/gen/bladder_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/bladder/gen_rand_0_09.ply -o ./tmp_img/gen/bladder_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/bladder/gen_rand_0_273.ply -o ./tmp_img/gen/bladder_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/bladder/gen_rand_0_305.ply -o ./tmp_img/gen/bladder_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/bladder/gen_rand_0_407.ply -o ./tmp_img/gen/bladder_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/210 --float_height 0.0

# echo brain
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/brain/006919_brain.ply -o ./tmp_img/gen/brain_ref.png --size 1600,1200 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_43.ply -o ./tmp_img/gen/brain_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_234.ply -o ./tmp_img/gen/brain_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_118.ply -o ./tmp_img/gen/brain_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_245.ply -o ./tmp_img/gen/brain_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/brain/gen_rand_290.ply -o ./tmp_img/gen/brain_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_437.ply -o ./tmp_img/gen/brain_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_298.ply -o ./tmp_img/gen/brain_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_416.ply -o ./tmp_img/gen/brain_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_49.ply -o ./tmp_img/gen/brain_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/brain/gen_rand_494.ply -o ./tmp_img/gen/brain_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_11.ply -o ./tmp_img/gen/brain_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_99.ply -o ./tmp_img/gen/brain_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_273.ply -o ./tmp_img/gen/brain_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_285.ply -o ./tmp_img/gen/brain_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/brain/gen_rand_492.ply -o ./tmp_img/gen/brain_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-469.ply -o ./tmp_img/gen/brain_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-425.ply -o ./tmp_img/gen/brain_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-315.ply -o ./tmp_img/gen/brain_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-475.ply -o ./tmp_img/gen/brain_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/brain/00000002-22.ply -o ./tmp_img/gen/brain_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/brain/gen_rand_0_334.ply -o ./tmp_img/gen/brain_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/brain/gen_rand_0_73.ply -o ./tmp_img/gen/brain_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/brain/gen_rand_0_124.ply -o ./tmp_img/gen/brain_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/brain/gen_rand_0_352.ply -o ./tmp_img/gen/brain_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/brain/gen_rand_0_27.ply -o ./tmp_img/gen/brain_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,0,-90

# echo colon
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/colon/s0074_colon.ply -o ./tmp_img/gen/colon_ref.png --size 1600,1200 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_350.ply -o ./tmp_img/gen/colon_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_400.ply -o ./tmp_img/gen/colon_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_144.ply -o ./tmp_img/gen/colon_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_281.ply -o ./tmp_img/gen/colon_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_245.ply -o ./tmp_img/gen/colon_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_237.ply -o ./tmp_img/gen/colon_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_316.ply -o ./tmp_img/gen/colon_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_246.ply -o ./tmp_img/gen/colon_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_51.ply -o ./tmp_img/gen/colon_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_98.ply -o ./tmp_img/gen/colon_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_179.ply -o ./tmp_img/gen/colon_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_51.ply -o ./tmp_img/gen/colon_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_290.ply -o ./tmp_img/gen/colon_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_72.ply -o ./tmp_img/gen/colon_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_180.ply -o ./tmp_img/gen/colon_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-177.ply -o ./tmp_img/gen/colon_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-434.ply -o ./tmp_img/gen/colon_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-92.ply -o ./tmp_img/gen/colon_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-331.ply -o ./tmp_img/gen/colon_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-490.ply -o ./tmp_img/gen/colon_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_193.ply -o ./tmp_img/gen/colon_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_17.ply -o ./tmp_img/gen/colon_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_257.ply -o ./tmp_img/gen/colon_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_427.ply -o ./tmp_img/gen/colon_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_333.ply -o ./tmp_img/gen/colon_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-60,x/-40 --float_height=0.0

# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/colon/069186_colon.ply -o ./tmp_img/gen/colon_1_ref.png --size 1600,1200 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_215.ply -o ./tmp_img/gen/colon_1_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_168.ply -o ./tmp_img/gen/colon_1_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_468.ply -o ./tmp_img/gen/colon_1_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_319.ply -o ./tmp_img/gen/colon_1_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_43.ply -o ./tmp_img/gen/colon_1_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_335.ply -o ./tmp_img/gen/colon_1_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_374.ply -o ./tmp_img/gen/colon_1_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_174.ply -o ./tmp_img/gen/colon_1_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_432.ply -o ./tmp_img/gen/colon_1_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_421.ply -o ./tmp_img/gen/colon_1_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_107.ply -o ./tmp_img/gen/colon_1_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_34.ply -o ./tmp_img/gen/colon_1_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_174.ply -o ./tmp_img/gen/colon_1_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_228.ply -o ./tmp_img/gen/colon_1_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_423.ply -o ./tmp_img/gen/colon_1_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-107.ply -o ./tmp_img/gen/colon_1_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-42.ply -o ./tmp_img/gen/colon_1_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-334.ply -o ./tmp_img/gen/colon_1_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-134.ply -o ./tmp_img/gen/colon_1_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-114.ply -o ./tmp_img/gen/colon_1_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_479.ply -o ./tmp_img/gen/colon_1_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_212.ply -o ./tmp_img/gen/colon_1_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_125.ply -o ./tmp_img/gen/colon_1_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_115.ply -o ./tmp_img/gen/colon_1_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_189.ply -o ./tmp_img/gen/colon_1_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles y/-30,x/-20 --float_height 0.05

# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/colon/s0370_colon.ply -o ./tmp_img/gen/colon_2_ref.png --size 1600,1200 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_198.ply -o ./tmp_img/gen/colon_2_diffpcd_0.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_173.ply -o ./tmp_img/gen/colon_2_diffpcd_1.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_36.ply -o ./tmp_img/gen/colon_2_diffpcd_2.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_375.ply -o ./tmp_img/gen/colon_2_diffpcd_3.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/colon/gen_rand_224.ply -o ./tmp_img/gen/colon_2_diffpcd_4.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_494.ply -o ./tmp_img/gen/colon_2_edm_0.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_113.ply -o ./tmp_img/gen/colon_2_edm_1.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_474.ply -o ./tmp_img/gen/colon_2_edm_2.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_189.ply -o ./tmp_img/gen/colon_2_edm_3.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/colon/gen_rand_263.ply -o ./tmp_img/gen/colon_2_edm_4.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_494.ply -o ./tmp_img/gen/colon_2_pvd_0.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_263.ply -o ./tmp_img/gen/colon_2_pvd_1.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_364.ply -o ./tmp_img/gen/colon_2_pvd_2.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_113.ply -o ./tmp_img/gen/colon_2_pvd_3.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/colon/gen_rand_08.ply -o ./tmp_img/gen/colon_2_pvd_4.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-420.ply -o ./tmp_img/gen/colon_2_gem3d_0.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-66.ply -o ./tmp_img/gen/colon_2_gem3d_1.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-406.ply -o ./tmp_img/gen/colon_2_gem3d_2.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-237.ply -o ./tmp_img/gen/colon_2_gem3d_3.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/colon/00000003-299.ply -o ./tmp_img/gen/colon_2_gem3d_4.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_42.ply -o ./tmp_img/gen/colon_2_ours_0.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_62.ply -o ./tmp_img/gen/colon_2_ours_1.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_252.ply -o ./tmp_img/gen/colon_2_ours_2.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_307.ply -o ./tmp_img/gen/colon_2_ours_3.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/colon/gen_rand_0_247.ply -o ./tmp_img/gen/colon_2_ours_4.png --size 1600,1200 --color_id 41 --float_height=0.0 --xyz_angles y/-90


# echo coronary_artery_left_d
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d/12020033_0.ply -o ./tmp_img/gen/coronary_artery_left_d_ref.png --size 1600,1200 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_11.ply -o ./tmp_img/gen/coronary_artery_left_d_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_271.ply -o ./tmp_img/gen/coronary_artery_left_d_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_214.ply -o ./tmp_img/gen/coronary_artery_left_d_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_140.ply -o ./tmp_img/gen/coronary_artery_left_d_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_65.ply -o ./tmp_img/gen/coronary_artery_left_d_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_322.ply -o ./tmp_img/gen/coronary_artery_left_d_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_487.ply -o ./tmp_img/gen/coronary_artery_left_d_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_70.ply -o ./tmp_img/gen/coronary_artery_left_d_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_239.ply -o ./tmp_img/gen/coronary_artery_left_d_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_113.ply -o ./tmp_img/gen/coronary_artery_left_d_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_286.ply -o ./tmp_img/gen/coronary_artery_left_d_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_258.ply -o ./tmp_img/gen/coronary_artery_left_d_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_450.ply -o ./tmp_img/gen/coronary_artery_left_d_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_155.ply -o ./tmp_img/gen/coronary_artery_left_d_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_471.ply -o ./tmp_img/gen/coronary_artery_left_d_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-63.ply -o ./tmp_img/gen/coronary_artery_left_d_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-87.ply -o ./tmp_img/gen/coronary_artery_left_d_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-201.ply -o ./tmp_img/gen/coronary_artery_left_d_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-488.ply -o ./tmp_img/gen/coronary_artery_left_d_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-113.ply -o ./tmp_img/gen/coronary_artery_left_d_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_224.ply -o ./tmp_img/gen/coronary_artery_left_d_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_480.ply -o ./tmp_img/gen/coronary_artery_left_d_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_186.ply -o ./tmp_img/gen/coronary_artery_left_d_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_141.ply -o ./tmp_img/gen/coronary_artery_left_d_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_45.ply -o ./tmp_img/gen/coronary_artery_left_d_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-60,z/180

# echo coronary_artery_left_d
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d/12074624_0.ply -o ./tmp_img/gen/coronary_artery_left_d_1_ref.png --size 1600,1200 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_271.ply -o ./tmp_img/gen/coronary_artery_left_d_1_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_439.ply -o ./tmp_img/gen/coronary_artery_left_d_1_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_68.ply -o ./tmp_img/gen/coronary_artery_left_d_1_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_65.ply -o ./tmp_img/gen/coronary_artery_left_d_1_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_214.ply -o ./tmp_img/gen/coronary_artery_left_d_1_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_487.ply -o ./tmp_img/gen/coronary_artery_left_d_1_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_201.ply -o ./tmp_img/gen/coronary_artery_left_d_1_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_146.ply -o ./tmp_img/gen/coronary_artery_left_d_1_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_366.ply -o ./tmp_img/gen/coronary_artery_left_d_1_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_197.ply -o ./tmp_img/gen/coronary_artery_left_d_1_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_303.ply -o ./tmp_img/gen/coronary_artery_left_d_1_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_258.ply -o ./tmp_img/gen/coronary_artery_left_d_1_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_235.ply -o ./tmp_img/gen/coronary_artery_left_d_1_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_334.ply -o ./tmp_img/gen/coronary_artery_left_d_1_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_286.ply -o ./tmp_img/gen/coronary_artery_left_d_1_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-173.ply -o ./tmp_img/gen/coronary_artery_left_d_1_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-442.ply -o ./tmp_img/gen/coronary_artery_left_d_1_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-291.ply -o ./tmp_img/gen/coronary_artery_left_d_1_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-50.ply -o ./tmp_img/gen/coronary_artery_left_d_1_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-197.ply -o ./tmp_img/gen/coronary_artery_left_d_1_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_211.ply -o ./tmp_img/gen/coronary_artery_left_d_1_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_452.ply -o ./tmp_img/gen/coronary_artery_left_d_1_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_224.ply -o ./tmp_img/gen/coronary_artery_left_d_1_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_480.ply -o ./tmp_img/gen/coronary_artery_left_d_1_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_304.ply -o ./tmp_img/gen/coronary_artery_left_d_1_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles 50,-90,0

# echo coronary_artery_left_d
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d/10814698_0.ply -o ./tmp_img/gen/coronary_artery_left_d_2_ref.png --size 1600,1200 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_73.ply -o ./tmp_img/gen/coronary_artery_left_d_2_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_185.ply -o ./tmp_img/gen/coronary_artery_left_d_2_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_96.ply -o ./tmp_img/gen/coronary_artery_left_d_2_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_250.ply -o ./tmp_img/gen/coronary_artery_left_d_2_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_490.ply -o ./tmp_img/gen/coronary_artery_left_d_2_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_87.ply -o ./tmp_img/gen/coronary_artery_left_d_2_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_360.ply -o ./tmp_img/gen/coronary_artery_left_d_2_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_52.ply -o ./tmp_img/gen/coronary_artery_left_d_2_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_294.ply -o ./tmp_img/gen/coronary_artery_left_d_2_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_141.ply -o ./tmp_img/gen/coronary_artery_left_d_2_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_33.ply -o ./tmp_img/gen/coronary_artery_left_d_2_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_337.ply -o ./tmp_img/gen/coronary_artery_left_d_2_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_182.ply -o ./tmp_img/gen/coronary_artery_left_d_2_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_274.ply -o ./tmp_img/gen/coronary_artery_left_d_2_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_164.ply -o ./tmp_img/gen/coronary_artery_left_d_2_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-262.ply -o ./tmp_img/gen/coronary_artery_left_d_2_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-266.ply -o ./tmp_img/gen/coronary_artery_left_d_2_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-432.ply -o ./tmp_img/gen/coronary_artery_left_d_2_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-186.ply -o ./tmp_img/gen/coronary_artery_left_d_2_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-256.ply -o ./tmp_img/gen/coronary_artery_left_d_2_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_85.ply -o ./tmp_img/gen/coronary_artery_left_d_2_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_121.ply -o ./tmp_img/gen/coronary_artery_left_d_2_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_76.ply -o ./tmp_img/gen/coronary_artery_left_d_2_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_91.ply -o ./tmp_img/gen/coronary_artery_left_d_2_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_188.ply -o ./tmp_img/gen/coronary_artery_left_d_2_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/90,z/-30,y/-30,x/-20,z/10

# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_left_d/12073828_0.ply -o ./tmp_img/gen/coronary_artery_left_d_3_ref.png --size 1600,1200 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_73.ply -o ./tmp_img/gen/coronary_artery_left_d_3_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_185.ply -o ./tmp_img/gen/coronary_artery_left_d_3_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_96.ply -o ./tmp_img/gen/coronary_artery_left_d_3_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_250.ply -o ./tmp_img/gen/coronary_artery_left_d_3_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_left_d/gen_rand_490.ply -o ./tmp_img/gen/coronary_artery_left_d_3_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_87.ply -o ./tmp_img/gen/coronary_artery_left_d_3_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_360.ply -o ./tmp_img/gen/coronary_artery_left_d_3_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_52.ply -o ./tmp_img/gen/coronary_artery_left_d_3_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_294.ply -o ./tmp_img/gen/coronary_artery_left_d_3_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_left_d/gen_rand_141.ply -o ./tmp_img/gen/coronary_artery_left_d_3_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_33.ply -o ./tmp_img/gen/coronary_artery_left_d_3_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_337.ply -o ./tmp_img/gen/coronary_artery_left_d_3_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_182.ply -o ./tmp_img/gen/coronary_artery_left_d_3_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_274.ply -o ./tmp_img/gen/coronary_artery_left_d_3_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_left_d/gen_rand_164.ply -o ./tmp_img/gen/coronary_artery_left_d_3_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-262.ply -o ./tmp_img/gen/coronary_artery_left_d_3_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-266.ply -o ./tmp_img/gen/coronary_artery_left_d_3_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-432.ply -o ./tmp_img/gen/coronary_artery_left_d_3_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-186.ply -o ./tmp_img/gen/coronary_artery_left_d_3_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_left_d/00000004-256.ply -o ./tmp_img/gen/coronary_artery_left_d_3_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_85.ply -o ./tmp_img/gen/coronary_artery_left_d_3_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_121.ply -o ./tmp_img/gen/coronary_artery_left_d_3_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_76.ply -o ./tmp_img/gen/coronary_artery_left_d_3_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_91.ply -o ./tmp_img/gen/coronary_artery_left_d_3_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_left_d/gen_rand_0_188.ply -o ./tmp_img/gen/coronary_artery_left_d_3_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/-60,x/10


# echo coronary_artery_right_d
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d/11019652_1.ply -o ./tmp_img/gen/coronary_artery_right_d_ref.png --size 1600,1200 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_291.ply -o ./tmp_img/gen/coronary_artery_right_d_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_25.ply -o ./tmp_img/gen/coronary_artery_right_d_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_495.ply -o ./tmp_img/gen/coronary_artery_right_d_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_115.ply -o ./tmp_img/gen/coronary_artery_right_d_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_198.ply -o ./tmp_img/gen/coronary_artery_right_d_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_211.ply -o ./tmp_img/gen/coronary_artery_right_d_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_481.ply -o ./tmp_img/gen/coronary_artery_right_d_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_28.ply -o ./tmp_img/gen/coronary_artery_right_d_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_220.ply -o ./tmp_img/gen/coronary_artery_right_d_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_379.ply -o ./tmp_img/gen/coronary_artery_right_d_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_417.ply -o ./tmp_img/gen/coronary_artery_right_d_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_52.ply -o ./tmp_img/gen/coronary_artery_right_d_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_75.ply -o ./tmp_img/gen/coronary_artery_right_d_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_317.ply -o ./tmp_img/gen/coronary_artery_right_d_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_38.ply -o ./tmp_img/gen/coronary_artery_right_d_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-336.ply -o ./tmp_img/gen/coronary_artery_right_d_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-72.ply -o ./tmp_img/gen/coronary_artery_right_d_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-166.ply -o ./tmp_img/gen/coronary_artery_right_d_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-209.ply -o ./tmp_img/gen/coronary_artery_right_d_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-35.ply -o ./tmp_img/gen/coronary_artery_right_d_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_390.ply -o ./tmp_img/gen/coronary_artery_right_d_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_37.ply -o ./tmp_img/gen/coronary_artery_right_d_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_205.ply -o ./tmp_img/gen/coronary_artery_right_d_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_131.ply -o ./tmp_img/gen/coronary_artery_right_d_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_474.ply -o ./tmp_img/gen/coronary_artery_right_d_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-40,x/180,z/40,y/-10 --float_height 0

# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d/12020120_1.ply -o ./tmp_img/gen/coronary_artery_right_d_1_ref.png --size 1600,1200 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_298.ply -o ./tmp_img/gen/coronary_artery_right_d_1_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_41.ply -o ./tmp_img/gen/coronary_artery_right_d_1_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_310.ply -o ./tmp_img/gen/coronary_artery_right_d_1_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_277.ply -o ./tmp_img/gen/coronary_artery_right_d_1_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_406.ply -o ./tmp_img/gen/coronary_artery_right_d_1_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_42.ply -o ./tmp_img/gen/coronary_artery_right_d_1_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_476.ply -o ./tmp_img/gen/coronary_artery_right_d_1_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_67.ply -o ./tmp_img/gen/coronary_artery_right_d_1_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_338.ply -o ./tmp_img/gen/coronary_artery_right_d_1_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_74.ply -o ./tmp_img/gen/coronary_artery_right_d_1_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_327.ply -o ./tmp_img/gen/coronary_artery_right_d_1_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_310.ply -o ./tmp_img/gen/coronary_artery_right_d_1_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_69.ply -o ./tmp_img/gen/coronary_artery_right_d_1_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_228.ply -o ./tmp_img/gen/coronary_artery_right_d_1_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_317.ply -o ./tmp_img/gen/coronary_artery_right_d_1_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-92.ply -o ./tmp_img/gen/coronary_artery_right_d_1_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-37.ply -o ./tmp_img/gen/coronary_artery_right_d_1_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-28.ply -o ./tmp_img/gen/coronary_artery_right_d_1_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-457.ply -o ./tmp_img/gen/coronary_artery_right_d_1_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-77.ply -o ./tmp_img/gen/coronary_artery_right_d_1_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_218.ply -o ./tmp_img/gen/coronary_artery_right_d_1_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_25.ply -o ./tmp_img/gen/coronary_artery_right_d_1_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_128.ply -o ./tmp_img/gen/coronary_artery_right_d_1_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_168.ply -o ./tmp_img/gen/coronary_artery_right_d_1_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_00.ply -o ./tmp_img/gen/coronary_artery_right_d_1_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/30,z/120,y/130,x/-10,z/20,y/10 --float_height 0

# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/coronary_artery_right_d/12072633_1.ply -o ./tmp_img/gen/coronary_artery_right_d_2_ref.png --size 1600,1200 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_69.ply -o ./tmp_img/gen/coronary_artery_right_d_2_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_298.ply -o ./tmp_img/gen/coronary_artery_right_d_2_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_12.ply -o ./tmp_img/gen/coronary_artery_right_d_2_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_74.ply -o ./tmp_img/gen/coronary_artery_right_d_2_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/coronary_artery_right_d/gen_rand_406.ply -o ./tmp_img/gen/coronary_artery_right_d_2_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_52.ply -o ./tmp_img/gen/coronary_artery_right_d_2_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_253.ply -o ./tmp_img/gen/coronary_artery_right_d_2_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_165.ply -o ./tmp_img/gen/coronary_artery_right_d_2_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_241.ply -o ./tmp_img/gen/coronary_artery_right_d_2_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/coronary_artery_right_d/gen_rand_437.ply -o ./tmp_img/gen/coronary_artery_right_d_2_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_347.ply -o ./tmp_img/gen/coronary_artery_right_d_2_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_51.ply -o ./tmp_img/gen/coronary_artery_right_d_2_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_188.ply -o ./tmp_img/gen/coronary_artery_right_d_2_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_75.ply -o ./tmp_img/gen/coronary_artery_right_d_2_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/coronary_artery_right_d/gen_rand_367.ply -o ./tmp_img/gen/coronary_artery_right_d_2_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-296.ply -o ./tmp_img/gen/coronary_artery_right_d_2_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-303.ply -o ./tmp_img/gen/coronary_artery_right_d_2_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-234.ply -o ./tmp_img/gen/coronary_artery_right_d_2_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-340.ply -o ./tmp_img/gen/coronary_artery_right_d_2_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/coronary_artery_right_d/00000005-366.ply -o ./tmp_img/gen/coronary_artery_right_d_2_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_156.ply -o ./tmp_img/gen/coronary_artery_right_d_2_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_89.ply -o ./tmp_img/gen/coronary_artery_right_d_2_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_271.ply -o ./tmp_img/gen/coronary_artery_right_d_2_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_405.ply -o ./tmp_img/gen/coronary_artery_right_d_2_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/coronary_artery_right_d/gen_rand_0_470.ply -o ./tmp_img/gen/coronary_artery_right_d_2_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-60,x/180,y/-10,z/20 --float_height 0.0

# echo duodenum
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/duodenum/045963_duodenum.ply -o ./tmp_img/gen/duodenum_ref.png --size 1600,1200 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_447.ply -o ./tmp_img/gen/duodenum_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_100.ply -o ./tmp_img/gen/duodenum_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_391.ply -o ./tmp_img/gen/duodenum_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_301.ply -o ./tmp_img/gen/duodenum_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/duodenum/gen_rand_496.ply -o ./tmp_img/gen/duodenum_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_06.ply -o ./tmp_img/gen/duodenum_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_450.ply -o ./tmp_img/gen/duodenum_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_214.ply -o ./tmp_img/gen/duodenum_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_384.ply -o ./tmp_img/gen/duodenum_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/duodenum/gen_rand_149.ply -o ./tmp_img/gen/duodenum_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_193.ply -o ./tmp_img/gen/duodenum_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_486.ply -o ./tmp_img/gen/duodenum_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_174.ply -o ./tmp_img/gen/duodenum_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_29.ply -o ./tmp_img/gen/duodenum_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/duodenum/gen_rand_319.ply -o ./tmp_img/gen/duodenum_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-87.ply -o ./tmp_img/gen/duodenum_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-381.ply -o ./tmp_img/gen/duodenum_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-478.ply -o ./tmp_img/gen/duodenum_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-54.ply -o ./tmp_img/gen/duodenum_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/duodenum/00000006-365.ply -o ./tmp_img/gen/duodenum_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/duodenum/gen_rand_0_301.ply -o ./tmp_img/gen/duodenum_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/duodenum/gen_rand_0_169.ply -o ./tmp_img/gen/duodenum_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/duodenum/gen_rand_0_220.ply -o ./tmp_img/gen/duodenum_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/duodenum/gen_rand_0_354.ply -o ./tmp_img/gen/duodenum_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/duodenum/gen_rand_0_46.ply -o ./tmp_img/gen/duodenum_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles 180,-20,0 --float_height 0

# echo gallbladder
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/gallbladder/008517_gallbladder.ply -o ./tmp_img/gen/gallbladder_ref.png --size 1600,1200 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_350.ply -o ./tmp_img/gen/gallbladder_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_402.ply -o ./tmp_img/gen/gallbladder_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_205.ply -o ./tmp_img/gen/gallbladder_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_132.ply -o ./tmp_img/gen/gallbladder_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_87.ply -o ./tmp_img/gen/gallbladder_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_248.ply -o ./tmp_img/gen/gallbladder_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_124.ply -o ./tmp_img/gen/gallbladder_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_29.ply -o ./tmp_img/gen/gallbladder_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_333.ply -o ./tmp_img/gen/gallbladder_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_384.ply -o ./tmp_img/gen/gallbladder_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_170.ply -o ./tmp_img/gen/gallbladder_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_386.ply -o ./tmp_img/gen/gallbladder_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_59.ply -o ./tmp_img/gen/gallbladder_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_497.ply -o ./tmp_img/gen/gallbladder_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_114.ply -o ./tmp_img/gen/gallbladder_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-428.ply -o ./tmp_img/gen/gallbladder_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-46.ply -o ./tmp_img/gen/gallbladder_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-365.ply -o ./tmp_img/gen/gallbladder_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-229.ply -o ./tmp_img/gen/gallbladder_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-425.ply -o ./tmp_img/gen/gallbladder_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_115.ply -o ./tmp_img/gen/gallbladder_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_330.ply -o ./tmp_img/gen/gallbladder_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_337.ply -o ./tmp_img/gen/gallbladder_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_238.ply -o ./tmp_img/gen/gallbladder_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_398.ply -o ./tmp_img/gen/gallbladder_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles 0,0,30

# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/gallbladder/006412_gallbladder.ply -o ./tmp_img/gen/gallbladder_1_ref.png --size 1600,1200 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_203.ply -o ./tmp_img/gen/gallbladder_1_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_151.ply -o ./tmp_img/gen/gallbladder_1_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_325.ply -o ./tmp_img/gen/gallbladder_1_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_463.ply -o ./tmp_img/gen/gallbladder_1_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/gallbladder/gen_rand_356.ply -o ./tmp_img/gen/gallbladder_1_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_315.ply -o ./tmp_img/gen/gallbladder_1_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_356.ply -o ./tmp_img/gen/gallbladder_1_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_492.ply -o ./tmp_img/gen/gallbladder_1_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_232.ply -o ./tmp_img/gen/gallbladder_1_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/gallbladder/gen_rand_51.ply -o ./tmp_img/gen/gallbladder_1_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_67.ply -o ./tmp_img/gen/gallbladder_1_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_235.ply -o ./tmp_img/gen/gallbladder_1_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_180.ply -o ./tmp_img/gen/gallbladder_1_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_433.ply -o ./tmp_img/gen/gallbladder_1_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/gallbladder/gen_rand_265.ply -o ./tmp_img/gen/gallbladder_1_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-331.ply -o ./tmp_img/gen/gallbladder_1_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-145.ply -o ./tmp_img/gen/gallbladder_1_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-295.ply -o ./tmp_img/gen/gallbladder_1_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-191.ply -o ./tmp_img/gen/gallbladder_1_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/gallbladder/00000007-16.ply -o ./tmp_img/gen/gallbladder_1_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_207.ply -o ./tmp_img/gen/gallbladder_1_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_438.ply -o ./tmp_img/gen/gallbladder_1_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_29.ply -o ./tmp_img/gen/gallbladder_1_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_258.ply -o ./tmp_img/gen/gallbladder_1_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/gallbladder/gen_rand_0_360.ply -o ./tmp_img/gen/gallbladder_1_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles y/30,x/20,z/-100,y/70 --float_height 0.0


# echo liver
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/liver/047501_liver.ply -o ./tmp_img/gen/liver_ref.png --size 1600,1200 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_225.ply -o ./tmp_img/gen/liver_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_265.ply -o ./tmp_img/gen/liver_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_484.ply -o ./tmp_img/gen/liver_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_148.ply -o ./tmp_img/gen/liver_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_313.ply -o ./tmp_img/gen/liver_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_123.ply -o ./tmp_img/gen/liver_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_255.ply -o ./tmp_img/gen/liver_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_296.ply -o ./tmp_img/gen/liver_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_338.ply -o ./tmp_img/gen/liver_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_146.ply -o ./tmp_img/gen/liver_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_196.ply -o ./tmp_img/gen/liver_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_288.ply -o ./tmp_img/gen/liver_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_461.ply -o ./tmp_img/gen/liver_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_456.ply -o ./tmp_img/gen/liver_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_39.ply -o ./tmp_img/gen/liver_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-358.ply -o ./tmp_img/gen/liver_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-485.ply -o ./tmp_img/gen/liver_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-327.ply -o ./tmp_img/gen/liver_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-292.ply -o ./tmp_img/gen/liver_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-390.ply -o ./tmp_img/gen/liver_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_31.ply -o ./tmp_img/gen/liver_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_149.ply -o ./tmp_img/gen/liver_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_335.ply -o ./tmp_img/gen/liver_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_235.ply -o ./tmp_img/gen/liver_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_249.ply -o ./tmp_img/gen/liver_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/60,y/180,z/20,y/20,z/10,y/10

# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/liver/s0342_liver.ply -o ./tmp_img/gen/liver_1_ref.png --size 1600,1200 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_165.ply -o ./tmp_img/gen/liver_1_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_153.ply -o ./tmp_img/gen/liver_1_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_32.ply -o ./tmp_img/gen/liver_1_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_163.ply -o ./tmp_img/gen/liver_1_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/liver/gen_rand_273.ply -o ./tmp_img/gen/liver_1_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_396.ply -o ./tmp_img/gen/liver_1_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_395.ply -o ./tmp_img/gen/liver_1_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_269.ply -o ./tmp_img/gen/liver_1_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_171.ply -o ./tmp_img/gen/liver_1_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/liver/gen_rand_129.ply -o ./tmp_img/gen/liver_1_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_27.ply -o ./tmp_img/gen/liver_1_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_472.ply -o ./tmp_img/gen/liver_1_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_119.ply -o ./tmp_img/gen/liver_1_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_386.ply -o ./tmp_img/gen/liver_1_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/liver/gen_rand_160.ply -o ./tmp_img/gen/liver_1_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-185.ply -o ./tmp_img/gen/liver_1_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-16.ply -o ./tmp_img/gen/liver_1_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-43.ply -o ./tmp_img/gen/liver_1_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-50.ply -o ./tmp_img/gen/liver_1_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/liver/00000008-414.ply -o ./tmp_img/gen/liver_1_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_126.ply -o ./tmp_img/gen/liver_1_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_463.ply -o ./tmp_img/gen/liver_1_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_477.ply -o ./tmp_img/gen/liver_1_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_383.ply -o ./tmp_img/gen/liver_1_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/liver/gen_rand_0_93.ply -o ./tmp_img/gen/liver_1_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles z/30,x/-30,z/10 --float_height 0.0


# echo pancreas
python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/pancreas/s0213_pancreas.ply -o ./tmp_img/gen/pancreas_ref.png --size 1600,1200 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_237.ply -o ./tmp_img/gen/pancreas_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_186.ply -o ./tmp_img/gen/pancreas_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_54.ply -o ./tmp_img/gen/pancreas_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_99.ply -o ./tmp_img/gen/pancreas_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/pancreas/gen_rand_390.ply -o ./tmp_img/gen/pancreas_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_440.ply -o ./tmp_img/gen/pancreas_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_202.ply -o ./tmp_img/gen/pancreas_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_230.ply -o ./tmp_img/gen/pancreas_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_267.ply -o ./tmp_img/gen/pancreas_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/pancreas/gen_rand_03.ply -o ./tmp_img/gen/pancreas_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_46.ply -o ./tmp_img/gen/pancreas_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_49.ply -o ./tmp_img/gen/pancreas_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_278.ply -o ./tmp_img/gen/pancreas_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_375.ply -o ./tmp_img/gen/pancreas_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/pancreas/gen_rand_172.ply -o ./tmp_img/gen/pancreas_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-86.ply -o ./tmp_img/gen/pancreas_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-91.ply -o ./tmp_img/gen/pancreas_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-185.ply -o ./tmp_img/gen/pancreas_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-437.ply -o ./tmp_img/gen/pancreas_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/pancreas/00000009-290.ply -o ./tmp_img/gen/pancreas_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/pancreas/gen_rand_0_430.ply -o ./tmp_img/gen/pancreas_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/pancreas/gen_rand_0_214.ply -o ./tmp_img/gen/pancreas_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/pancreas/gen_rand_0_80.ply -o ./tmp_img/gen/pancreas_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/pancreas/gen_rand_0_379.ply -o ./tmp_img/gen/pancreas_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/pancreas/gen_rand_0_152.ply -o ./tmp_img/gen/pancreas_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/-90,z/180,y/10 --float_height 0.0

# echo spleen
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/spleen/s0429_spleen.ply -o ./tmp_img/gen/spleen_ref.png --size 1600,1200 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_270.ply -o ./tmp_img/gen/spleen_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_137.ply -o ./tmp_img/gen/spleen_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_125.ply -o ./tmp_img/gen/spleen_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_220.ply -o ./tmp_img/gen/spleen_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/spleen/gen_rand_444.ply -o ./tmp_img/gen/spleen_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_179.ply -o ./tmp_img/gen/spleen_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_60.ply -o ./tmp_img/gen/spleen_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_98.ply -o ./tmp_img/gen/spleen_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_324.ply -o ./tmp_img/gen/spleen_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/spleen/gen_rand_334.ply -o ./tmp_img/gen/spleen_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_185.ply -o ./tmp_img/gen/spleen_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_65.ply -o ./tmp_img/gen/spleen_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_372.ply -o ./tmp_img/gen/spleen_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_429.ply -o ./tmp_img/gen/spleen_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/spleen/gen_rand_377.ply -o ./tmp_img/gen/spleen_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-87.ply -o ./tmp_img/gen/spleen_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-179.ply -o ./tmp_img/gen/spleen_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-20.ply -o ./tmp_img/gen/spleen_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-402.ply -o ./tmp_img/gen/spleen_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/spleen/00000011-23.ply -o ./tmp_img/gen/spleen_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/spleen/gen_rand_0_73.ply -o ./tmp_img/gen/spleen_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/spleen/gen_rand_0_10.ply -o ./tmp_img/gen/spleen_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/spleen/gen_rand_0_26.ply -o ./tmp_img/gen/spleen_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/spleen/gen_rand_0_460.ply -o ./tmp_img/gen/spleen_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/spleen/gen_rand_0_359.ply -o ./tmp_img/gen/spleen_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles z/-120,x/-20,y/-20

# echo stomach
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/stomach/s0578_stomach.ply -o ./tmp_img/gen/stomach_ref.png --size 1600,1200 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_31.ply -o ./tmp_img/gen/stomach_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_158.ply -o ./tmp_img/gen/stomach_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_212.ply -o ./tmp_img/gen/stomach_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_433.ply -o ./tmp_img/gen/stomach_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/stomach/gen_rand_223.ply -o ./tmp_img/gen/stomach_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_77.ply -o ./tmp_img/gen/stomach_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_48.ply -o ./tmp_img/gen/stomach_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_183.ply -o ./tmp_img/gen/stomach_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_171.ply -o ./tmp_img/gen/stomach_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/stomach/gen_rand_91.ply -o ./tmp_img/gen/stomach_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_194.ply -o ./tmp_img/gen/stomach_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_28.ply -o ./tmp_img/gen/stomach_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_08.ply -o ./tmp_img/gen/stomach_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_372.ply -o ./tmp_img/gen/stomach_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/stomach/gen_rand_419.ply -o ./tmp_img/gen/stomach_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-455.ply -o ./tmp_img/gen/stomach_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-413.ply -o ./tmp_img/gen/stomach_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-110.ply -o ./tmp_img/gen/stomach_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-175.ply -o ./tmp_img/gen/stomach_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/stomach/00000012-486.ply -o ./tmp_img/gen/stomach_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/stomach/gen_rand_0_474.ply -o ./tmp_img/gen/stomach_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/stomach/gen_rand_0_228.ply -o ./tmp_img/gen/stomach_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/stomach/gen_rand_0_427.ply -o ./tmp_img/gen/stomach_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/stomach/gen_rand_0_328.ply -o ./tmp_img/gen/stomach_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/stomach/gen_rand_0_205.ply -o ./tmp_img/gen/stomach_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/100,z/180,x/-60,y/50,z/-30

# echo trachea
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/trachea/051928_trachea.ply -o ./tmp_img/gen/trachea_ref.png --size 1600,1200 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_238.ply -o ./tmp_img/gen/trachea_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_113.ply -o ./tmp_img/gen/trachea_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_356.ply -o ./tmp_img/gen/trachea_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_381.ply -o ./tmp_img/gen/trachea_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/trachea/gen_rand_126.ply -o ./tmp_img/gen/trachea_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_464.ply -o ./tmp_img/gen/trachea_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_175.ply -o ./tmp_img/gen/trachea_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_144.ply -o ./tmp_img/gen/trachea_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_134.ply -o ./tmp_img/gen/trachea_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/trachea/gen_rand_358.ply -o ./tmp_img/gen/trachea_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_343.ply -o ./tmp_img/gen/trachea_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_487.ply -o ./tmp_img/gen/trachea_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_40.ply -o ./tmp_img/gen/trachea_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_362.ply -o ./tmp_img/gen/trachea_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/trachea/gen_rand_115.ply -o ./tmp_img/gen/trachea_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-304.ply -o ./tmp_img/gen/trachea_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-123.ply -o ./tmp_img/gen/trachea_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-94.ply -o ./tmp_img/gen/trachea_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-494.ply -o ./tmp_img/gen/trachea_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/trachea/00000013-247.ply -o ./tmp_img/gen/trachea_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/trachea/gen_rand_0_220.ply -o ./tmp_img/gen/trachea_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/trachea/gen_rand_0_373.ply -o ./tmp_img/gen/trachea_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/trachea/gen_rand_0_64.ply -o ./tmp_img/gen/trachea_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/trachea/gen_rand_0_307.ply -o ./tmp_img/gen/trachea_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles x/160
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/trachea/gen_rand_0_39.ply -o ./tmp_img/gen/trachea_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles x/160

# echo uterus
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/gen/uterus/075415_uterus.ply -o ./tmp_img/gen/uterus_ref.png --size 1600,1200 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_183.ply -o ./tmp_img/gen/uterus_diffpcd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_319.ply -o ./tmp_img/gen/uterus_diffpcd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_171.ply -o ./tmp_img/gen/uterus_diffpcd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_178.ply -o ./tmp_img/gen/uterus_diffpcd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/DiffPCD_Mesh/uterus/gen_rand_208.ply -o ./tmp_img/gen/uterus_diffpcd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_362.ply -o ./tmp_img/gen/uterus_edm_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_364.ply -o ./tmp_img/gen/uterus_edm_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_150.ply -o ./tmp_img/gen/uterus_edm_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_118.ply -o ./tmp_img/gen/uterus_edm_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/EDM_NONE_Mesh/uterus/gen_rand_29.ply -o ./tmp_img/gen/uterus_edm_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_365.ply -o ./tmp_img/gen/uterus_pvd_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_37.ply -o ./tmp_img/gen/uterus_pvd_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_237.ply -o ./tmp_img/gen/uterus_pvd_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_358.ply -o ./tmp_img/gen/uterus_pvd_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/PVD_Mesh/uterus/gen_rand_343.ply -o ./tmp_img/gen/uterus_pvd_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-114.ply -o ./tmp_img/gen/uterus_gem3d_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-395.ply -o ./tmp_img/gen/uterus_gem3d_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-376.ply -o ./tmp_img/gen/uterus_gem3d_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-284.ply -o ./tmp_img/gen/uterus_gem3d_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/GeM3D_origin/uterus/00000014-250.ply -o ./tmp_img/gen/uterus_gem3d_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20

# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/uterus/gen_rand_0_135.ply -o ./tmp_img/gen/uterus_ours_0.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/uterus/gen_rand_0_253.ply -o ./tmp_img/gen/uterus_ours_1.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/uterus/gen_rand_0_12.ply -o ./tmp_img/gen/uterus_ours_2.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/uterus/gen_rand_0_13.ply -o ./tmp_img/gen/uterus_ours_3.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
# python ../render_mesh.py -i /data/guoqingzhang/vcg-results/MedSDF/gen/LDM_EDM_SkCNN_with_la/uterus/gen_rand_0_29.ply -o ./tmp_img/gen/uterus_ours_4.png --size 1600,1200 --color_id 41 --xyz_angles y/90,z/-120,y/90,x/-20
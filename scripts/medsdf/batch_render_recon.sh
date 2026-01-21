#!/bin/bash
#SBATCH --job-name=render
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

# echo bladder
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/020636_bladder.ply -o ./tmp_img/recon/bladder_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/020636_bladder.ply -o ./tmp_img/recon/bladder_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/020636_bladder.ply -o ./tmp_img/recon/bladder_pct2.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/020636_bladder.ply -o ./tmp_img/recon/bladder_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/020636_bladder.ply -o ./tmp_img/recon/bladder_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/020636_bladder.ply -o ./tmp_img/recon/bladder_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/020636_bladder.ply -o ./tmp_img/recon/bladder_ours.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/020636_bladder.ply -o ./tmp_img/recon/bladder_gt.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/bladder/020636_bladder.ply -o ./tmp_img/recon/bladder_input.png --size 1600,1200 --xyz_angles -90,110,-10
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/bladder/020636_bladder.ply -o ./tmp_img/recon/bladder_skeleton.png --size 1600,1200 --xyz_angles -90,110,-10

# echo bladder
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/001313_bladder.ply -o ./tmp_img/recon/bladder_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/001313_bladder.ply -o ./tmp_img/recon/bladder_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/001313_bladder.ply -o ./tmp_img/recon/bladder_pct2.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/001313_bladder.ply -o ./tmp_img/recon/bladder_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/001313_bladder.ply -o ./tmp_img/recon/bladder_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/001313_bladder.ply -o ./tmp_img/recon/bladder_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/001313_bladder.ply -o ./tmp_img/recon/bladder_ours.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/001313_bladder.ply -o ./tmp_img/recon/bladder_gt.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02

# echo brain
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s1075_brain.ply -o ./tmp_img/recon/brain_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s1075_brain.ply -o ./tmp_img/recon/brain_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s1075_brain.ply -o ./tmp_img/recon/brain_pct2.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s1075_brain.ply -o ./tmp_img/recon/brain_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s1075_brain.ply -o ./tmp_img/recon/brain_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s1075_brain.ply -o ./tmp_img/recon/brain_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s1075_brain.ply -o ./tmp_img/recon/brain_ours.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s1075_brain.ply -o ./tmp_img/recon/brain_gt.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/brain/s1075_brain.ply -o ./tmp_img/recon/brain_input.png --size 1600,1200 --xyz_angles z/90,x/-10
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/brain/s1075_brain.ply -o ./tmp_img/recon/brain_skeleton.png --size 1600,1200 --xyz_angles z/90,x/-10

# echo colon
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0549_colon.ply -o ./tmp_img/recon/colon_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0549_colon.ply -o ./tmp_img/recon/colon_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0549_colon.ply -o ./tmp_img/recon/colon_pct2.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0549_colon.ply -o ./tmp_img/recon/colon_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0549_colon.ply -o ./tmp_img/recon/colon_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0549_colon.ply -o ./tmp_img/recon/colon_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0549_colon.ply -o ./tmp_img/recon/colon_ours.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0549_colon.ply -o ./tmp_img/recon/colon_gt.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/colon/s0549_colon.ply -o ./tmp_img/recon/colon_input.png --size 1600,1200 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/colon/s0549_colon.ply -o ./tmp_img/recon/colon_skeleton.png --size 1600,1200 --xyz_angles y/-45,x/45 --float_height 0.0

# echo duodenum
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_pct2.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_ours.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_gt.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/duodenum/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_input.png --size 1600,1200 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/duodenum/s0702_duodenum.ply -o ./tmp_img/recon/duodenum_skeleton.png --size 1600,1200 --xyz_angles 0,20,-20 --mirror_flip y

# echo gallbladder
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_pct2.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_ours.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_gt.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/gallbladder/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_input.png --size 1600,1200 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/gallbladder/s0720_gallbladder.ply -o ./tmp_img/recon/gallbladder_skeleton.png --size 1600,1200 --xyz_angles 20,0,-60 --float_height 0.02

# echo liver
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s1120_liver.ply -o ./tmp_img/recon/liver_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s1120_liver.ply -o ./tmp_img/recon/liver_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s1120_liver.ply -o ./tmp_img/recon/liver_pct2.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s1120_liver.ply -o ./tmp_img/recon/liver_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s1120_liver.ply -o ./tmp_img/recon/liver_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s1120_liver.ply -o ./tmp_img/recon/liver_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s1120_liver.ply -o ./tmp_img/recon/liver_ours.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s1120_liver.ply -o ./tmp_img/recon/liver_gt.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/liver/s1120_liver.ply -o ./tmp_img/recon/liver_input.png --size 1600,1200 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/liver/s1120_liver.ply -o ./tmp_img/recon/liver_skeleton.png --size 1600,1200 --xyz_angles -36,18.5,30 --float_height 0.04

# echo pancreas
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_pct2.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_ours.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_gt.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/pancreas/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_input.png --size 1600,1200 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/pancreas/s0311_pancreas.ply -o ./tmp_img/recon/pancreas_skeleton.png --size 1600,1200 --xyz_angles 50,180,20 --float_height -0.1

# echo stomach
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0699_stomach.ply -o ./tmp_img/recon/stomach_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0699_stomach.ply -o ./tmp_img/recon/stomach_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0699_stomach.ply -o ./tmp_img/recon/stomach_pct2.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0699_stomach.ply -o ./tmp_img/recon/stomach_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0699_stomach.ply -o ./tmp_img/recon/stomach_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0699_stomach.ply -o ./tmp_img/recon/stomach_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0699_stomach.ply -o ./tmp_img/recon/stomach_ours.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0699_stomach.ply -o ./tmp_img/recon/stomach_gt.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/stomach/s0699_stomach.ply -o ./tmp_img/recon/stomach_input.png --size 1600,1200 --xyz_angles -60,0,180
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/stomach/s0699_stomach.ply -o ./tmp_img/recon/stomach_skeleton.png --size 1600,1200 --xyz_angles -60,0,180

# echo spleen
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0437_spleen.ply -o ./tmp_img/recon/spleen_pointnet2.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0437_spleen.ply -o ./tmp_img/recon/spleen_dgcnn.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0437_spleen.ply -o ./tmp_img/recon/spleen_pct2.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0437_spleen.ply -o ./tmp_img/recon/spleen_pointmamba2.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0437_spleen.ply -o ./tmp_img/recon/spleen_diffpcd.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0437_spleen.ply -o ./tmp_img/recon/spleen_gem3d.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0437_spleen.ply -o ./tmp_img/recon/spleen_ours.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0437_spleen.ply -o ./tmp_img/recon/spleen_gt.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/spleen/s0437_spleen.ply -o ./tmp_img/recon/spleen_input.png --size 1600,1200 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/spleen/s0437_spleen.ply -o ./tmp_img/recon/spleen_skeleton.png --size 1600,1200 --float_height 0.04 --xyz_angles -180,5,0


# echo trachea
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s1052_trachea.ply -o ./tmp_img/recon/trachea_pointnet2.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s1052_trachea.ply -o ./tmp_img/recon/trachea_dgcnn.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s1052_trachea.ply -o ./tmp_img/recon/trachea_pct2.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s1052_trachea.ply -o ./tmp_img/recon/trachea_pointmamba2.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s1052_trachea.ply -o ./tmp_img/recon/trachea_diffpcd.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s1052_trachea.ply -o ./tmp_img/recon/trachea_gem3d.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s1052_trachea.ply -o ./tmp_img/recon/trachea_ours.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s1052_trachea.ply -o ./tmp_img/recon/trachea_gt.png --size 1600,1200 --color_id 6 
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/trachea/s1052_trachea.ply -o ./tmp_img/recon/trachea_input.png --size 1600,1200
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/trachea/s1052_trachea.ply -o ./tmp_img/recon/trachea_skeleton.png --size 1600,1200


# echo uterus
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/078651_uterus.ply -o ./tmp_img/recon/uterus_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/078651_uterus.ply -o ./tmp_img/recon/uterus_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/078651_uterus.ply -o ./tmp_img/recon/uterus_pct2.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/078651_uterus.ply -o ./tmp_img/recon/uterus_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/078651_uterus.ply -o ./tmp_img/recon/uterus_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/078651_uterus.ply -o ./tmp_img/recon/uterus_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/078651_uterus.ply -o ./tmp_img/recon/uterus_ours.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/078651_uterus.ply -o ./tmp_img/recon/uterus_gt.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/uterus/078651_uterus.ply -o ./tmp_img/recon/uterus_input.png --size 1600,1200 --xyz_angles 170,-5,5
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/uterus/078651_uterus.ply -o ./tmp_img/recon/uterus_skeleton.png --size 1600,1200 --xyz_angles 170,-5,5

# echo coronary-left
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/12023987_0.ply -o ./tmp_img/recon/coronary_left_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/12023987_0.ply -o ./tmp_img/recon/coronary_left_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/12023987_0.ply -o ./tmp_img/recon/coronary_left_pct2.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/12023987_0.ply -o ./tmp_img/recon/coronary_left_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/12023987_0.ply -o ./tmp_img/recon/coronary_left_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/12023987_0.ply -o ./tmp_img/recon/coronary_left_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/12023987_0.ply -o ./tmp_img/recon/coronary_left_ours.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/12023987_0.ply -o ./tmp_img/recon/coronary_left_gt.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/coronary_artery_left_d/12023987_0.ply -o ./tmp_img/recon/coronary_left_input.png --size 1600,1200 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/coronary_artery_left_d/12023987_0.ply -o ./tmp_img/recon/coronary_left_skeleton.png --size 1600,1200 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0

# echo coronary-right
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/12069367_1.ply -o ./tmp_img/recon/coronary_right_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/12069367_1.ply -o ./tmp_img/recon/coronary_right_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/12069367_1.ply -o ./tmp_img/recon/coronary_right_pct2.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/12069367_1.ply -o ./tmp_img/recon/coronary_right_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/12069367_1.ply -o ./tmp_img/recon/coronary_right_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/12069367_1.ply -o ./tmp_img/recon/coronary_right_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/12069367_1.ply -o ./tmp_img/recon/coronary_right_ours.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/12069367_1.ply -o ./tmp_img/recon/coronary_right_gt.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/input/coronary_artery_right_d/12069367_1.ply -o ./tmp_img/recon/coronary_right_input.png --size 1600,1200 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.01 --color_id 3 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/coronary_artery_right_d/12069367_1.ply -o ./tmp_img/recon/coronary_right_skeleton.png --size 1600,1200 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0

# echo coronary-both
# python ../render_pcd.py -i "/data/guoqingzhang/datasets/imageCAS/fold5/surface/12070943.ply" -o ./tmp_img/coronary_ours_surface.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --sphere_radius 0.005 --point_size 4096 --color_id 8
# python ../render_pcd.py -i "/data/guoqingzhang/datasets/imageCAS/fold5/surface/12070943.ply" -o ./tmp_img/coronary_ours_init_skeleton.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --point_size 400 --sphere_radius 0.01 --color_id 3 
# python ../render_pcd.py -i "/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/skeleton/SkCNN_val_with_SKC_LA_for_test/12070943.ply" -o ./tmp_img/coronary_ours_skeleton.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --sphere_radius 0.01 --color_id 3 
# python ../render_pcd.py -i "/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/sdf/SkCNN_val_with_SKC_LA_for_test/12070943.sdf.ply" -o coronary_ours_sdf_high_resolution.png --size 6400,4800 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --sphere_radius 0.004 --point_size 20000
# python ../render_pcd.py -i "./query_vis.ply" -o query_vis.png --size 3200,1600 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --sphere_radius 0.01
# python ../render_mesh.py -i "/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/mesh/SkCNN_val_with_SKC_LA_for_test/12070943.ply" -o ./tmp_img/coronary_ours.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --color_id 6
# python ../render_mesh.py -i "/data/guoqingzhang/datasets/imageCAS/fold5/rmesh022/12070943_mesh.ply" -o ./tmp_img/recon/coronary_gt.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --color_id 6

python ../render_pcd.py -i "/data/guoqingzhang/vcg-for-figure/gen/imagecas/10526738.ply" -o ./tmp_img/imagecas_surface.png --size 1600,1200 --float_height 0.0 --sphere_radius 0.005 --point_size 4096 --color_id 8 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-15,x/-10,z/10,x/-20
python ../render_pcd.py -i "/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/skeleton/SkCNN_val_with_SKC_LA_for_test/10526738.ply" -o ./tmp_img/imagecas_ours_skeleton.png --size 1600,1200 --float_height 0.0 --sphere_radius 0.01 --color_id 3 --point_size 400 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-15,x/-10,z/10,x/-20
python ../render_mesh.py -i "/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/mesh/SkCNN_val_with_SKC_LA_for_test/10526738.ply" -o ./tmp_img/imagecas_ours.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-15,x/-10,z/10,x/-20 --color_id 6
python ../render_mesh.py -i "/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/mesh/VessDiff/10526738_4096.ply" -o ./tmp_img/imagecas_vess_diff.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-15,x/-10,z/10,x/-20 --color_id 6

# python ../render_pcd.py -i "/data/guoqingzhang/vcg-for-figure/gen/cow/TOF_MRA_039_0000_multi.ply" -o ./tmp_img/cow_surface.png --size 1600,1200 --sphere_radius 0.005 --point_size 4096 --color_id 8 --xyz_angles y/-90,z/-5,x/20,y/-5
# python ../render_pcd.py -i "/data/guoqingzhang/vcg-results/CoW_vessel_diff/skeleton/SkCNN_val_with_SKC_LA_for_test/TOF_MRA_039_0000_multi.ply" -o ./tmp_img/cow_ours_skeleton.png --size 1600,1200 --sphere_radius 0.01 --color_id 3 --point_size 400 --xyz_angles y/-90,z/-5,x/20,y/-5
# python ../render_mesh.py -i "/data/guoqingzhang/vcg-results/CoW_vessel_diff/mesh/SkCNN_val_with_SKC_LA_for_test/TOF_MRA_039_0000_multi.ply" -o ./tmp_img/cow_ours.png --size 1600,1200 --xyz_angles y/-90,z/-5,x/20,y/-5 --color_id 6
# python ../render_mesh.py -i "/data/guoqingzhang/vcg-results/CoW_vessel_diff/mesh/VessDiff/TOF_MRA_039_0000_multi_4096.ply" -o ./tmp_img/cow_vess_diff.png --size 1600,1200 --xyz_angles y/-90,z/-5,x/20,y/-5 --color_id 6


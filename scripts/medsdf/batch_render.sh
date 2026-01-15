#!/bin/bash
#SBATCH --job-name=render
#SBATCH -w inspur1
#SBATCH --gres=shard:1
#SBATCH -c 8
#SBATCH -t 144000

# echo bladder
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/020636_bladder.ply -o bladder_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/020636_bladder.ply -o bladder_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/020636_bladder.ply -o bladder_pct2.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/020636_bladder.ply -o bladder_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/020636_bladder.ply -o bladder_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/020636_bladder.ply -o bladder_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/020636_bladder.ply -o bladder_ours.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/020636_bladder.ply -o bladder_gt.png --size 1600,1200 --color_id 6 --xyz_angles -90,110,-10
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/bladder/020636_bladder.ply -o bladder_input.png --size 1600,1200 --xyz_angles -90,110,-10
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/bladder/020636_bladder.ply -o bladder_skeleton.png --size 1600,1200 --xyz_angles -90,110,-10

# echo bladder
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/001313_bladder.ply -o bladder_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/001313_bladder.ply -o bladder_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/001313_bladder.ply -o bladder_pct2.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/001313_bladder.ply -o bladder_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/001313_bladder.ply -o bladder_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/001313_bladder.ply -o bladder_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/001313_bladder.ply -o bladder_ours.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/001313_bladder.ply -o bladder_gt.png --size 1600,1200 --color_id 6 --xyz_angles -80,10,0 --float_height 0.02

# echo brain
python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s1075_brain.ply -o brain_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s1075_brain.ply -o brain_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s1075_brain.ply -o brain_pct2.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s1075_brain.ply -o brain_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s1075_brain.ply -o brain_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s1075_brain.ply -o brain_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s1075_brain.ply -o brain_ours.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s1075_brain.ply -o brain_gt.png --size 1600,1200 --color_id 6 --xyz_angles z/90,x/-10
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/brain/s1075_brain.ply -o brain_input.png --size 1600,1200 --xyz_angles z/90,x/-10
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/brain/s1075_brain.ply -o brain_skeleton.png --size 1600,1200 --xyz_angles z/90,x/-10

# echo colon
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0549_colon.ply -o colon_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0549_colon.ply -o colon_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0549_colon.ply -o colon_pct2.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0549_colon.ply -o colon_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0549_colon.ply -o colon_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0549_colon.ply -o colon_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0549_colon.ply -o colon_ours.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0549_colon.ply -o colon_gt.png --size 1600,1200 --color_id 6 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/colon/s0549_colon.ply -o colon_input.png --size 1600,1200 --xyz_angles y/-45,x/45 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/colon/s0549_colon.ply -o colon_skeleton.png --size 1600,1200 --xyz_angles y/-45,x/45 --float_height 0.0

# echo duodenum
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0702_duodenum.ply -o duodenum_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0702_duodenum.ply -o duodenum_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0702_duodenum.ply -o duodenum_pct2.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0702_duodenum.ply -o duodenum_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0702_duodenum.ply -o duodenum_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0702_duodenum.ply -o duodenum_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0702_duodenum.ply -o duodenum_ours.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0702_duodenum.ply -o duodenum_gt.png --size 1600,1200 --color_id 6 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/duodenum/s0702_duodenum.ply -o duodenum_input.png --size 1600,1200 --xyz_angles 0,20,-20 --mirror_flip y
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/duodenum/s0702_duodenum.ply -o duodenum_skeleton.png --size 1600,1200 --xyz_angles 0,20,-20 --mirror_flip y

# echo gallbladder
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0720_gallbladder.ply -o gallbladder_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0720_gallbladder.ply -o gallbladder_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0720_gallbladder.ply -o gallbladder_pct2.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0720_gallbladder.ply -o gallbladder_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0720_gallbladder.ply -o gallbladder_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0720_gallbladder.ply -o gallbladder_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0720_gallbladder.ply -o gallbladder_ours.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0720_gallbladder.ply -o gallbladder_gt.png --size 1600,1200 --color_id 6 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/gallbladder/s0720_gallbladder.ply -o gallbladder_input.png --size 1600,1200 --xyz_angles 20,0,-60 --float_height 0.02
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/gallbladder/s0720_gallbladder.ply -o gallbladder_skeleton.png --size 1600,1200 --xyz_angles 20,0,-60 --float_height 0.02

# echo liver
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s1120_liver.ply -o liver_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s1120_liver.ply -o liver_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s1120_liver.ply -o liver_pct2.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s1120_liver.ply -o liver_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s1120_liver.ply -o liver_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s1120_liver.ply -o liver_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s1120_liver.ply -o liver_ours.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s1120_liver.ply -o liver_gt.png --size 1600,1200 --color_id 6 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/liver/s1120_liver.ply -o liver_input.png --size 1600,1200 --xyz_angles -36,18.5,30 --float_height 0.04
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/liver/s1120_liver.ply -o liver_skeleton.png --size 1600,1200 --xyz_angles -36,18.5,30 --float_height 0.04

# echo pancreas
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0311_pancreas.ply -o pancreas_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0311_pancreas.ply -o pancreas_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0311_pancreas.ply -o pancreas_pct2.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0311_pancreas.ply -o pancreas_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0311_pancreas.ply -o pancreas_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0311_pancreas.ply -o pancreas_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0311_pancreas.ply -o pancreas_ours.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0311_pancreas.ply -o pancreas_gt.png --size 1600,1200 --color_id 6 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/pancreas/s0311_pancreas.ply -o pancreas_input.png --size 1600,1200 --xyz_angles 50,180,20 --float_height -0.1
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/pancreas/s0311_pancreas.ply -o pancreas_skeleton.png --size 1600,1200 --xyz_angles 50,180,20 --float_height -0.1

# echo stomach
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0699_stomach.ply -o stomach_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0699_stomach.ply -o stomach_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0699_stomach.ply -o stomach_pct2.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0699_stomach.ply -o stomach_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0699_stomach.ply -o stomach_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0699_stomach.ply -o stomach_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0699_stomach.ply -o stomach_ours.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0699_stomach.ply -o stomach_gt.png --size 1600,1200 --color_id 6 --xyz_angles -60,0,180
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/stomach/s0699_stomach.ply -o stomach_input.png --size 1600,1200 --xyz_angles -60,0,180
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/stomach/s0699_stomach.ply -o stomach_skeleton.png --size 1600,1200 --xyz_angles -60,0,180

# echo spleen
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s0437_spleen.ply -o spleen_pointnet2.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s0437_spleen.ply -o spleen_dgcnn.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s0437_spleen.ply -o spleen_pct2.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s0437_spleen.ply -o spleen_pointmamba2.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s0437_spleen.ply -o spleen_diffpcd.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s0437_spleen.ply -o spleen_gem3d.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s0437_spleen.ply -o spleen_ours.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s0437_spleen.ply -o spleen_gt.png --size 1600,1200 --color_id 6 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/spleen/s0437_spleen.ply -o spleen_input.png --size 1600,1200 --float_height 0.04 --xyz_angles -180,5,0
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/spleen/s0437_spleen.ply -o spleen_skeleton.png --size 1600,1200 --float_height 0.04 --xyz_angles -180,5,0


# echo trachea
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/s1052_trachea.ply -o trachea_pointnet2.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/s1052_trachea.ply -o trachea_dgcnn.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/s1052_trachea.ply -o trachea_pct2.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/s1052_trachea.ply -o trachea_pointmamba2.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/s1052_trachea.ply -o trachea_diffpcd.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/s1052_trachea.ply -o trachea_gem3d.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/s1052_trachea.ply -o trachea_ours.png --size 1600,1200 --color_id 6 
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/s1052_trachea.ply -o trachea_gt.png --size 1600,1200 --color_id 6 
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/trachea/s1052_trachea.ply -o trachea_input.png --size 1600,1200
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/trachea/s1052_trachea.ply -o trachea_skeleton.png --size 1600,1200


# echo uterus
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/078651_uterus.ply -o uterus_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/078651_uterus.ply -o uterus_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/078651_uterus.ply -o uterus_pct2.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/078651_uterus.ply -o uterus_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/078651_uterus.ply -o uterus_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/078651_uterus.ply -o uterus_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/078651_uterus.ply -o uterus_ours.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/078651_uterus.ply -o uterus_gt.png --size 1600,1200 --color_id 6 --xyz_angles 170,-5,5
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/uterus/078651_uterus.ply -o uterus_input.png --size 1600,1200 --xyz_angles 170,-5,5
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/uterus/078651_uterus.ply -o uterus_skeleton.png --size 1600,1200 --xyz_angles 170,-5,5

# echo coronary-left
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/12023987_0.ply -o coronary_left_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/12023987_0.ply -o coronary_left_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/12023987_0.ply -o coronary_left_pct2.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/12023987_0.ply -o coronary_left_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/12023987_0.ply -o coronary_left_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/12023987_0.ply -o coronary_left_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/12023987_0.ply -o coronary_left_ours.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/12023987_0.ply -o coronary_left_gt.png --size 1600,1200 --color_id 6 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/coronary_artery_left_d/12023987_0.ply -o coronary_left_input.png --size 1600,1200 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/coronary_artery_left_d/12023987_0.ply -o coronary_left_skeleton.png --size 1600,1200 --xyz_angles x/100,z/-20,y/-40,x/-30,z/-10 --float_height 0.0

# echo coronary-right
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointNet2/12069367_1.ply -o coronary_right_pointnet2.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DGCNN/12069367_1.ply -o coronary_right_dgcnn.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PCT2/12069367_1.ply -o coronary_right_pct2.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/PointMamba2/12069367_1.ply -o coronary_right_pointmamba2.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/DiffPCD/12069367_1.ply -o coronary_right_diffpcd.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GeM3D_origin/12069367_1.ply -o coronary_right_gem3d.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/SkCNN_val_512_with_la/12069367_1.ply -o coronary_right_ours.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_mesh.py -i /data/guoqingzhang/vcg-for-figure/recon/GT/12069367_1.ply -o coronary_right_gt.png --size 1600,1200 --color_id 6 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.005 --point_size 2560 --color_id 34 -i /data/guoqingzhang/vcg-for-figure/recon/input/coronary_artery_right_d/12069367_1.ply -o coronary_right_input.png --size 1600,1200 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0
# python ../render_pcd.py --sphere_radius 0.01 --color_id 8 -i /data/guoqingzhang/vcg-for-figure/recon/skeleton/coronary_artery_right_d/12069367_1.ply -o coronary_right_skeleton.png --size 1600,1200 --xyz_angles x/-100,y/-90,x/-80,y/5 --float_height 0.0

# echo coronary-both
# python ../render_pcd.py -i "/data/guoqingzhang/datasets/imageCAS/fold5/surface/12070943.ply" -o coronary_ours_surface.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --sphere_radius 0.005 --point_size 4096 --color_id 34
# python ../render_pcd.py -i "/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/skeleton/SkCNN_val_with_SKC_LA_for_test/12070943.ply" -o coronary_ours_skeleton.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --sphere_radius 0.01 --color_id 8 
# python ../render_mesh.py -i "/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/mesh/SkCNN_val_with_SKC_LA_for_test/12070943.ply" -o coronary_ours.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --color_id 6
# python ../render_mesh.py -i "/data/guoqingzhang/datasets/imageCAS/fold5/rmesh022/12070943_mesh.ply" -o coronary_gt.png --size 1600,1200 --float_height 0.0 --xyz_angles y/-100,x/-30,y/-10,z/-45,y/-20,x/-20 --color_id 6
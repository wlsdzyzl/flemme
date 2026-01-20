import numpy as np
from sklearn.neighbors import KDTree
from flemme.utils import save_ply, load_ply
from flemme.color_table import color_table
point_size = 1024
skeleton_size = 10
k = 8
skeleton_pcd = load_ply("/data/guoqingzhang/vcg-results/imageCAS_vessel_diff/skeleton/SkCNN_val_with_SKC_LA_for_test/12070943.ply")
surface_pcd = load_ply("/data/guoqingzhang/datasets/imageCAS/fold5/surface/12070943.ply")
surface_pcd = surface_pcd[np.random.choice(range(surface_pcd.shape[0]), point_size, replace=False)]
skeleton_pcd = skeleton_pcd[np.random.choice(range(skeleton_pcd.shape[0]), skeleton_size, replace=False)]

kdtree = KDTree(surface_pcd)   # surface_pcd: (n, 3)
dist, idx = kdtree.query(skeleton_pcd, k=3 * k)

green_idx = np.unique(idx[:, :k].reshape(-1))
red_idx = np.unique(idx[:, k:].reshape(-1))

## remove duplicated elements
i = 0
j = 0
red_idx_array = []
while i < len(green_idx) and j < len(red_idx):
    if red_idx[j] < green_idx[i]:
        red_idx_array.append(red_idx[j])  
        j += 1
    elif red_idx[j] > green_idx[i]:
        i += 1
    else:
        i+=1
        j+=1
red_idx = np.array(red_idx_array)
surface_pcd = np.concatenate((surface_pcd[green_idx.reshape(-1)], surface_pcd[red_idx.reshape(-1)]), axis = 0)

surface_colors = np.zeros((surface_pcd.shape[0], 3))
## green
surface_colors[:green_idx.shape[0]] = color_table[15]  
## red
surface_colors[green_idx.shape[0]:] = color_table[8]

# ------------------------
# skeleton colors (blue)
# ------------------------
skeleton_colors = np.zeros((skeleton_size, 3))
skeleton_colors[:] = color_table[3]  # blue

# ------------------------
# combine point cloud
# ------------------------
points = np.concatenate([surface_pcd, skeleton_pcd], axis=0)
colors = np.concatenate([surface_colors, skeleton_colors], axis=0)

save_ply("./query_vis.ply", (points,colors))
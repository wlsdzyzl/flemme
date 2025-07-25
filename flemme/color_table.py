import numpy as np
from matplotlib import pyplot as plt
from flemme.config import module_config
scannet_color_table = np.array([[150, 150, 150],
            [174, 199, 232],     # wall
            [152, 223, 138],     # floor
            [31, 119, 180],      # cabinet
            [255, 187, 120],     # bed
            [188, 189, 34],      # chair
            [140, 86, 75],       # sofa
            [255, 152, 150],     # table
            [214, 39, 40],       # door
            [197, 176, 213],     # window
            [148, 103, 189],     # bookshelf
            [196, 156, 148],     # picture
            [23, 190, 207],      # counter
            [178, 76, 76],
            [247, 182, 210],     # desk
            [66, 188, 102],
            [219, 219, 141],     # curtain
            [140, 57, 197],
            [202, 185, 52],
            [51, 176, 203],
            [200, 54, 131],
            [92, 193, 61],
            [78, 71, 183],
            [172, 114, 82],
            [255, 127, 14],      # refrigerator
            [91, 163, 138],
            [153, 98, 156],
            [140, 153, 101],
            [158, 218, 229],     # shower curtain
            [100, 125, 154],
            [178, 127, 135],
            [120, 185, 128],
            [146, 111, 194],
            [44, 160, 44],       # toilet
            [112, 128, 144],     # sink
            [96, 207, 209],
            [227, 119, 194],     # bathtub
            [213, 92, 176],
            [94, 106, 211],
            [82, 84, 163],       # otherfurn
            [100, 85, 144]]) / 255
custom_color_table = np.array([
                [51, 255, 255],
                [101, 255, 101],
                [255, 153, 204], 
                [127, 0, 255],
                [255, 255, 51],
                [0, 0, 255],
                ]) / 255
def get_mpl_cmap(name, label_count):
    if name in ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                      'tab20c']:
        colors = plt.get_cmap(name).colors
    else:
        cm = plt.get_cmap(name, label_count)
        colors = tuple(cm(l) for l in range(label_count))
    return np.array(colors)
def get_color_table(table_name, label_count = 20):
    if table_name == 'Scannet':
        return scannet_color_table
    elif table_name == 'Custom':
        return custom_color_table
    else:
        return get_mpl_cmap(table_name, label_count)
color_table = get_color_table(module_config['color_map'])
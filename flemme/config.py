import logging
module_config = {
    ### einops
    'transformer': True,
    ### mamba-ssm, cuda version >= 11.6
    'mamba': True,
    ### plyfile, POT
    'point-cloud': True,
    ### geometric_torch
    'graph': False,
    'logger_level': logging.INFO,
    'color_map': 'Scannet',
    'suppress_simpleitk_warning': True
    'cuda_arch_list': "5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
}
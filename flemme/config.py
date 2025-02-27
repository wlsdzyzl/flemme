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
    'color_map': 'Scannet'

}
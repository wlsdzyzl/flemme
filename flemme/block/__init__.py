from flemme.config import module_config
from flemme.block.common import *
if module_config['transformer']:
    from flemme.block.swin import *
if module_config['mamba']:
    from flemme.block.vmamba import *
if module_config['point-cloud']:
    from flemme.block.pcd import *
if module_config['graph']:
    from flemme.block.graph import *

from flemme.logger import get_logger
logger = get_logger('flemme.block')

def get_building_block(name, **kwargs):
    logger.debug('building block parameters: {}'.format(kwargs))
    # image
    if name in ['single','conv']:
        return partial(ConvBlock, **kwargs)
    elif name in ['double', 'double_conv']:
        return partial(DoubleConvBlock, **kwargs)
    elif name in ['res', 'res_conv']:
        return partial(ResConvBlock, **kwargs)
    elif name == 'dense' or name == 'fc':
        return partial(DenseBlock, **kwargs)
    elif name == 'double_dense' or name == 'double_fc':
        return partial(DoubleDenseBlock, **kwargs)
    elif name == 'res_dense' or name == 'res_fc':
        return partial(ResDenseBlock, **kwargs)
    elif name == 'vit':
        return partial(VisionTransformerBlock, **kwargs)
    elif name == 'swin':
        return partial(SwinTransformerBlock, **kwargs)
    elif name == 'double_swin':
        return partial(DoubleSwinTransformerBlock, **kwargs)
    elif name == 'res_swin':
        return partial(ResSwinTransformerBlock, **kwargs)
    elif name == 'vmamba':
        return partial(VMambaBlock, **kwargs)
    elif name == 'double_vmamba':
        return partial(DoubleVMambaBlock, **kwargs)
    elif name == 'res_vmamba':
        return partial(ResVMambaBlock, **kwargs)
    elif name == 'vmamba2':
        return partial(VMamba2Block, **kwargs)
    elif name == 'double_vmamba2':
        return partial(DoubleVMamba2Block, **kwargs)
    elif name == 'res_vmamba2':
        return partial(ResVMamba2Block, **kwargs)
    # point cloud
    ## point cloud transformer with self attention 
    elif name == 'pct_sa':
        return partial(PointTransformerBlock, attention='SA', **kwargs)
    ## point cloud transformer with offset attention
    elif name == 'pct_oa':
        return partial(PointTransformerBlock, attention='OA', **kwargs)
    elif name == 'pct_sa_non_ffn':
        return partial(PointTransformerNonFFNBlock, attention='SA', **kwargs)
    ## point cloud transformer with offset attention
    elif name == 'pct_oa_non_ffn':
        return partial(PointTransformerNonFFNBlock, attention='OA', **kwargs)
    ## point cloud mamba
    elif name == 'pmamba':
        return partial(PointMambaBlock, mamba='Mamba', **kwargs)
    ## point cloud mamba2
    elif name == 'pmamba2':
        return partial(PointMambaBlock, mamba='Mamba2', **kwargs)
    elif name == 'pmamba_non_ffn':
        return partial(PointMambaNonFFNBlock, mamba='Mamba', **kwargs)
    ## point cloud mamba2
    elif name == 'pmamba2_non_ffn':
        return partial(PointMambaNonFFNBlock, mamba='Mamba2', **kwargs)
    ## graph convolution layer
    elif name == 'gcn':
        return partial(GraphConvBlock, **kwargs)
    elif name == 'cheb':
        return partial(ChebConvBlock, **kwargs)
    elif name == 'gtrans':
        return partial(TransformerConv, **kwargs)
    else:
        logger.error(f'Unsupported building block: {name}')
        exit(1)

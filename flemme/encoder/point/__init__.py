from flemme.config import module_config
supported_point_encoders = {}
if module_config['point-cloud']:
    from .pointnet import *
    from .pointtrans import *
    supported_point_encoders['PointNet'] = (PointNetEncoder, PointNetDecoder)
    supported_point_encoders['PointTrans'] = (PointTransEncoder, PointTransDecoder)
    if module_config['mamba']:
        from .pointmamba import *
        supported_point_encoders['PointMamba'] = (PointMambaEncoder, PointMambaDecoder)
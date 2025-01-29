from flemme.config import module_config
supported_point_encoders = {}
if module_config['point-cloud']:
    from .pointnet import *
    from .pointtrans import *
    from .pointnet2 import *
    from .pointtrans2 import *
    
    supported_point_encoders['PointNet'] = (PointNetEncoder, PointNetDecoder)
    supported_point_encoders['PointNet2'] = (PointNet2Encoder, PointNet2Decoder)
    supported_point_encoders['PointTrans'] = (PointTransEncoder, PointNetDecoder)
    supported_point_encoders['PointTrans2'] = (PointTrans2Encoder, PointTrans2Decoder)
    if module_config['mamba']:
        from .pointmamba import *
        from .pointmamba2 import *
        supported_point_encoders['PointMamba'] = (PointMambaEncoder, PointNetDecoder)
        supported_point_encoders['PointMamba2'] = (PointMamba2Encoder, PointMamba2Decoder)
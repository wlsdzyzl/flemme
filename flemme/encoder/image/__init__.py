from flemme.config import module_config
from .cnn import *
from .unet import *
from .dnet import *
supported_image_encoders = {'CNN':(CNNEncoder, CNNDecoder), 
                      'UNet':(UNetEncoder, UNetDecoder), 
                      'DNet': (DNetEncoder, DNetDecoder),}
if module_config['transformer']:
    from .vit import *
    from .swin import *
    supported_image_encoders['ViT'] = (ViTEncoder, ViTDecoder)
    supported_image_encoders['Swin'] = (SwinEncoder, SwinDecoder)
    supported_image_encoders['ViTU'] = (ViTUNetEncoder, ViTUNetDecoder)
    supported_image_encoders['SwinU'] = (SwinUNetEncoder, SwinUNetDecoder)
    supported_image_encoders['ViTD'] = (ViTDNetEncoder, ViTDNetDecoder)
    supported_image_encoders['SwinD'] = (SwinDNetEncoder, SwinDNetDecoder)
if module_config['mamba']:
    from .vmamba import *
    supported_image_encoders['VMamba'] = (VMambaEncoder, VMambaDecoder)
    supported_image_encoders['VMambaU'] = (VMambaUNetEncoder, VMambaUNetDecoder)
    supported_image_encoders['VMambaD'] = (VMambaDNetEncoder, VMambaDNetDecoder)
from flemme.config import module_config

from .cnn import *
from .unet import *
from .dnet import *
from .pointwise import *
from flemme.utils import DataForm
from flemme.logger import get_logger

supported_encoders = {'CNN':(CNNEncoder, CNNDecoder), 
                      'UNet':(UNetEncoder, UNetDecoder), 
                      'DNet': (DNetEncoder, DNetDecoder),
                      'PointWise':(PointWiseEncoder, PointWiseDecoder)}
if module_config['transformer']:
    from .vit import *
    from .swin import *
    supported_encoders['ViT'] = (ViTEncoder, ViTDecoder)
    supported_encoders['Swin'] = (SwinEncoder, SwinDecoder)
    supported_encoders['ViTU'] = (ViTUNetEncoder, ViTUNetDecoder)
    supported_encoders['SwinU'] = (SwinUNetEncoder, SwinUNetDecoder)
    supported_encoders['ViTD'] = (ViTDNetEncoder, ViTDNetDecoder)
    supported_encoders['SwinD'] = (SwinDNetEncoder, SwinDNetDecoder)
if module_config['mamba']:
    from .vmamba import *
    supported_encoders['VMamba'] = (VMambaEncoder, VMambaDecoder)
    supported_encoders['VMambaU'] = (VMambaUNetEncoder, VMambaUNetDecoder)
    supported_encoders['VMambaD'] = (VMambaDNetEncoder, VMambaDNetDecoder)
if module_config['point-cloud']:
    from .pointnet import *
    from .pointtrans import *
    supported_encoders['PointNet'] = (PointNetEncoder, PointNetDecoder)
    supported_encoders['PointTrans'] = (PointTransEncoder, PointTransDecoder)

    if module_config['mamba']:
        from .pointmamba import *
        supported_encoders['PointMamba'] = (PointMambaEncoder, PointMambaDecoder)

supported_buildingblocks_for_encoder = {'CNN': ['single', 'conv', 'double', 'double_conv', 'res', 'res_conv'],
                        'UNet': ['single', 'conv', 'double', 'double_conv', 'res', 'res_conv'],
                        'DNet': ['single', 'conv', 'double', 'double_conv', 'res', 'res_conv'],
                        'ViT': ['vit'],
                        'ViTU': ['vit'],
                        'ViTD': ['vit'],
                        'Swin': ['swin', 'double_swin', 'res_swin',],
                        'SwinU': ['swin', 'double_swin', 'res_swin',],
                        'SwinD': ['swin', 'double_swin', 'res_swin',],
                        'VMamba': ['vmamba', 'double_vmamba', 'res_vmamba', 'vmamba2', 'double_vmamba2', 'res_vmamba2'],
                        'VMambaU': ['vmamba', 'double_vmamba', 'res_vmamba', 'vmamba2', 'double_vmamba2', 'res_vmamba2'],
                        'VMambaD': ['vmamba', 'double_vmamba', 'res_vmamba', 'vmamba2', 'double_vmamba2', 'res_vmamba2'],
                        'PointWise': ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc'],
                        'PointNet': ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc'],
                        'PointTrans': ['pct_sa', 'pct_oa'],
                        'PointMamba': ['pmamba', 'pmamba2'],}

logger = get_logger('model.encoder.create_encoder')

def create_encoder(encoder_config, return_encoder = True, return_decoder = True):
        encoder_name = encoder_config.pop('name')
        if not encoder_name in supported_encoders:
            logger.error(f'Unsupported encoder: {encoder_name}')
            exit(1)        
        Encoder, Decoder = supported_encoders[encoder_name]
        ### building block usually can be fully-connected block, convolution block, swin or mamba block
        building_block = encoder_config.pop('building_block', 'single')
        time_channel = encoder_config.pop('time_channel', 0)
        dai_channel = encoder_config.pop('decoder_additional_in_channel', 0)
        eai_channel = encoder_config.pop('encoder_additional_in_channel', 0)
        encoder, decoder = None, None
        ### construct encoder and decoder
        if encoder_name in ('CNN', 'UNet', 'DNet', 'ViT', 'ViTU', 'ViTD', 'Swin', 'SwinU', 'SwinD', 'VMamba',  'VMambaU', 'VMambaD'):
            data_form = DataForm.IMG
        elif encoder_name in ('PointNet', 'PointTrans', 'PointMamba'):
            data_form = DataForm.PCD
        ### point-wise encoder can be applied on various data forms.
        elif encoder_name in ('PointWise', ):
            data_form = encoder_config.pop('data_form', 'PCD')
            if data_form == 'PCD':
                data_form = DataForm.PCD
            elif data_form == 'VEC':
                data_form = DataForm.VEC
            else:
                raise RuntimeError(f'Unsupported data form for point-wise encoder: {data_form}')
                
        else:
            raise RuntimeError(f'Unsupported encoder: {encoder_name}')
        if not building_block in supported_buildingblocks_for_encoder[encoder_name]:
            logger.error(f'Unsupported building block \'{building_block}\' for encoder {encoder_name}, please use one of {supported_buildingblocks_for_encoder[encoder_name]}.')
            exit(1)
        ### FC channels
        dense_channels = encoder_config.pop('dense_channels', [])
        if not isinstance(dense_channels, list):
            dense_channels = [dense_channels]    

        de_dense_channels = encoder_config.pop('decoder_dense_channels', None) or dense_channels[::-1]    
        
        ## read in / out channel of data form
        #### in channel of encoder
        in_channel = encoder_config.pop('in_channel', 1)
        #### out channel of decoder
        out_channel = encoder_config.pop('out_channel', None) or in_channel

        ### doesn't create encoder, encoder out channel should be specified.
        if not return_encoder:
            decoder_in_channel = encoder_config.pop('decoder_in_channel', None)
            assert decoder_in_channel is not None, "Need in channel for decoder."

        if encoder_name == 'PointWise':
            if return_encoder:
                encoder = Encoder(point_dim=in_channel + eai_channel, 
                                            time_channel=time_channel, 
                                            dense_channels=dense_channels,
                                            building_block=building_block,
                                            **encoder_config)
                decoder_in_channel = encoder.out_channel
            if return_decoder:
                decoder = Decoder(point_dim=out_channel, 
                                        in_channel=decoder_in_channel + dai_channel,
                                        time_channel=time_channel, 
                                        dense_channels=de_dense_channels,
                                        building_block=building_block,
                                        **encoder_config)
        if data_form == DataForm.IMG:
            logger.debug('Model is constructed for images.')
            image_size = encoder_config.pop('image_size', None)
            assert image_size is not None, "Image size need to be specified."
            if not isinstance(image_size, list):
                image_size = [image_size, image_size]       
            ## feature channels in the up/donw sampling stages
            down_channels = encoder_config.pop('down_channels', [])  
            up_channels = encoder_config.pop('up_channels', None)   
            if up_channels is None:
                up_channels = down_channels[::-1]    
            ## feature channels in the middle convolution stages
            middle_channels = encoder_config.pop('middle_channels', [])
            final_channels = encoder_config.pop('final_channels', [])
            patch_channel = encoder_config.pop('patch_channel', 32)
            patch_size = encoder_config.pop('patch_size', 2)
            assert isinstance(down_channels, list) and \
                    isinstance(up_channels, list) and \
                    isinstance(middle_channels, list) and \
                    isinstance(final_channels, list),\
                'feature channels should be a list.'
            assert len(down_channels) + len(middle_channels) != 0, \
                'Model for image doesn\'t contain any convolution layers'
            assert len(up_channels) == len(down_channels), \
                "The up-sampling in encoder is not consistent with the down-sampling in decoder!" 
            
            if 'UNet' in encoder_name or encoder_name[-1] == 'U' or \
                    'DNet' in encoder_name or encoder_name[-1] == 'D':
                assert len(down_channels) >= 2, \
                    "the number of down-sampling layers for U-Net should be larger than 1"
                assert sum( d != u for d, u in zip(down_channels, up_channels[::-1])) == 0, \
                    'For DNet and UNet, up_channels should be a reversed down_channels'
                if 'UNet' in encoder_name or encoder_name[-1] == 'U':
                    assert len(middle_channels) > 0 and middle_channels[-1] == 2 * down_channels[-1],\
                        "UNet follows a special structure, the last value of middle channels should be twice of the last value of down channels."
                
            if len(dense_channels) == 0:
                logger.debug('Current model doesn\'t contain any fully connected (dense) layers')

            ### convolution based mmodel
            if encoder_name in ['UNet', 'DNet', 'CNN']:
                down_attens = encoder_config.pop('down_attens', None)
                if not isinstance(down_attens, list): 
                    down_attens = [down_attens for _ in down_channels] 
               
                up_attens = encoder_config.pop('up_attens', None)
                if not isinstance(up_attens, list): 
                    up_attens = [up_attens for _ in up_channels] 
                ### define shape scale factor
                shape_scaling = encoder_config.pop('shape_scaling', 2)
                if not isinstance(shape_scaling, list):
                    shape_scaling = [shape_scaling for _ in down_channels]

                middle_attens = encoder_config.pop('middle_attens', None)
                if not isinstance(middle_attens, list): 
                    middle_attens = [middle_attens for _ in middle_channels] 

                final_attens = encoder_config.pop('final_attens', None)
                if not isinstance(final_attens, list): 
                    final_attens = [final_attens for _ in final_channels] 
                
                if return_encoder:
                    encoder = Encoder(image_size=image_size, image_channel = in_channel + eai_channel, 
                                                time_channel = time_channel,
                                                patch_size = patch_size,
                                                patch_channel = patch_channel,
                                                down_channels=down_channels, 
                                                down_attens=down_attens,
                                                shape_scaling = shape_scaling,
                                                middle_channels=middle_channels,
                                                middle_attens=middle_attens,
                                                dense_channels = dense_channels, 
                                                building_block=building_block, 
                                                **encoder_config)
                    decoder_in_channel = encoder.out_channel
                if return_decoder:
                    decoder = Decoder(image_size=image_size, image_channel = out_channel, 
                                                in_channel=decoder_in_channel + dai_channel, 
                                                time_channel = time_channel,
                                                patch_size = patch_size,
                                                dense_channels = de_dense_channels, 
                                                up_channels=up_channels, 
                                                up_attens=up_attens,
                                                shape_scaling = shape_scaling[::-1],
                                                final_channels=final_channels,
                                                final_attens=final_attens,
                                                building_block=building_block, 
                                                **encoder_config)
            elif encoder_name in ['ViT', 'ViTU', 'ViTD', 'Swin', 'SwinU', 'SwinD']:

                down_num_heads = encoder_config.pop('down_num_heads', 3)
                if not isinstance(down_num_heads, list): 
                    down_num_heads = [down_num_heads for _ in down_channels] 
               
                up_num_heads = encoder_config.pop('up_num_heads', 3)
                if not isinstance(up_num_heads, list): 
                    up_num_heads = [up_num_heads for _ in up_channels] 

                middle_num_heads = encoder_config.pop('middle_num_heads', 3)
                if not isinstance(middle_num_heads, list): 
                    middle_num_heads = [middle_num_heads for _ in middle_channels] 

                final_num_heads = encoder_config.pop('final_num_heads', 3)
                if not isinstance(final_num_heads, list): 
                    final_num_heads = [final_num_heads for _ in final_channels] 
                if return_encoder:
                    encoder = Encoder(image_size = image_size, 
                                        image_channel = in_channel + eai_channel, 
                                        time_channel = time_channel, 
                                        patch_size = patch_size,
                                        patch_channel = patch_channel,
                                        down_channels = down_channels, 
                                        middle_channels = middle_channels,
                                        down_num_heads = down_num_heads, 
                                        middle_num_heads = middle_num_heads,
                                        building_block=building_block,
                                        **encoder_config)
                    decoder_in_channel = encoder.out_channel
                if return_decoder:
                    decoder = Decoder(image_size = image_size, 
                                    image_channel = out_channel, 
                                    in_channel = decoder_in_channel + dai_channel,
                                    time_channel = time_channel, 
                                    patch_size = patch_size,
                                    up_channels = up_channels, 
                                    final_channels = final_channels,
                                    up_num_heads = up_num_heads, 
                                    final_num_heads = final_num_heads,
                                    building_block=building_block,
                                    **encoder_config)
            elif encoder_name in ['VMamba', 'VMambaU', 'VMambaD']:
                if return_encoder:
                    encoder = Encoder(image_size = image_size, 
                                image_channel = in_channel + eai_channel, 
                                time_channel = time_channel, 
                                patch_size = patch_size,
                                patch_channel = patch_channel,
                                down_channels = down_channels, 
                                middle_channels = middle_channels,
                                building_block=building_block,
                                **encoder_config)
                    decoder_in_channel = encoder.out_channel
                if return_decoder:
                    decoder = Decoder(image_size = image_size, 
                                image_channel = out_channel, 
                                in_channel = decoder_in_channel + dai_channel,
                                time_channel = time_channel, 
                                patch_size = patch_size,
                                up_channels = up_channels, 
                                final_channels = final_channels,
                                building_block=building_block,
                                **encoder_config)
            ## set image size
            if return_encoder:
                encoder.image_size = image_size     
            if return_decoder:
                decoder.image_size = image_size       
        elif data_form == DataForm.PCD:
            logger.debug('Model is constructed for point cloud.')
            #### point cloud encoder
            point_num = encoder_config.pop('point_num', 2048)
            if not encoder_name == 'PointWise':
                # encoder
                projection_channel = encoder_config.pop('projection_channel', 64)
                local_feature_channels = encoder_config.pop('local_feature_channels', [64, 128, 256])  
                assert isinstance(local_feature_channels, list), 'feature channels should be a list.'
                ## 0: without using local graph
                local_graph_k = encoder_config.pop('local_graph_k', 0)
                # decoder
                folding_times = encoder_config.pop('folding_times', 0)
                base_shape_config = encoder_config.pop('base_shape', {})
                if return_encoder:
                    encoder = Encoder(point_dim=in_channel + eai_channel, 
                                                projection_channel = projection_channel,
                                                time_channel = time_channel,
                                                local_graph_k = local_graph_k,
                                                local_feature_channels=local_feature_channels, 
                                                dense_channels=dense_channels, 
                                                building_block=building_block,
                                                **encoder_config)
                    decoder_in_channel = encoder.out_channel
                ## Different encoders of point clouds could use the same decoder.
                if return_decoder:
                    decoder = Decoder(point_dim=out_channel, point_num=point_num, 
                                                time_channel = time_channel, 
                                                in_channel=decoder_in_channel + dai_channel, 
                                                dense_channels=de_dense_channels,
                                                folding_times = folding_times,
                                                base_shape_config = base_shape_config,
                                                **encoder_config)
            if return_encoder:
                encoder.point_num = point_num
            if return_decoder:
                decoder.point_num = point_num
        elif data_form == DataForm.VEC:
            logger.debug('Model is constructed for points.')
        else:
            raise NotImplementedError
        ## set data_form
        if return_encoder:
            encoder.data_form = data_form
        if return_decoder:
            decoder.data_form = data_form

        return encoder, decoder
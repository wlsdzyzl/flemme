from flemme.config import module_config
from .fcn import *
from .image import *
from .point import *
from .graph import *
from flemme.utils import DataForm
from flemme.logger import get_logger

supported_encoders = {'FCN':(FCNEncoder, FCNDecoder)}
supported_encoders.update(supported_image_encoders)
supported_encoders.update(supported_point_encoders)
supported_encoders.update(supported_graph_encoders)

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
                        'FCN': ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc'],
                        'SeqNet': ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc'],
                        'PointNet': ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc'],
                        'PointNet2': ['dense', 'double_dense', 'res_dense', 'fc', 'double_fc', 'res_fc'],
                        'SeqTrans': ['pct_sa', 'pct_oa', 'pct_sa_non_ffn', 'pct_oa_non_ffn'],
                        'PointTrans': ['pct_sa', 'pct_oa', 'pct_sa_non_ffn', 'pct_oa_non_ffn'],
                        'PointTrans2': ['pct_sa', 'pct_oa', 'pct_sa_non_ffn', 'pct_oa_non_ffn'],
                        'SeqMamba': ['pmamba', 'pmamba2', 'pmamba_non_ffn', 'pmamba2_non_ffn'],
                        'PointMamba': ['pmamba', 'pmamba2', 'pmamba_non_ffn', 'pmamba2_non_ffn'],
                        'PointMamba2': ['pmamba', 'pmamba2', 'pmamba_non_ffn', 'pmamba2_non_ffn'],
                        'GCN': ['gcn'],
                        'Cheb': ['cheb'],
                        'GTrans': ['gtrans']}

logger = get_logger('encoder.create_encoder')

def create_encoder(encoder_config, return_encoder = True, return_decoder = True):
        encoder_name = encoder_config.pop('name')
        if not encoder_name in supported_encoders:
            logger.error(f'Unsupported encoder: {encoder_name}, should be one of {supported_encoders.keys()}')
            exit(1)        
        Encoder, Decoder = supported_encoders[encoder_name]
        ### building block usually can be fully-connected block, convolution block, swin or mamba block
        building_block = encoder_config.pop('building_block', 'single')

        encoder, decoder = None, None
        ### construct encoder and decoder
        if encoder_name in supported_image_encoders:
            data_form = DataForm.IMG
        elif encoder_name in supported_point_encoders:
            data_form = DataForm.PCD
        elif encoder_name in supported_graph_encoders:
            data_form = DataForm.GRAPH
        ### point-wise encoder can be applied on various data forms.
        elif encoder_name in ('FCN', ):
            data_form = DataForm.VEC
        else:
            raise RuntimeError(f'Unsupported encoder: {encoder_name}')
        if not building_block in supported_buildingblocks_for_encoder[encoder_name]\
            and not (return_encoder == False and encoder_name in ['PointNet', 'PointTrans', 'PointMamba']):
            ### pointnet decoder doesn't need building block
            logger.error(f'Unsupported building block \'{building_block}\' for encoder {encoder_name}, please use one of {supported_buildingblocks_for_encoder[encoder_name]}.')
            exit(1)
        ### FC channels
        dense_channels = encoder_config.pop('dense_channels', [])
        if not isinstance(dense_channels, list):
            dense_channels = [dense_channels]    

        de_dense_channels = encoder_config.pop('decoder_dense_channels', [])   
        
        ## read in / out channel of data form
        #### in channel of encoder
        in_channel = encoder_config.pop('in_channel')
        #### out channel of decoder
        out_channel = encoder_config.pop('out_channel')

        ### doesn't create encoder, encoder out channel should be specified.
        if not return_encoder:
            latent_channel = encoder_config.pop('latent_channel', None)
            latent_channels = encoder_config.pop('latent_channels', None)
            assert latent_channel is not None, "Input channel of decoder (latent_channel) is not specified."

        if encoder_name == 'FCN':
            if return_encoder:
                encoder = Encoder(vec_dim=in_channel,
                                        dense_channels=dense_channels,
                                        building_block=building_block,
                                        **encoder_config)
                latent_channel = encoder.out_channel
            if return_decoder:
                decoder = Decoder(vec_dim=out_channel, 
                                        latent_channel=latent_channel,
                                        dense_channels=de_dense_channels,
                                        building_block=building_block,
                                        **encoder_config)
        if data_form == DataForm.IMG:
            ### images' vector embedding is determined by dense_channels.
            if not len(de_dense_channels):
                de_dense_channels = dense_channels[::-1]
            if not len(dense_channels) and len(de_dense_channels):
                logger.error("Please specify the dense layers in image encoder if decoder contained dense layers.")
                exit(1)
            logger.info('Model is constructed for images.')
            image_size = encoder_config.pop('image_size', None)
            assert image_size is not None, "Image size need to be specified."
            if not isinstance(image_size, list):
                image_size = [image_size, image_size]       
            ## feature channels in the up/donw sampling stages
            down_channels = encoder_config.pop('down_channels', [])  
            up_channels = encoder_config.pop('up_channels', [])   
            ## feature channels in the middle convolution stages
            middle_channels = encoder_config.pop('middle_channels', [])
            final_channels = encoder_config.pop('final_channels', [])
            assert isinstance(down_channels, list) and \
                    isinstance(up_channels, list) and \
                    isinstance(middle_channels, list) and \
                    isinstance(final_channels, list),\
                'feature channels should be a list.'
            # if len(down_channels) + len(up_channels) == 0:
            #     down_channels = [64, 128, 256]
            if not len(up_channels):
                up_channels = down_channels[::-1]    
            if not len(down_channels):
                down_channels = up_channels[::-1]
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
                assert len(dense_channels) == 0, "DNet and UNet doesn't support dense layers in encoder."
                if 'UNet' in encoder_name or encoder_name[-1] == 'U':
                    assert len(middle_channels) > 0 and middle_channels[-1] == 2 * down_channels[-1],\
                        "UNet follows a special structure, the last value of middle channels should be twice of the last value of down channels."
            elif len(dense_channels) == 0:
                logger.info('Current model doesn\'t contain any fully connected (dense) layers')

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
                    encoder = Encoder(image_size=image_size, image_channel = in_channel, 
                                                down_channels=down_channels, 
                                                down_attens=down_attens,
                                                shape_scaling = shape_scaling,
                                                middle_channels=middle_channels,
                                                middle_attens=middle_attens,
                                                dense_channels = dense_channels, 
                                                building_block=building_block, 
                                                **encoder_config)
                    latent_channel = encoder.out_channel
                if return_decoder:
                    decoder = Decoder(image_size=image_size, image_channel = out_channel, 
                                                latent_channel=latent_channel, 
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
                                        image_channel = in_channel, 
                                        down_channels = down_channels, 
                                        middle_channels = middle_channels,
                                        down_num_heads = down_num_heads, 
                                        middle_num_heads = middle_num_heads,
                                        building_block=building_block,
                                        dense_channels=dense_channels,
                                        **encoder_config)
                    latent_channel = encoder.out_channel
                if return_decoder:
                    decoder = Decoder(image_size = image_size, 
                                    image_channel = out_channel, 
                                    latent_channel = latent_channel,
                                    up_channels = up_channels, 
                                    final_channels = final_channels,
                                    up_num_heads = up_num_heads, 
                                    final_num_heads = final_num_heads,
                                    building_block=building_block,
                                    dense_channels = de_dense_channels, 
                                    **encoder_config)
            elif encoder_name in ['VMamba', 'VMambaU', 'VMambaD']:
                if return_encoder:
                    encoder = Encoder(image_size = image_size, 
                                image_channel = in_channel, 
                                down_channels = down_channels, 
                                middle_channels = middle_channels,
                                building_block=building_block,
                                dense_channels = dense_channels, 
                                **encoder_config)
                    latent_channel = encoder.out_channel
                if return_decoder:
                    decoder = Decoder(image_size = image_size, 
                                image_channel = out_channel, 
                                latent_channel = latent_channel,
                                up_channels = up_channels, 
                                final_channels = final_channels,
                                dense_channels = de_dense_channels, 
                                building_block=building_block,
                                **encoder_config)
            ## set image size
            if return_encoder:
                encoder.image_size = image_size     
            if return_decoder:
                decoder.image_size = image_size       
        elif data_form == DataForm.PCD:
            logger.info('Model is constructed for point cloud.')
            #### point cloud encoder
            point_num = encoder_config.pop('point_num', 2048)
            voxel_resolutions = encoder_config.pop('voxel_resolutions', [])
            voxel_attens = encoder_config.pop('voxel_attens', None)
            assert type(voxel_resolutions) == list, "voxel_resolutions should be a list."
            if not isinstance(voxel_attens, list): 
                voxel_attens = [voxel_attens,] * len(voxel_resolutions)
            assert len(voxel_attens) == len(voxel_resolutions), 'voxel_attens and voxel_resolutions should have the same length.'
            
            if not 'Seq' in encoder_name:
                
                if not encoder_name[-1] == '2':
                    ## 0: without using local graph
                    local_feature_channels = encoder_config.pop('local_feature_channels', [64, 128, 256])  
                    assert isinstance(local_feature_channels, list), 'feature channels should be a list.'
                    # decoder
                    if return_encoder:
                        encoder = Encoder(point_dim=in_channel, 
                                                    local_feature_channels=local_feature_channels, 
                                                    dense_channels=dense_channels, 
                                                    building_block=building_block,
                                                    voxel_resolutions = voxel_resolutions,
                                                    voxel_attens = voxel_attens,
                                                    **encoder_config)
                        latent_channel = encoder.out_channel
                    ## Different encoders of point clouds could use the same decoder.
                    if return_decoder:
                        decoder = Decoder(point_dim=out_channel, point_num=point_num, 
                                                    latent_channel=latent_channel, 
                                                    dense_channels=de_dense_channels,
                                                    **encoder_config)
                else:
                    fps_feature_channels = encoder_config.pop('fps_feature_channels', [])
                    fp_channels = encoder_config.pop('fp_channels', [])
                    assert isinstance(fps_feature_channels, list) and \
                        isinstance(fp_channels, list), 'feature channels should be a list.'

                    if len(fps_feature_channels) + len(fp_channels) == 0:
                        fps_feature_channels = [128, 256, 512, 1024]
                    if not len(fp_channels):
                        fp_channels = fps_feature_channels[::-1]
                    if not len(fps_feature_channels):
                        fps_feature_channels = fp_channels[::-1]
                    assert len(fp_channels) == len(fps_feature_channels), \
                        'Point2Encoder should have a same number of feature propagating layers and sampling and grouping layers.'

                    if return_encoder:
                        encoder = Encoder(point_dim=in_channel, 
                                            fps_feature_channels=fps_feature_channels, 
                                            dense_channels=dense_channels, 
                                            building_block=building_block,
                                            voxel_resolutions = voxel_resolutions,
                                            voxel_attens = voxel_attens,
                                            **encoder_config)
                        
                        latent_channels = encoder.out_channels
                    ## Different encoders of point clouds could use the same decoder.
                    if return_decoder:
                        de_voxel_resolutions = encoder_config.pop('decoder_voxel_resolutions', [])
                        de_voxel_attens = encoder_config.pop('decoder_voxel_attens', None)
                        assert type(de_voxel_resolutions) == list, "decoder_voxel_resolutions should be a list."
                        if not isinstance(de_voxel_attens, list): 
                            de_voxel_attens = [de_voxel_attens,] * len(de_voxel_resolutions)
                        assert len(de_voxel_attens) == len(de_voxel_resolutions), 'decoder_voxel_attens and decoder_voxel_resolutions should have the same length.'

                        decoder = Decoder(point_dim=out_channel, point_num=point_num, 
                                                latent_channels=latent_channels, 
                                                dense_channels=de_dense_channels,
                                                building_block=building_block,
                                                fp_channels = fp_channels,
                                                voxel_resolutions = de_voxel_resolutions,
                                                voxel_attens = de_voxel_attens,
                                                **encoder_config)
            ### sequential net
            else:
                seq_feature_channels = encoder_config.pop('seq_feature_channels', [64, 128, 256])
                decoder_seq_feature_channels = encoder_config.pop('decoder_seq_feature_channels', [])
                assert isinstance(seq_feature_channels, list) and \
                    isinstance(decoder_seq_feature_channels, list), 'feature channels should be a list.'
                assert len(seq_feature_channels) > 0, "sequential feature channels in encoder shouldn't be an empty list."
                if return_encoder:
                    encoder = Encoder(point_dim=in_channel, 
                                                seq_feature_channels = seq_feature_channels,
                                                building_block=building_block,
                                                voxel_resolutions = voxel_resolutions,
                                                voxel_attens = voxel_attens,
                                                **encoder_config)
                    latent_channel = encoder.out_channel
                ## Different encoders of point clouds could use the same decoder.
                if return_decoder:
                    decoder = Decoder(point_dim=out_channel, 
                                                latent_channel=latent_channel, 
                                                seq_feature_channels = decoder_seq_feature_channels,
                                                building_block=building_block,
                                                **encoder_config)
            if return_encoder:
                encoder.point_num = point_num
            if return_decoder:
                decoder.point_num = point_num
        elif data_form == DataForm.VEC:
            logger.info('Model is constructed for points.')
        elif data_form == DataForm.GRAPH:
            logger.info('Model is constructed for graph.')
            ## for graph, in_channel and out channel indicates pos_dim by default.
            feature_in_channel = encoder_config.pop('feature_in_channel', 0)
            feature_out_channel = encoder_config.pop('feature_out_channel', None) or feature_in_channel
            message_passing_channels = encoder_config.pop('message_passing_channels', [64, 128, 256])
            assert isinstance(message_passing_channels, list), 'message_passing_channels should be a list.'
            node_num = encoder_config.pop('node_num', 2048)
            if return_encoder:
                encoder = Encoder(pos_dim=in_channel, 
                                    node_dim = feature_in_channel,
                                    message_passing_channels = message_passing_channels,
                                    dense_channels=dense_channels, 
                                    building_block=building_block,
                                    **encoder_config)
                latent_channel = encoder.out_channel
                encoder.node_num = node_num
            if return_decoder:
                decoder = Decoder(pos_dim=out_channel, 
                                    node_dim = feature_out_channel,
                                    node_num = node_num, 
                                    latent_channel=latent_channel, 
                                    dense_channels=de_dense_channels,
                                    **encoder_config)
        else:
            raise NotImplementedError
        ## set data_form
        if return_encoder:
            encoder.data_form = data_form
            encoder.channel_dim = 1 if data_form == DataForm.IMG else -1
            encoder.feature_channel_dim = 1 if building_block in ['single', 'conv', 'double', 'double_conv', 'res', 'res_conv']\
                else -1
        if return_decoder:
            decoder.data_form = data_form
            decoder.channel_dim = 1 if data_form == DataForm.IMG else -1
            decoder.feature_channel_dim = 1 if building_block in ['single', 'conv', 'double', 'double_conv', 'res', 'res_conv']\
                else -1
        return encoder, decoder
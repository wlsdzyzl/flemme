### codes related to computing embedding of image or categories
import torch
import torch.nn as nn
from flemme.encoder import create_encoder, supported_encoders
from flemme.block import OneHotEmbeddingBlock, PositionEmbeddingBlock, new_add, new_cat
from flemme.logger import get_logger
logger = get_logger('model.embedding')

def get_embedding(emb_config):
    ##### construct condition encoder
    # note that condition encoder can be category (classification) or image (label of segmentation or raw data)
    # for category, we can use one-hot embedding;
    # for image, we can use unet, cnn or clip to compute embedding.
    if emb_config is None:
        return None
    emb_name = emb_config.get('name')

    ### wait for image embedding and point embedding.
    if emb_name in supported_encoders:
        logger.info('Using encoder\'{}\' to compute embedding'.format(emb_name))    
        return create_encoder(emb_config, return_decoder=False)[0]
    else:
        ### activation function for building block
        activation = emb_config.get('activation', 'relu')     
        out_channel = emb_config.get('out_channel', None)
        assert out_channel is not None, "Out channel of embedding should be specified."
        if emb_name == 'Identity':
            emb = nn.Identity()   
            logger.info('Using original input for embedding')         
        elif emb_name == 'OneHot':
            logger.info('Using one-hot embedding, input should be categories.')         
            ## use one-hot embedding for categories
            num_classes = emb_config.get('num_classes')
            apply_onehot = emb_config.get('apply_onehot', False)
            emb = OneHotEmbeddingBlock(num_classes=num_classes, out_channel=out_channel,
                                                    activation=activation, apply_onehot=apply_onehot)
        elif emb_name == "Position":
            logger.info('Using Sinusoidal position (time_step) embedding.')    
            emb = PositionEmbeddingBlock(out_channel=out_channel, activation=activation)
        else:
            logger.error('Unsupported data type or embedding method.')
            exit(1)
        emb.out_channel = out_channel
        return emb

### add is one way to merge x and embedding.
### another way is concat x and embedding, which needs they have the same dimension.
def add_embedding(x, c_emb, channel_dim = 1):
    #### merge x and c_emb
    # channel should be the same
    ### x can be a tuple of data and time
    if isinstance(x, tuple) or isinstance(x, list):
        ### return the tuple
        return type(x)((new_add(x[0], c_emb, channel_dim), ) )  + x[1:]
    return new_add(x, c_emb, channel_dim = channel_dim)
### concat over the channel dimension.
def concat_embedding(x, c_emb, channel_dim = 1):
    #### merge x and c_emb
    # channel should be the same
    ### x can be a tuple of data and time
    if isinstance(x, tuple) or isinstance(x, list):
        ### return the tuple
        return type(x)((new_cat(x[0], c_emb, channel_dim), ) )  + x[1:]
    return new_cat(x, c_emb, channel_dim = channel_dim)

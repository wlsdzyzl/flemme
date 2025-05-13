### codes related to computing embedding of image or categories
import torch
import torch.nn as nn
import torch.nn.functional as F
from flemme.encoder import create_encoder, supported_encoders
from flemme.block import OneHotEmbeddingBlock, TimeEmbeddingBlock, expand_as
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
    # print(emb_name, supported_encoders)
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
            apply_onehot = emb_config.get('apply_onehot', True)
            emb = OneHotEmbeddingBlock(num_classes=num_classes, out_channel=out_channel,
                                                    activation=activation, apply_onehot=apply_onehot)
        elif emb_name == "Time":
            logger.info('Using time step embedding, input should be time step or single value position.')    
            emb = TimeEmbeddingBlock(out_channel=out_channel, activation=activation)
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
    if x is None: return c_emb
    if c_emb is None: return x
    if isinstance(x, tuple) or isinstance(x, list):
        ### return the tuple
        return type(x)((add_embedding(x[0], c_emb, channel_dim), ) )  + x[1:]
    assert c_emb.shape[0] == x.shape[0], \
        "Batch size inconsistency."
    assert c_emb.shape[channel_dim] == x.shape[channel_dim], \
        "Number of channels of x and condition encoder should be the same, get {} and {}".format(x.shape[1], c_emb.shape[1])
    if len(c_emb.shape) == len(x.shape):
        if len(x.shape) > 2 and c_emb.shape != x.shape:
            c_emb = F.interpolate(c_emb, size = x.shape)
    elif len(c_emb.shape) == 2:
        c_emb = expand_as(c_emb, x, channel_dim)
    else:
        logger.error('Unknow embedding.')
        raise NotImplementedError
    return x + c_emb
### concat over the channel dimension.
def concat_embedding(x, c_emb, channel_dim = 1):
    #### merge x and c_emb
    # channel should be the same
    ### x can be a tuple of data and time
    if x is None: return c_emb
    if c_emb is None: return x
    if isinstance(x, tuple) or isinstance(x, list):
        ### return the tuple
        return type(x)((concat_embedding(x[0], c_emb, channel_dim), ) )  + x[1:]

    assert c_emb.shape[0] == x.shape[0], \
        "Batch size inconsistency."
    ## for concat, we don't need the number of channels to be the same.
    if len(c_emb.shape) == len(x.shape):
        if len(x.shape) > 2 and c_emb.shape != x.shape:
            c_emb = F.interpolate(c_emb, size = x.shape)
    elif len(c_emb.shape) == 2:
        c_emb = expand_as(c_emb, x, channel_dim)
    else:
        logger.error('Unknow embedding.')
        raise NotImplementedError
    return torch.concat([x, c_emb], dim = channel_dim)

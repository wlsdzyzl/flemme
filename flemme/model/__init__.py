from flemme.model.base import BaseModel as Base, HBaseModel as HBase
from flemme.model.sem import SegmentationModel as SeM, HSegmentationModel as HSeM
from flemme.model.ae import AutoEncoder as AE, HAutoEncoder as HAE
from flemme.model.vae import VariationalAutoEncoder as VAE
from flemme.model.ddpm import DiffusionProbabilistic as DDPM
from flemme.model.ddim import DiffusionImplicit as DDIM
from flemme.model.edm import EDM
from flemme.model.ldm import LatentDiffusion as LDM, supported_ae_models, supported_diff_models
from flemme.model.sdm import SupervisedDiffusion as SDM
from flemme.model.half import OnlyDecoder, OnlyEncoder
from flemme.model.clm import ClassificationModel as ClM
from flemme.utils import load_config
from flemme.logger import get_logger
from flemme.encoder import create_encoder
logger = get_logger('model.create_model')
supported_models = {
    #### base model
    'Base': Base,
    #### h-base model
    'HBase': HBase,
    #### classification model: encoder with classification loss
    'ClM': ClM,
    #### segmentation model: base model with segmentation loss
    'SeM': SeM,
    #### h-base model with segmentation loss
    'HSeM': HSeM,
    #### auto-encoder: base model with reconstruction loss
    'AE': AE,
    #### h-auto-encoder: h-base model with reconstruction loss
    'HAE': HAE,
    #### variational auto-encoder: kl-regularized auto encoder (with KL loss)
    'VAE': VAE,
    #### diffusion model
    'DDPM': DDPM,
    #### diffusion implicit model
    'DDIM': DDIM,
    ### edm: Elucidating the Design Space of Diffusion-Based Generative Models
    'EDM': EDM,
    #### latent diffusion model: diffusion model with a pre-trained auto-encoder
    'LDM': LDM,
    #### supervised diffusion model: use for reconstruction or segmentation
    'SDM': SDM,
    #### model with only encoder
    'OnlyEncoder': OnlyEncoder,
    #### model with only decoder
    'OnlyDecoder': OnlyDecoder
}
def create_model(model_config, 
    supported_models = supported_models, 
    create_encoder_fn = create_encoder):
    tmpl_path = model_config.pop('template_path', None)
    if tmpl_path is not None:
        logger.info('creating model from template ...')
        model_config = load_config(tmpl_path).get('model')
        return create_model(model_config, 
                        supported_models, 
                        create_encoder_fn)
    
    logger.info('creating model from specific configuration ...')
    
    model_name = model_config.pop('name', 'Base')
    model_class = None
    if model_name in supported_models:
        model_class = supported_models[model_name]
    else:
        logger.error(f'Unsupported model class: {model_name}, should be one of {supported_models.keys()}')
        exit(1)
    return model_class(model_config, create_encoder_fn = create_encoder_fn)
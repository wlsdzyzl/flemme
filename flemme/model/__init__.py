from flemme.model.base import BaseModel as Base, HBaseModel as HBase
from flemme.model.sem import SegmentationModel as SeM, HSegmentationModel as HSeM
from flemme.model.ae import AutoEncoder as AE, HAutoEncoder as HAE
from flemme.model.vae import VariationalAutoEncoder as VAE
from flemme.model.ddpm import DiffusionProbabilistic as DDPM, supported_eps_models
from flemme.model.ddim import DiffusionImplicit as DDIM
from flemme.model.ldm import LatentDiffusionProbabilistic as LDPM, LatentDiffusionImplicit as LDIM, supported_ae_models
from flemme.model.sdm import SupervisedDiffusionProbabilistic as SDPM, SupervisedDiffusionImplicit as SDIM
from flemme.model.half import OnlyDecoder, OnlyEncoder
from flemme.model.clm import ClassificationModel as ClM
from flemme.utils import load_config
from flemme.logger import get_logger
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
    #### latent diffusion model: diffusion model with a pre-trained auto-encoder
    'LDPM': LDPM,
    #### latent diffusion implicit model
    'LDIM': LDIM,
    #### supervised diffusion model: use for reconstruction or segmentation
    'SDPM': SDPM,
    #### supervised diffusion implicit model
    'SDIM': SDIM,
    #### model with only encoder
    'OnlyEncoder': OnlyEncoder,
    #### model with only decoder
    'OnlyDecoder': OnlyDecoder
}
def create_model(model_config):
    tmpl_path = model_config.pop('template_path', None)
    if tmpl_path is not None:
        logger.info('creating model from template ...')
        model_config = load_config(tmpl_path).get('model')
        return create_model(model_config)
    
    logger.info('creating model from specific configuration ...')
    
    model_name = model_config.pop('name', 'Base')
    model_class = None
    if model_name in supported_models:
        model_class = supported_models[model_name]
    else:
        logger.error(f'Unsupported model class: {model_name}')
        exit(1)
    return model_class(model_config)
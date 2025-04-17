# auto-encoder for 2D image and 3D point cloud
from flemme.model.base import BaseModel, HBaseModel
from flemme.loss import get_loss
from flemme.logger import get_logger
import torch.nn.functional as F
logger = get_logger('model.ae')
class AutoEncoder(BaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        # assert self.data_form == DataForm.IMG and self.encoder.image_channel == self.decoder.image_channel or \
        #     (self.data_form == DataForm.PCD or self.data_form == DataForm.VEC) and self.encoder.point_dim == self.decoder.point_dim,\
        #     "For auto-encoder, the number of input image channels (point dim) and output image channels (point dim) should be the same."
        self.recon_losses = []
        self.recon_loss_names = []
        self.recon_loss_weights = []
        recon_loss_configs = model_config.get('reconstruction_losses', [{'name':'MSE'}])
        if not type(recon_loss_configs) == list:
            recon_loss_configs = [recon_loss_configs, ]
        for loss_config in recon_loss_configs:
            loss_config['reduction'] = self.loss_reduction
            self.recon_loss_names.append(loss_config.get('name'))
            self.recon_loss_weights.append(loss_config.pop('weight', 1.0))
            self.recon_losses.append(get_loss(loss_config, self.data_form))
        self.is_supervised = model_config.get('is_supervised', False)
        if self.is_supervised:
            logger.info('Reconstruction under supervision.')
        
    def __str__(self):
        _str = '********************* Auto-Encoder ({} - {}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.decoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def get_loss_name(self):
        return self.recon_loss_names
    def encode(self, x, c=None):
        return super().encode(x, c = c)
    def decode(self, z, c = None):
        return super().decode(z, c = c)
    def forward(self, x, c=None):
        res, z = super().forward(x, c = c, return_z = True)
        ### unet
        if type(z) == tuple:
            z = z[0]
        return {'recon':res, 'latent':z}
    def compute_loss(self, x, c = None, res = None, y = None):
        if res is None:
            res = self.forward(x, c)   
        losses = []
        target = y if y is not None else x
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            losses.append(loss(res['recon'], target) * weight) 
        return losses, res


class HAutoEncoder(HBaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        # assert self.data_form == DataForm.IMG and self.encoder.image_channel == self.decoder.image_channel or \
        #     (self.data_form == DataForm.PCD or self.data_form == DataForm.VEC) and self.encoder.point_dim == self.decoder.point_dim,\
        #     "For auto-encoder, the number of input image channels (point dim) and output image channels (point dim) should be the same."
        self.recon_losses = []
        self.recon_loss_names = []
        self.recon_loss_weights = []
        recon_loss_configs = model_config.get('reconstruction_losses', [{'name':'MSE'}])
        if not type(recon_loss_configs) == list:
            recon_loss_configs = [recon_loss_configs, ]
        for loss_config in recon_loss_configs:
            loss_config['reduction'] = self.loss_reduction
            self.recon_loss_names.append(loss_config.get('name'))
            self.recon_loss_weights.append(loss_config.pop('weight', 1.0))
            self.recon_losses.append(get_loss(loss_config, self.data_form))
        self.is_supervised = model_config.get('is_supervised', False)
        
    def __str__(self):
        _str = '********************* H-Auto-Encoder ({} - {}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.decoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def get_loss_name(self):
        return self.recon_loss_names + ['hierarchical_' + s for s in self.recon_loss_names]
    def encode(self, x, c=None):
        return super().encode(x, c = c)
    def decode(self, z, c = None):
        return super().decode(z, c = c)
    def forward(self, x, c=None):
        (res, h_res), z = super().forward(x, c = c, return_z = True)
        if type(z) == tuple:
            z = z[0]
        return {'recon':res, 'hierarchical_recon':h_res, 'latent':z}
    def compute_loss(self, x, c = None, res = None, y = None):
        if res is None:
            res = self.forward(x, c)   
        losses = []
        target = y if y is not None else x
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            losses.append(loss(res['recon'], target) * weight) 
        ### loss for training better intermediate features

        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            sublosses = []
            for h_x in res['hierarchical_recon']:
                h_target = F.interpolate(target, size = h_x.shape[2:], mode = self.inter_mode)
                sublosses.append(loss(h_x, h_target) * weight)
            losses.append(sum(sublosses) / len(sublosses) ) 
        return losses, res
# variational autoencoder for 2D image and 3D point cloud
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ae import AutoEncoder
from flemme.loss import get_loss
from flemme.model.distribution import GaussianDistribution as Gaussian
from flemme.logger import get_logger
from flemme.utils import DataForm
from flemme.block import channel_recover, channel_transfer
logger = get_logger('model.vae')
class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, model_config):
        super().__init__(model_config)
        if self.with_time_embedding:
            raise NotImplementedError("Time embedding is not implemented for VAE model.")
        assert not self.decoder_name in ['UNet', 'ViTU', 'SwinU', 'MambaU'], \
            'UNet are not suitable for constructing a VAE.'
        ## for point cloud, pointnet2 could be used just for feature extraction without skip connection
        assert not self.decoder_name in ['PointNet2', 'PointTrans2', 'PointMamba2'], \
            'Point2Decoder are not suitable for constructing a VAE.'
        distr_loss_config = model_config.get('distribution_loss', {'name':'KL'})
        distr_loss_config['reduction'] = self.loss_reduction
        self.distr_loss_name = distr_loss_config.get('name')
        self.distr_loss_weight = distr_loss_config.pop('weight', 1.0)
        self.distr_loss = get_loss(distr_loss_config, self.data_form)
        self.is_generative = True
        latent_channel = self.encoder.out_channel
        self.mean_layer = nn.Linear(latent_channel, latent_channel,)
        self.logvar_layer = nn.Linear(latent_channel, latent_channel,)
    def __str__(self):
        _str = '********************* Variational Auto-Encoder ({} - {}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.decoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def encode(self, x, c=None):
        try:
            z = super().encode(x, c = c)
            if type(z) == tuple:
                z = z[0]
            if self.feature_channel_dim == 1:
                z = channel_transfer(z)
            mean, logvar = self.mean_layer(z), self.logvar_layer(z)
            if self.feature_channel_dim == 1:
                mean, logvar = channel_recover(mean), channel_recover(logvar)
            gauss = Gaussian(mean = mean, logvar = logvar)
        except Exception as e:
            logger.error(f'Parsing mean and logvar failed: {e}')
            exit(1)
        return gauss
    def decode(self, z, c=None):
        return super().decode(z, c = c)
    def forward(self, x, c = None):
        gaussian = self.encode(x, c)
        z = gaussian.sample()
        # print(z.shape, c.shape)
        # sample from the mean (mean) and log-variance (logvar)
        # reparameterization
        return {"recon":self.decode(z, c), "gaussian":gaussian, 'latent':z} 
    def get_loss_name(self):
         return [self.distr_loss_name, ] + self.recon_loss_names

    def compute_loss(self, x, c = None, res = None, y = None):
        if res is None:
            res = self.forward(x, c)   
        ## compute the KL Divergence between N(mean, var) and N(0, 1)
        distr_loss = self.distr_loss(res['gaussian']) * self.distr_loss_weight
        ## compute the reconstruction loss
        recon_losses = []
        target = y if y is not None else x
        for loss, weight in zip(self.recon_losses, self.recon_loss_weights):
            recon_losses.append(loss(res['recon'], target) * weight) 

        return [distr_loss,] + recon_losses, res
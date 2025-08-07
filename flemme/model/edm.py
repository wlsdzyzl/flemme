#### ddpm of 2D image and 3D point cloud
# part of the code is adopted from https://github.com/1zb/3DShape2VecSet
#  techniques are described in the paper "Elucidating the Design Space of Diffusion-Based Generative Models"
import torch
import torch.nn as nn
import torch.nn.functional as F
from flemme.model.base import HBaseModel as HBase
from flemme.loss import get_loss
from flemme.logger import get_logger
from flemme.model.ddpm import _create_eps_model

logger = get_logger('model.edm')


class EDM(nn.Module):

    def __init__(self, model_config, create_encoder_func):
        super().__init__()
        self.loss_reduction = model_config.get('loss_reduction', 'mean')
        eps_config = model_config.get('eps_model')
        eps_config['loss_reduction'] = self.loss_reduction
        self.eps_model, self.eps_model_name = \
            _create_eps_model(eps_config, create_encoder_func)
        self.num_steps = model_config.get('num_steps', 20)

        self.sigma_min = model_config.get('sigma_min', 0.002)
        self.sigma_max = model_config.get('sigma_max', 80)
        self.sigma_data = model_config.get('sigma_data', 0.5)
        self.sigma_shape = (-1, ) + (1, ) * (len(self.eps_model.get_input_shape()))
        self.rho = model_config.get('rho', 7)
        self.S_churn = model_config.get('S_churn', 0)
        self.S_min = model_config.get('S_min', 0)
        self.S_max = model_config.get('S_max', float('inf'))
        self.S_noise = model_config.get('S_noise', 1)
        self.P_std = model_config.get('P_std', 1.2)
        self.P_mean = model_config.get('P_mean', -1.2)

        self.is_conditional = self.eps_model.is_conditional
        self.is_generative = True
        self.is_supervised = False
        self.data_form = self.eps_model.data_form

        # self.eps_model, self.eps_model_name = \
        #     self.__create_eps_model(eps_config)
        eps_loss_config = model_config.get('eps_loss', {'name':'MSE'})
        eps_loss_config['reduction'] = self.loss_reduction
        self.eps_loss_name = eps_loss_config.get('name')
        self.eps_loss = get_loss(eps_loss_config, self.data_form)
        ### recon losses
        self.recon_losses = []
        self.recon_loss_names = []
        self.recon_loss_weights = []
        recon_loss_configs = model_config.pop('reconstruction_losses', [])
        if not type(recon_loss_configs) == list:
            recon_loss_configs = [recon_loss_configs,] 
        if len(recon_loss_configs) > 0:
            for loss_config in recon_loss_configs:
                loss_config['reduction'] = self.loss_reduction
                self.recon_loss_names.append(loss_config.get('name'))
                self.recon_loss_weights.append(loss_config.pop('weight', 1.0))
                self.recon_losses.append(get_loss(loss_config, self.data_form))

    @property
    def device(self):
        return self.eps_model.device

    # edm_sampler
    @torch.no_grad()
    def sample(self,
        latents, c = None, 
        num_steps=None, 
        sigma_min=None, sigma_max=None, rho=None,
        # S_churn=40, S_min=0.05, S_max=50, S_noise=1.003,
        S_churn=None, S_min=None, S_max=None, S_noise=None,
        return_processing = False
    ):
        ### by default we use the model parameter
        ### but keep a more flexible calling for developer
        num_steps = num_steps or self.num_steps
        sigma_min = sigma_min or self.sigma_min
        sigma_max = sigma_max or self.sigma_max
        rho = rho or self.rho
        S_churn = S_churn or self.S_churn
        S_min = S_min or self.S_min
        S_max = S_max or self.S_max
        S_noise = S_noise or self.S_noise
        # Time step discretization.
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
        t_steps = (sigma_max ** (1 / self.rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / self.rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        # Main sampling loop.
        x_next = latents.to(torch.float64) * t_steps[0]
        if return_processing:
            processing = [x_next.float32()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            # sqrt(2) - 1
            gamma = min(S_churn / num_steps, 0.41421) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.denoise(x_hat, t_hat, c).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(x_next, t_next, c).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            if return_processing:
                processing.append(x_next.float32())
        if return_processing:
            return processing
        return x_next

    def denoise(self, x, sigma, cond):
                
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(self.sigma_shape)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.eps_model((c_in * x).to(dtype), t = c_noise.flatten(), c = cond)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    def get_latent_shape(self):
        return self.eps_model.get_input_shape()
    def get_input_shape(self):
        return self.eps_model.get_input_shape()
    def get_output_shape(self):
        return self.eps_model.get_output_shape()
    def get_loss_name(self):
        if isinstance(self.eps_model, HBase):
            return [self.eps_loss_name, 'hierarchical_' + self.eps_loss_name] +\
                self.recon_loss_names + [ 'hierarchical_' + rln for rln in self.recon_loss_names]
        return [self.eps_loss_name,] + self.recon_loss_names
    def __str__(self):
        _str = "********************* ElucidatedDiffusion *********************\n{}"\
            .format(self.eps_model.__str__())
        return _str
    def compute_loss(self, x0, c = None, res = None):
        if res is None:
            res = self.forward(x0, c)
        model_out, sample_weight = res['recon_dpm'], res['weight_dpm']
        losses = []
        if not type(model_out) == tuple:
            losses += [self.eps_loss(model_out, x0, sample_weight = sample_weight), ]
        else:
            #### h-base model
            loss = self.eps_loss(model_out[0], x0)
            sublosses = []
            for h_x in model_out[1]:
                h_target = F.interpolate(x0, size = h_x.shape[2:], mode = self.eps_model.inter_mode)
                sublosses.append(self.eps_loss(h_x, h_target, sample_weight = sample_weight))
            losses += [loss, sum(sublosses) / len(sublosses)]

        if len(self.recon_losses) > 0:
            if type(model_out) == tuple:
                final_output = model_out[0]
            else:
                final_output = model_out

            for l, w in zip(self.recon_losses, self.recon_loss_weights):
                losses += [l(final_output, x0, sample_weight = sample_weight) * w, ]
            
            ## h-base eps_model
            if type(model_out) == tuple:                
                for l, w in zip(self.recon_losses, self.recon_loss_weights):
                    sublosses = []
                    for h_mo in model_out[1]:
                        h_x0 = F.interpolate(x0, size = h_mo.shape[2:], mode = self.eps_model.inter_mode)
                        sublosses += [l(h_mo, h_x0, sample_weight = sample_weight) * w, ]
                    losses.append(sum(sublosses) / len(sublosses))
        return losses, res

    def forward(self, x, c = None):
        rnd_normal = torch.randn([x.shape[0]], device=x.device)
        # rnd_normal[]
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x) * sigma.reshape(self.sigma_shape)
        ### sigma is time step, c is condition
        D_yn = self.denoise(x + n, sigma, c)
        return {'recon_dpm': D_yn, 'weight_dpm': weight}
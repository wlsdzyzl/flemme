#### ddpm of 2D image and 3D point cloud
# part of the code is adopted from https://nn.labml.ai/diffusion/ddpm/index.html
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flemme.model.base import BaseModel as Base, HBaseModel as HBase
from flemme.loss import get_loss
# from flemme.utils import DataForm
from flemme.logger import get_logger
from flemme.model.distribution import GaussianDistribution as Gaussian

logger = get_logger('model.ddpm')
def gather(consts, t, dim = 2):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    shape = (-1,  1) + tuple( 1 for _ in range(dim))
    return c.reshape(*shape)

def linear_schedule(num_steps, scale = 1.0):
    # Linear schedule from Ho et al, extended to work for any number of
    # diffusion steps.
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, num_steps)
def cosine_schedule(num_steps, max_beta = 0.999):
    # cosine schedule from Alex et al
    # max_beta shouldn't be used, unless there are some numerical problems 
    cos = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    beta = []
    for i in range(num_steps):
        # cos(t2) / cos(t1) is decreasing by time steps
        t1 = i / num_steps
        t2 = (i + 1) / num_steps
        beta.append( min(1 - cos(t2) / cos(t1), max_beta) )
    return torch.Tensor(beta)

supported_eps_models = {'Base': Base, 'HBase': HBase}

def _create_eps_model(eps_config, create_encoder_func):
    model_name = eps_config.pop('name', 'Base')
    if not model_name in supported_eps_models:
        raise RuntimeError(f'Unsupported model class: {model_name}')
    eps_model = supported_eps_models[model_name](eps_config, create_encoder_func)
    assert eps_model.with_time_embedding, \
        'You need a encoder with time embedding for ddpm.'
    return eps_model, model_name

class DiffusionProbabilistic(nn.Module):
    
    def __init__(self, model_config, create_encoder_func):
        super().__init__()
        self.loss_reduction = model_config.get('loss_reduction', 'mean')
        # noise predictor should be a base model with time embedding.
        eps_config = model_config.get('eps_model')
        eps_config['loss_reduction'] = self.loss_reduction
        self.eps_model, self.eps_model_name = \
            _create_eps_model(eps_config, create_encoder_func)
        self.num_steps = model_config.get('num_steps', 1000)
        # Create $\beta_1, \dots, \beta_T$ linearly increasing variance schedule
        self.is_conditional = self.eps_model.is_conditional
        self.is_generative = True
        self.is_supervised = False

        self.data_form = self.eps_model.data_form
        self.beta_schedule = model_config.get('beta_schedule', 'linear')
        ### classifier-free guided ddpm
        classifier_free_config = model_config.get('classifier_free_guidance', None)
        self.classifier_free = classifier_free_config is not None
        if self.classifier_free:
            self.condition_dropout = classifier_free_config.get('condition_dropout', 0.1)
            self.guidance_weight = classifier_free_config.get('guidance_weight', 1.0)
            logger.info('Using classifier-free guidance with {} condition dropout and weight = {}'.format(self.condition_dropout, self.guidance_weight))
            
            if self.eps_model.combine_condition == 'cat':
                logger.error('Diffusion model with classifier-free guidance doesn\'t support concatination of conditions.')
                exit(1)
        if self.beta_schedule == 'linear':
            beta = linear_schedule(self.num_steps)
        else:
            beta = cosine_schedule(self.num_steps)
        self.gather_dim = len(self.eps_model.get_input_shape()) - 1
        eps_loss_config = model_config.get('eps_loss', {'name':'MSE'})
        eps_loss_config['reduction'] = self.loss_reduction
        self.eps_loss_name = eps_loss_config.get('name')
        self.eps_loss = get_loss(eps_loss_config, self.data_form)
        ### these values are not parameter and will not be changed through training
        ### but they will be saved when saving checkpoint and moved to GPU when calling to(device)
        self.register_buffer('beta', beta)
        # $\alpha_t = 1 - \beta_t$
        self.register_buffer('alpha', 1 - beta)
        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        self.register_buffer('alpha_bar', torch.cumprod(1 - beta, dim=0))
        self.register_buffer('sqrt_alpha_bar', torch.cumprod(1 - beta, dim=0) ** .5)
        self.register_buffer('sqrt_one_minus_alpha_bar', (1 - torch.cumprod(1 - beta, dim=0)) ** .5)
        # $\sigma^2 = \beta$
        self.register_buffer('sigma2', beta)
        self.clipped = model_config.get('clipped', False)
        self.clip_range = model_config.get('clip_range', [-1.0, 1.0])
        self.parameterization = model_config.get('parameterization', 'epsilon')
        assert self.parameterization in ['epsilon', 'velocity', 'start'], \
            'Parameterization should be one of ["epsilon", "velocity", "start"]'
        logger.info(f'Using {self.parameterization} parameterization for diffusion model.')
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
    def add_noise(self, x0, t):
        """
        #### Sample from $q(x_t|x_0)$

        \begin{align}
        q(x_t|x_0) &= \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)
        \end{align}
        """
        # [gather](utils.html) $\alpha_t$ and compute $\sqrt{\bar\alpha_t} x_0$
        mean = gather(self.sqrt_alpha_bar, t, self.gather_dim) * x0
        # $(1-\bar\alpha_t) \mathbf{I}$
        var = 1 - gather(self.alpha_bar, t, self.gather_dim)
        
        # reparameterization
        return Gaussian(mean, var = var).sample(True)
        # eps = torch.randn_like(x0)
        # return mean + (var ** 0.5) * eps, eps 
    def __str__(self):
        _str = "********************* DiffusionProbabilistic *********************\n{}"\
            .format(self.eps_model.__str__())
        return _str
    ## from stable diffusion
    def get_velocity(self, x0, t, eps):
        return gather(self.sqrt_alpha_bar, t, self.gather_dim) * eps - \
            gather(self.sqrt_one_minus_alpha_bar, t, self.gather_dim) * x0
    
    def get_eps_from_velocity(self, xt, t, v):

        return gather(self.sqrt_alpha_bar, t, self.gather_dim) * v +\
             gather(self.sqrt_one_minus_alpha_bar, t, self.gather_dim) * xt
    
    def get_eps_from_start(self, xt, t, s):
        return (xt - gather(self.sqrt_alpha_bar, t, self.gather_dim) * s ) / \
            gather(self.sqrt_one_minus_alpha_bar, t, self.gather_dim)

    def get_start_from_eps(self, xt, t, eps):
        ## x0 will be clipped to [-1, 1]
        x0_pred = 1 / gather(self.sqrt_alpha_bar, t, self.gather_dim) * \
            (xt - gather(self.sqrt_one_minus_alpha_bar, t, self.gather_dim) * eps)
        return x0_pred

    def get_start_from_velocity(self, xt, t, v):
        return gather(self.sqrt_alpha_bar, t, self.gather_dim)*xt - \
            gather(self.sqrt_one_minus_alpha_bar, t, self.gather_dim)*v
    
    def get_eps_from_model(self, xt, t, c):
        if self.eps_model.is_conditional and c is not None:
            model_out = self.eps_model(xt, t, c)
            if type(model_out) == tuple:
                model_out = model_out[0]
            if self.classifier_free:
                cf_model_out = self.eps_model(xt, t)
                if type(cf_model_out) == tuple:
                    cf_model_out = cf_model_out[0]
                model_out = (1.0 + self.guidance_weight) * model_out \
                    - self.guidance_weight * cf_model_out              
        else:
            model_out = self.eps_model(xt, t)
            if type(model_out) == tuple:
                model_out = model_out[0]

        if self.parameterization == 'velocity':
            eps_theta = self.get_eps_from_velocity(xt, t, model_out)
        elif self.parameterization == 'start':
            eps_theta = self.get_eps_from_start(xt, t, model_out)
        else:
            eps_theta = model_out
        return eps_theta
    @torch.no_grad()
    def denoise(self, xt, t, c = None, clipped = None, clip_range = None):
        """
        #### Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$

        \begin{align}
        \textcolor{lightgreen}{p_\theta}(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1};
        \textcolor{lightgreen}{\mu_\theta}(x_t, t), \sigma_t^2 \mathbf{I} \big) \\
        \textcolor{lightgreen}{\mu_\theta}(x_t, t)
          &= \frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)
        \end{align}
        """
        # $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
        ## classifier-free
        ## see: https://openreview.net/pdf?id=qw8AKxfYbI
        eps_theta = self.get_eps_from_model(xt, t, c)
        # [gather](utils.html) $\bar\alpha_t$
        alpha_bar = gather(self.alpha_bar, t, self.gather_dim)
        # $\alpha_t$
        alpha = gather(self.alpha, t, self.gather_dim)
        var = gather(self.sigma2, t, self.gather_dim)
        sqrt_alpha_bar = gather(self.sqrt_alpha_bar, t, self.gather_dim)
        sqrt_one_minus_alpha_bar = gather(self.sqrt_one_minus_alpha_bar, t, self.gather_dim)

        if clipped:
            ## t is a tensor, however, all the values should be the same.
            if t.min() > 0:
                alpha_bar_pre = gather(self.alpha_bar, t-1, self.gather_dim)
            else:
                alpha_bar_pre = 1.0
            ## x0 will be clipped to [-1, 1]
            x0_pred = (xt - sqrt_one_minus_alpha_bar * eps_theta) / sqrt_alpha_bar 
            x0_pred = x0_pred.clamp(*clip_range)
            beta = gather(self.beta, t, self.gather_dim)
            mean = (alpha ** 0.5 * (1 - alpha_bar_pre)  * xt + 
                alpha_bar_pre ** 0.5 * beta * x0_pred) / (1 - alpha_bar) 
        else:
            # $\frac{\beta}{\sqrt{1-\bar\alpha_t}}$
            eps_coef = (1 - alpha) / sqrt_one_minus_alpha_bar
            # $$\frac{1}{\sqrt{\alpha_t}} \Big(x_t -
            #      \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\textcolor{lightgreen}{\epsilon_\theta}(x_t, t) \Big)$$
            mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
            # $\sigma^2$
            
            # reparameterization
        return Gaussian(mean, var = var).sample()
        # return mean + (var ** .5) * torch.randn_like(xt)

    @torch.no_grad()
    def sample(self, xt, end_step = 100, c = None, clipped = None, 
               clip_range = None, return_processing = False):
        if clipped is None:
            clipped = self.clipped
        if clip_range is None:
            clip_range = self.clip_range
        if end_step < 0:
            end_step = self.num_steps-1
        ## recover x0 from a noise x
        batch_size = xt.shape[0]
        if return_processing:
            processing = [xt.clone()]
        for t in range(end_step, -1, -1):
            ### batch time step
            bt = torch.ones((batch_size,), device=xt.device, dtype=torch.long) * t
            xt = self.denoise(xt, bt, c, clipped = clipped, clip_range=clip_range)
            if return_processing:
                processing.append(xt.clone())

        if return_processing:
            return processing
        return xt
    
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
    def forward(self, x0, c = None):
        # Get batch size
        batch_size = x0.shape[0]
        # Get random $t$ for each sample in the batch
        t = torch.randint(0, self.num_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # Sample $x_t$ for $q(x_t|x_0)$
        xt, eps = self.add_noise(x0, t)
        # Get $\textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)$
        if not self.eps_model.is_conditional or \
            self.classifier_free and torch.rand(1).item() < self.condition_dropout:
            c = None
        model_out = self.eps_model(xt, t = t, c = c) 
        # hierarchical base model
        h_model_out = None
        
        if type(model_out) == tuple:
            model_out, h_model_out = model_out

        if self.parameterization == 'epsilon':
            x0_pred = self.get_start_from_eps(xt = xt, t = t, eps = model_out)
        elif self.parameterization == 'velocity':
            x0_pred = self.get_start_from_velocity(xt = xt, t = t, v = model_out)
        else:
            x0_pred = model_out
        res = {'recon_dpm':x0_pred,
               'eps': eps,
               'xt': xt,
               't': t,
               'model_out': model_out,
               'h_model_out': h_model_out}
        return res
    def compute_loss(self, x0, c = None, res = None):
        """
        #### Simplified Loss

        $$L_{\text{simple}}(\theta) = \mathbb{E}_{t,x_0, \epsilon} \Bigg[ \bigg\Vert
        \epsilon - \textcolor{lightgreen}{\epsilon_\theta}(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon, t)
        \bigg\Vert^2 \Bigg]$$
        """
        if res is None:
            res = self.forward(x0, c = c)
        model_out = res['model_out']
        h_model_out = res['h_model_out']
        t = res['t']
        eps = res['eps']
        
        ### eps loss
        if self.parameterization == 'epsilon':
            target = eps
        elif self.parameterization == 'start':
            target = x0
        else:
            target = self.get_velocity(x0, t, eps)

        losses = []
        losses += [self.eps_loss(model_out, target), ]
        if h_model_out:
            sublosses = []
            for h_x in h_model_out:
                h_target = F.interpolate(target, size = h_x.shape[2:], mode = self.eps_model.inter_mode)
                sublosses.append(self.eps_loss(h_x, h_target))
            losses += [sum(sublosses) / len(sublosses), ]

        
        ### recon loss
        if len(self.recon_losses) > 0:
            x0_pred = res['recon_dpm']
            xt = res['xt']
            for l, w in zip(self.recon_losses, self.recon_loss_weights):
                losses += [l(x0_pred, x0) * w, ]
            
            ## h-base eps_model
            if h_model_out:            
                for l, w in zip(self.recon_losses, self.recon_loss_weights):
                    sublosses = []
                    for h_mo in h_model_out:
                        h_x0 = F.interpolate(x0, size = h_mo.shape[2:], mode = self.eps_model.inter_mode)
                        if self.parameterization == 'epsilon':
                            h_x0_pred = self.get_start_from_eps(xt = xt, t = t, eps = h_mo)
                        elif self.parameterization == 'velocity':
                            h_x0_pred = self.get_start_from_velocity(xt = xt, t = t, v = h_mo)
                        else:
                            h_x0_pred = h_mo
                        sublosses += [l(h_x0_pred, h_x0) * w, ]
                    losses.append(sum(sublosses) / len(sublosses))
        return losses, res
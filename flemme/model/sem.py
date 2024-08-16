import torch
import torch.nn as nn
import torch.nn.functional as F
from flemme.model.base import BaseModel, HBaseModel
from flemme.loss import get_loss

# segmentation model, don't has condition and time input
class SegmentationModel(BaseModel):
    def __init__(self, model_config):

        super().__init__(model_config)

        seg_loss_configs = model_config.get('segmentation_losses', [{'name':'Dice'}])
        if not type(seg_loss_configs) == list:
            seg_loss_configs = [seg_loss_configs, ]
            
        self.seg_losses = []
        self.seg_loss_names = []
        self.seg_loss_weights = []
        self.is_supervising = True
        for loss_config in seg_loss_configs:
            loss_config['reduction'] = self.loss_reduction
            self.seg_loss_weights.append(loss_config.pop('weight', 1.0))
            self.seg_loss_names.append(loss_config.get('name'))
            self.seg_losses.append(get_loss(loss_config, self.data_form))
        
        
    def __str__(self):
        _str = '********************* SeM ({}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def get_loss_name(self):
        return self.seg_loss_names
    def encode(self, x, c = None):
        return super().encode(x, c = c)
    def decode(self, z, c = None):
        return super().decode(z, c = c)
    def forward(self, x, c = None):
        return {'seg_logits':super().forward(x, c = c)}
    def compute_loss(self, x, y, c = None):
        res = self.forward(x, c = c)
        losses = []
        for loss, weight in zip(self.seg_losses, self.seg_loss_weights):
            losses.append(loss(res['seg_logits'], y) * weight) 
        return losses, res
    
class HSegmentationModel(HBaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        seg_loss_configs = model_config.get('segmentation_losses', [{'name':'Dice'}])
        if not type(seg_loss_configs) == list:
            seg_loss_configs = [seg_loss_configs, ]
            
        self.seg_losses = []
        self.seg_loss_names = []
        self.seg_loss_weights = []
        self.is_supervising = True
        for loss_config in seg_loss_configs:
            loss_config['reduction'] = self.loss_reduction
            self.seg_loss_weights.append(loss_config.pop('weight', 1.0))
            self.seg_loss_names.append(loss_config.get('name'))
            self.seg_losses.append(get_loss(loss_config, self.data_form))
        
    def __str__(self):
        _str = '********************* HSeM ({}) *********************\n------- Encoder -------\n{}------- Decoder -------\n{}'.format(self.encoder_name, self.encoder.__str__(), self.decoder.__str__())
        return _str
    def get_loss_name(self):
        return self.seg_loss_names + ['hierarchical_' + s for s in self.seg_loss_names]
    def encode(self, x, c = None):
        return super().encode(x, c = c)
    def decode(self, z, c = None):
        return super().decode(z, c = c)
    def forward(self, x, c = None):
        x, h_x = super().forward(x, c = c)
        return {'seg_logits':x, 'hierarchical_logits': h_x}
    
    def compute_loss(self, x, y, c = None):
        res = self.forward(x, c = c)
        losses = []
        ## loss for original image
        for loss, weight in zip(self.seg_losses, self.seg_loss_weights):
            losses.append(loss(res['seg_logits'], y) * weight) 

        ### loss for training better intermediate features
        for loss, weight in zip(self.seg_losses, self.seg_loss_weights):
            sublosses = []
            for h_x in res['hierarchical_logits']:
                h_y = F.interpolate(y, size = h_x.shape[2:], mode = self.inter_mode)
                sublosses.append(loss(h_x, h_y) * weight)
            losses.append(sum(sublosses) / len(sublosses) ) 
        return losses, res
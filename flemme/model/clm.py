### classification model
import torch
import torch.nn as nn
from flemme.model.half import OnlyEncoder
from flemme.loss import get_loss

# classification model, don't has condition and time input
class ClassificationModel(OnlyEncoder):
    def __init__(self, model_config, create_encoder_fn):
        encoder_config = model_config.pop('encoder')
        self.class_num = encoder_config.get('out_channel', None)
        assert self.class_num is not None and self.class_num > 0, 'Invalid out channel.'
        if not 'vector_embedding' in encoder_config:
            encoder_config['vector_embedding'] = True
        assert encoder_config['vector_embedding'], \
            'Classification model needs vector embeddings (Don\'t set vector_embedding as False).'
        model_config['encoder'] = encoder_config
        super().__init__(model_config, create_encoder_fn)
        cls_loss_configs = model_config.get('classification_losses', [{'name':'CE'}])
        if not type(cls_loss_configs) == list:
            cls_loss_configs = [cls_loss_configs, ]
        self.classifier = nn.Linear(self.encoder.out_channel, self.class_num)
        self.cls_losses = []
        self.cls_loss_names = []
        self.cls_loss_weights = []
        self.is_supervised = True
        for loss_config in cls_loss_configs:
            loss_config['reduction'] = self.loss_reduction
            self.cls_loss_weights.append(loss_config.pop('weight', 1.0))
            self.cls_loss_names.append(loss_config.get('name'))
            self.cls_losses.append(get_loss(loss_config, self.data_form))
        
    def __str__(self):
        _str = '********************* ClM ({}) *********************\n------- Encoder -------\n{}'.format(self.encoder_name, self.encoder.__str__())
        return _str
    def get_loss_name(self):
        return self.cls_loss_names
    def forward(self, x, c = None):
        z = super().forward(x, c = c)
        return {'cls_logits':self.classifier(z), 'latent':z}
    def compute_loss(self, x, y, c = None):
        res = self.forward(x, c = c)
        losses = []
        for loss, weight in zip(self.cls_losses, self.cls_loss_weights):
            losses.append(loss(res['cls_logits'], y) * weight) 
        return losses, res

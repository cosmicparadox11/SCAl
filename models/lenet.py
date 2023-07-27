import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, loss_fn, make_batchnorm
from config import cfg

class LeNet(nn.Module):
    """
        Simple network for MNIST dataset and its variants
    """
    def __init__(self, data_shape, hidden_size, num_classes=10, **kwargs):
        super().__init__()
        self.in_channels = data_shape[0]
        self.hidden_size = hidden_size
        self.features = nn.Sequential(
            nn.Conv2d(data_shape[0], hidden_size[0], 5, 1),
            nn.GroupNorm(2, hidden_size[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_size[0], hidden_size[1], 5, 1),
            nn.GroupNorm(2, hidden_size[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(5*5*hidden_size[1], 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def f(self, x):
        x = self.features(x)
        f = x.view(x.size(0), -1)
        x = self.classifier(f)
        return f, x
    
    def forward(self, input):
        output = {}
        if 'sup' in cfg['loss_mode'] and 'test' not in input:
            _, output['target'] = self.f(input['augw'])
        elif f'fix' in cfg['loss_mode'] and 'test' not in input and cfg['pred'] == True:
            _, output['target'] = self.f(input['augw'])
        elif 'gen' in cfg['loss_mode']:
            _, output['target'] = self.f(input)
            return output['target'], None
        elif 'train-server' in cfg['loss_mode']:
            _, output['target'] = self.f(input['data'])
        else:
            # assert False, 'dataset: {}, input size: {}'.format(cfg['data_name'], input['data'].shape)
            _, output['target'] = self.f(input['data'])
        
        if 'loss_mode' in input and 'test' not in input:
            if 'sup' in cfg['loss_mode']:
                output['loss'] = loss_fn(output['target'], input['target'])
            elif input['loss_mode'] == 'fix':
                aug_output, _ = self.f(input['augs'])
                output['loss'] = loss_fn(aug_output, input['target'].detach())
            elif 'bmd' in input['loss_mode']:
                f, x = self.f(input['augw'])
                return f, F.softmax(x, dim=1)
            elif input['loss_mode'] == 'fix-mix' and 'kl_loss' not in input:
                _, aug_output = self.f(input['aug'])
                _, target = self.f(input['data'])
                _, mix_output = self.f(input['mix_data'])
                return aug_output, mix_output, target
            elif input['loss_mode'] == 'fix-mix' and 'kl_loss' in input:
                _, aug_output = self.f(input['aug'])
                return aug_output
            elif input['loss_mode'] == 'train-server':
                output['loss'] = loss_fn(output['target'], input['target'])
        else:
            if not torch.any(input['target'] == -1):
                output['loss'] = loss_fn(output['target'], input['target'])
        
        return output
    
def lenet(momentum=None, track=False, **kwargs):
    data_shape = cfg['data_shape']
    hidden_size = cfg['lenet']['hidden_size']
    target_size = cfg['target_size']
    model = LeNet(data_shape, hidden_size, num_classes=target_size, **kwargs)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model
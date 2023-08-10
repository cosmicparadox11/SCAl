import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn
from config import cfg
from data import SimDataset 
from net_utils import Entropy, CrossEntropyLabelSmooth

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        # self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn1 = nn.GroupNorm(num_groups=2,num_channels=in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_planes)
        self.bn2 = nn.GroupNorm(num_groups=2,num_channels=out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        self.equal_inout = (in_planes == out_planes)
        self.shortcut = (not self.equal_inout) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                             padding=0, bias=False) or None

    def forward(self, x):
        if not self.equal_inout:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equal_inout else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out = torch.add(x if self.equal_inout else self.shortcut(x), out)
        return out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate):
        super().__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, drop_rate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, data_shape, num_classes, depth, widen_factor, drop_rate):
        super().__init__()
        num_down = int(min(math.log2(data_shape[1]), math.log2(data_shape[2]))) - 3
        hidden_size = [16]
        for i in range(num_down + 1):
            hidden_size.append(16 * (2 ** i) * widen_factor)
        n = ((depth - 1) / (num_down + 1) - 1) / 2
        block = BasicBlock
        blocks = []
        blocks.append(nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False))
        blocks.append(NetworkBlock(n, hidden_size[0], hidden_size[1], block, 1, drop_rate))
        for i in range(num_down):
            blocks.append(NetworkBlock(n, hidden_size[i + 1], hidden_size[i + 2], block, 2, drop_rate))
        # blocks.append(nn.BatchNorm2d(hidden_size[-1]))
        blocks.append(nn.GroupNorm(num_groups=2,num_channels=hidden_size[-1]))
        blocks.append(nn.ReLU(inplace=True))
        blocks.append(nn.AdaptiveAvgPool2d(1))
        blocks.append(nn.Flatten())
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Linear(hidden_size[-1], num_classes)

    def f(self, x):
        x = self.blocks(x)
        f=x
        x = self.classifier(x)
        return f,x

    # def forward(self, input):
    #     output = {}
    #     output['target'] = self.f(input['data'])
    #     if 'loss_mode' in input:
    #         if 'sup' in input['loss_mode']:
    #             output['loss'] = loss_fn(output['target'], input['target'])
    #         elif 'fix' in input['loss_mode'] and 'mix' not in input['loss_mode']:
    #             aug_output = self.f(input['aug'])
    #             output['loss'] = loss_fn(aug_output, input['target'].detach())
    #         elif 'fix' in input['loss_mode'] and 'mix' in input['loss_mode']:
    #             aug_output = self.f(input['aug'])
    #             output['loss'] = loss_fn(aug_output, input['target'].detach())
    #             mix_output = self.f(input['mix_data'])
    #             output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
    #                     1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())
    #         else:
    #             raise ValueError('Not valid loss mode')
    #     else:
    #         if not torch.any(input['target'] == -1):
    #             output['loss'] = loss_fn(output['target'], input['target'])
    #     return output
    def forward(self, input):
        output = {}
        # print(cfg['loss_mode'])
        if 'sim' in cfg['loss_mode'] and 'test' not in input:
            if cfg['pred'] == True or 'bl' in cfg['loss_mode']:
                _,output['target'] = self.f(input['augw'])
            else:
                transform=SimDataset('CIFAR10')
                input = transform(input)
                # print(input.keys())
                if 'sim' in cfg['loss_mode'] and input['supervised_mode']!= True:
                    # input_ = torch.cat((input['aug1'],input['aug2']),dim = 0)
                    # N = len(input['aug1'])
                    # # print(N,len(input_))
                    # _,output_ = self.f(input_)
                    # output['sim_vector_i'] = output_[:N]
                    # output['sim_vector_j'] = output_[N:]
                    _,output['sim_vector_i'] = self.f(input['aug1'])
                    _,output['sim_vector_j'] = self.f(input['aug2'])
                    output['target'],_ = self.f(input['augw'])
                elif 'sim' in cfg['loss_mode'] and input['supervised_mode'] == True:
                    # input_ = torch.cat((input['aug1'],input['aug2']),dim = 0)
                    # N = len(input['aug1'])
                    # # print(N,len(input_))
                    # _,output_ = self.f(input_)
                    # output['sim_vector_i'] = output_[:N]
                    # output['sim_vector_j'] = output_[N:]
                    _,output['sim_vector_i'] = self.f(input['aug1'])
                    _,output['sim_vector_j'] = self.f(input['aug2'])
                    output['target'],__ = self.f(input['augw'])
        elif 'sup' in cfg['loss_mode'] and 'test' not in input:
            _,output['target'] = self.f(input['augw'])
            
            # _,output['target'] = self.f(input['data'])
        elif 'fix' in cfg['loss_mode'] and 'test' not in input and cfg['pred'] == True:
            _,output['target'] = self.f(input['augw'])
        elif 'gen' in cfg['loss_mode']:
            _,output['target'] = self.f(input)
            return output['target'],None
        elif 'train-server' in cfg['loss_mode']:
            _,output['target']=self.f(input['data'])

        else:
            _,output['target'] = self.f(input['data'])
        # output['target']= self.f(input['data'])
        
        if 'loss_mode' in input and 'test' not in input:
            # print(input.keys())
            if 'sup' in input['loss_mode']:
                # print(input['target'])
                # output['loss'] = loss_fn(output['target'], input['target'])
                criterion = CrossEntropyLabelSmooth(num_classes=cfg['target_size'], epsilon=0.1, reduction=True)
                output['loss'] = criterion(output['target'], input['target'])
            elif 'sim' in input['loss_mode']:
                if 'ft' in input['loss_mode'] and 'bl' not in input['loss_mode']:
                    if input['epoch']<= cfg['switch_epoch']:
                        # epochl=input['epoch']
                        # print(f'{epochl} training with Sim loss')
                        criterion = SimCLR_Loss(input['batch_size'])
                        # output['classification_loss'] = loss_fn(output['target'], input['target'])
                        output['sim_loss'] =  criterion(output['sim_vector_i'],output['sim_vector_j'])
                        output['loss'] = output['sim_loss']
                        # output['loss'] = info_nce_loss(input['batch_size'],input_)
                    elif input['epoch'] > cfg['switch_epoch']:
                        # epochl=input['epoch']
                        # print(f'{epochl} training with CE loss')
                        output['loss'] = loss_fn(output['target'], input['target'])
                elif 'ft' in input['loss_mode'] and 'bl'  in input['loss_mode']:
                    if input['epoch'] > cfg['switch_epoch']:
                        # epochl=input['epoch']
                        # print(f'{epochl} training with Sim loss')
                        criterion = SimCLR_Loss(input['batch_size'])
                        # output['classification_loss'] = loss_fn(output['target'], input['target'])
                        output['sim_loss'] =  criterion(output['sim_vector_i'],output['sim_vector_j'])
                        output['loss'] = output['sim_loss']
                    elif input['epoch'] <= cfg['switch_epoch']:
                        # epochl=input['epoch']
                        # print(f'{epochl} training with CE loss')
                        output['loss'] = loss_fn(output['target'], input['target'])
                elif 'at' in input['loss_mode']:
                    if cfg['srange'][0]<=input['epoch']<=cfg['srange'][1] or cfg['srange'][2]<=input['epoch']<=cfg['srange'][3] or cfg['srange'][4]<=input['epoch']<=cfg['srange'][5] or cfg['srange'][6]<=input['epoch']<=cfg['srange'][7]:
                        # epochl=input['epoch']
                        # print(f'{epochl} training with CE loss')
                        output['loss'] = loss_fn(output['target'], input['target'])
                    else :
                        # epochl=input['epoch']
                        # print(f'{epochl} training with Sim loss')
                        criterion = SimCLR_Loss(input['batch_size'])
                        # output['classification_loss'] = loss_fn(output['target'], input['target'])
                        output['sim_loss'] =  criterion(output['sim_vector_i'],output['sim_vector_j'])
                        output['loss'] = output['sim_loss']
                else:    
                    if input['supervised_mode'] == True:
                        criterion = SimCLR_Loss(input['batch_size'])
                        output['classification_loss'] = loss_fn(output['target'], input['target'])
                        output['sim_loss'] =  criterion(output['sim_vector_i'],output['sim_vector_j'])
                        output['loss'] = output['classification_loss']+output['sim_loss']
                    elif input['supervised_mode'] == False:
                        criterion = SimCLR_Loss(input['batch_size'])
                        # output['classification_loss'] = loss_fn(output['target'], input['target'])
                        output['sim_loss'] =  criterion(output['sim_vector_i'],output['sim_vector_j'])
                        output['loss'] = output['sim_loss']
            elif input['loss_mode'] == 'fix':
                # aug_output = self.f(input['aug'])
                aug_output,_ = self.f(input['augs'])
                print(type(aug_output))
                output['loss'] = loss_fn(aug_output, input['target'].detach())
            elif 'bmd' in input['loss_mode']:
                # print(input['augw'])
                # print(input.keys())
                f,x =self.f(input['augw'])
                # return f,x
                return f,torch.softmax(x,dim=1)
                
            elif input['loss_mode'] == 'fix-mix' and 'kl_loss' not in input:
                _,aug_output = self.f(input['aug'])
                _,target = self.f(input['data'])
                # print((input['aug'].shape)[0])
                # print(input['id'].tolist())
                # elr_loss_fn = elr_loss(500)
                # output['loss'] = loss_fn(aug_output, input['target'].detach())
                # print(f'input target')
                # print(input['target'])
                # output['loss']  = elr_loss_fn(input['id'].detach().tolist(),aug_output, input['target'].detach())
        
                _,mix_output = self.f(input['mix_data'])
                # print(mix_output)
                return aug_output,mix_output,target
                # if 'ci_data' in input:
                #     # print('entering ci')
                #     _,ci_output = self.f(input['ci_data'])
                #     output['loss'] += loss_fn(ci_output,input['ci_target'].detach())
                # # output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
                # #         1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())
                # output['loss'] += input['lam'] * elr_loss_fn(input['id'].detach(),mix_output, input['mix_target'][:, 0].detach()) + (
                #         1 - input['lam']) * elr_loss_fn(input['id'].detach(),mix_output, input['mix_target'][:, 1].detach())
            elif input['loss_mode'] == 'fix-mix' and 'kl_loss' in input:
                _,aug_output = self.f(input['aug'])
                return aug_output
            elif input['loss_mode'] == 'train-server':
                output['loss'] = loss_fn(output['target'], input['target'])

        else:
            if not torch.any(input['target'] == -1):
                # output['loss'] = loss_fn(output['target'], input['target'])
                criterion = CrossEntropyLabelSmooth(num_classes=cfg['target_size'], epsilon=0.1, reduction=True)
                output['loss'] = criterion(output['target'], input['target'])

        return output


def wresnet28x2(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet28x2']['depth']
    widen_factor = cfg['wresnet28x2']['widen_factor']
    drop_rate = cfg['wresnet28x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model


def wresnet28x8(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet28x8']['depth']
    widen_factor = cfg['wresnet28x8']['widen_factor']
    drop_rate = cfg['wresnet28x8']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model


def wresnet37x2(momentum=None, track=False):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    depth = cfg['wresnet37x2']['depth']
    widen_factor = cfg['wresnet37x2']['widen_factor']
    drop_rate = cfg['wresnet37x2']['drop_rate']
    model = WideResNet(data_shape, target_size, depth, widen_factor, drop_rate)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model

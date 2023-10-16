import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, make_batchnorm, loss_fn ,info_nce_loss, SimCLR_Loss,elr_loss, register_act_hooks
from utils import to_device, make_optimizer, collate, to_device
from data import SimDataset 
# import data.info_nce_loss as info_nce_loss
from config import cfg

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        # self.n1 = nn.GroupNorm(num_groups=2,num_channels=in_planes)
        # self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.n2 = nn.GroupNorm(num_groups=2,num_channels=planes)
        # self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        # out = F.relu(self.n1(x))
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # out = self.conv2(F.relu(self.n2(out)))
        out = self.conv2(F.relu(out))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride):
        super(Bottleneck, self).__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out = self.conv3(F.relu(self.n3(out)))
        out += shortcut
        return out

class LinearLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias = True,
                 use_bn = False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, 
                                self.out_features, 
                                bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x
class ProjectionHead(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type = 'nonlinear',
                 **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features,self.out_features,False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(
                LinearLayer(self.in_features,self.hidden_features,True, True),
                nn.ReLU(),
                LinearLayer(self.hidden_features,self.out_features,False,True))
        
    def forward(self,x):
        x = self.layers(x)
        return x
class Embedding(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
    
        super(Embedding, self).__init__()
        # self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        # self.bn = torch.nn.GroupNorm(2, embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        # self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        # print(self.bottleneck,x.shape)
        x = self.bottleneck(x)
        # if self.type == "bn":
        #     x = self.bn(x)
        return x 
class Classifier(nn.Module):
    def __init__(self, embed_dim, class_num, type="linear"):
        super(Classifier, self).__init__()
        
        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
            # self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            # self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x
class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, target_size,sim_out=int(128)):
        super().__init__()
        self.act_stats = {}
        self.in_planes = hidden_size[0]
        self.backbone_feat_dim = 512
        self.embed_feat_dim  = 256
        self.class_num = cfg['target_size']
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        # self.n4 = nn.GroupNorm(num_groups=2,num_channels=hidden_size[3] * block.expansion)
        # self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        # self.linear = nn.Linear(hidden_size[3] * block.expansion, target_size)
        self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim, type="bn")
        
        self.class_layer = Classifier(self.embed_feat_dim, class_num=self.class_num, type="wn")
        # self.projection = ProjectionHead(hidden_size[3] * block.expansion,hidden_size[3] * block.expansion,sim_out)
        # self.projection = nn.Sequential(nn.Linear(hidden_size[3] * block.expansion,hidden_size[3] * block.expansion),
        #                                 nn.ReLU(),
        #                                 nn.Linear(hidden_size[3] * block.expansion,sim_out))
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def f(self, x):
    #     x = self.conv1(x)
    #     x = self.layer1(x)
    #     x = self.layer2(x)
    #     x = self.layer3(x)
    #     x = self.layer4(x)
    #     x = F.relu(self.n4(x))
    #     x = F.adaptive_avg_pool2d(x, 1)
    #     x = x.view(x.size(0), -1)
    #     z = self.projection(x)
    #     x = self.linear(x)
    #     return x , z
    def f(self, x,apply_softmax=False):
        # print(x.shape)
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = F.relu(self.n4(x))
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        # f = x
        f = self.feat_embed_layer(x)
        # x = self.linear(x)
        x = self.class_layer(f)
        if apply_softmax:
            x = torch.softmax(x, dim=1)
        else:
            pass
        return f,x
    # def forward(self, input):
    #     output = {}
        # if 'sim' in cfg['loss_mode']:
        #     if cfg['pred'] == True:
        #         output['target'],_ = self.f(input['augw'])
        #     else:
        #         transform=SimDataset('CIFAR10')
        #         input = transform(input)
        #         # print(input.keys())
        #         if 'sim' in cfg['loss_mode'] and input['supervised_mode']!= True:
        #             _,output['sim_vector_i'] = self.f(input['aug1'])
        #             _,output['sim_vector_j'] = self.f(input['aug2'])
        #             output['target'],_ = self.f(input['data'])
        #         elif 'sim' in cfg['loss_mode'] and input['supervised_mode'] == True:
        #             _,output['sim_vector_i'] = self.f(input['aug1'])
        #             _,output['sim_vector_j'] = self.f(input['aug2'])
        #             output['target'],_ = self.f(input['data'])
        # else:
        #     output['target'],_ = self.f(input['data'])
    #     # output['target']= self.f(input['data'])
    #     if 'loss_mode' in input and 'test' not in input and cfg['pred'] == False:
    #         if input['loss_mode'] == 'sup':
    #             output['loss'] = loss_fn(output['target'], input['target'])
    #         elif input['loss_mode'] == 'sim':
    #             if input['supervised_mode'] == True:
    #                 criterion = SimCLR_Loss(input['batch_size'])
    #                 output['classification_loss'] = loss_fn(output['target'], input['target'])
    #                 output['sim_loss'] =  criterion(output['sim_vector_i'],output['sim_vector_j'])
    #                 output['loss'] = output['classification_loss']+output['sim_loss']
    #             elif input['supervised_mode'] == False:
    #                 criterion = SimCLR_Loss(input['batch_size'])
    #                 # output['classification_loss'] = loss_fn(output['target'], input['target'])
    #                 output['sim_loss'] =  criterion(output['sim_vector_i'],output['sim_vector_j'])
    #                 output['loss'] = output['sim_loss']
    #         elif input['loss_mode'] == 'fix':
    #             aug_output = self.f(input['aug'])
    #             output['loss'] = loss_fn(aug_output, input['target'].detach())
    #         elif input['loss_mode'] == 'fix-mix':
    #             aug_output = self.f(input['aug'])
    #             output['loss'] = loss_fn(aug_output, input['target'].detach())
    #             mix_output = self.f(input['mix_data'])
    #             output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
    #                     1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())

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
                act_loss = sum([item['mean_norm'] for item in list(self.act_stats.values())])
                output['loss'] = loss_fn(output['target'], input['target']) + act_loss
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
                output['loss'] = loss_fn(output['target'], input['target'])
        return output


# def resnet9(momentum=None, track=False):
def resnet9(momentum=None, track=True):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['resnet9']['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [1, 1, 1, 1], target_size)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))

    ## Register forward hook ##
    register_act_hooks(model, compute_mean_norm=cfg['compute_mean_norm'], compute_std_dev=cfg['compute_std_dev'])

    return model


def resnet18(momentum=None, track=False):
# def resnet18(momentum=0.1, track=True):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['resnet18']['hidden_size']
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], target_size)
    model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model
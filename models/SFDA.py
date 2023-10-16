import torch 
import numpy as np 
import torch.nn as nn
from torchvision import models
from config import cfg
from .utils import init_param, make_batchnorm, loss_fn ,info_nce_loss, SimCLR_Loss,elr_loss, register_act_hooks
from data import SimDataset 
from net_utils import Entropy, CrossEntropyLabelSmooth
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, 
            "vgg16":models.vgg16, "vgg19":models.vgg19, 
            "vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn,
            "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 

class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    # self.in_features = model_vgg.classifier[6].in_features
    self.backbone_feat_dim = model_vgg.classifier[6].in_features
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, 
            "resnet50":models.resnet50, "resnet101":models.resnet101,
            "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d,
            "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        # self.bn1 = torch.nn.GroupNorm(2, 64)
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
    
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        # self.bn = torch.nn.GroupNorm(2, embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        # print(self.bottleneck,x.shape)
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class Embedding_SDA(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
    
        super(Embedding_SDA, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type
        self.em = nn.Embedding(2, 256)
        
    def forward(self, x, t, s=100, all_mask=False):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        out = x
        if t == 0:
            t = torch.LongTensor([t]).cuda()
            self.mask = nn.Sigmoid()(self.em(t) * s)
            flg = torch.isnan(self.mask).sum()
            out = out * self.mask
        if t == 1:
            t_ = torch.LongTensor([0]).cuda()
            self.mask = nn.Sigmoid()(self.em(t_) * s)
            t = torch.LongTensor([t]).cuda()
            mask = nn.Sigmoid()(self.em(t) * s)
            out = out * mask
        if all_mask:
            t0 = torch.LongTensor([0]).cuda()
            t1 = torch.LongTensor([1]).cuda()
            mask0 = nn.Sigmoid()(self.em(t0) * s)
            mask1 = nn.Sigmoid()(self.em(t1) * s)
            self.mask=mask0
            out0 = out * mask0
            out1 = out * mask1
        if all_mask:
            return (out0,out1), (self.mask,mask1)
        else:
            return out, self.mask
    

class Classifier(nn.Module):
    def __init__(self, embed_dim, class_num, type="linear"):
        super(Classifier, self).__init__()
        
        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class SFDA(nn.Module):
    
    def __init__(self):
        
        super(SFDA, self).__init__()
        ## Activation statistics ##
        self.act_stats = {}

        self.backbone_arch = cfg['backbone_arch'] # resnet101
        self.embed_feat_dim = cfg['embed_feat_dim'] # 256
        self.class_num = cfg['target_size']          # 12 for VisDA

        if "resnet" in self.backbone_arch:   
            self.backbone_layer = ResBase(self.backbone_arch) 
        elif "vgg" in self.backbone_arch:
            self.backbone_layer = VGGBase(self.backbone_arch)
        else:
            raise ValueError("Unknown Feature Backbone ARCH of {}".format(self.backbone_arch))
        
        self.backbone_feat_dim = self.backbone_layer.backbone_feat_dim
        
        self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim, type="bn")
        
        self.class_layer = Classifier(self.embed_feat_dim, class_num=self.class_num, type="wn")
        # self.class_layer = Classifier(self.backbone_feat_dim, class_num=self.class_num)
    
    def get_emd_feat(self, input_imgs):
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        return embed_feat
    
    def f(self, input_imgs, apply_softmax=False):
        
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        # print(backbone_feat.shape)
        embed_feat = self.feat_embed_layer(backbone_feat)
        
        cls_out = self.class_layer(embed_feat)
        # cls_out = self.class_layer(backbone_feat)
        if apply_softmax:
            cls_out = torch.softmax(cls_out, dim=1)
        else:
            pass
        
        return embed_feat, cls_out
        # return backbone_feat,cls_out
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
                # print('label smoothning')
                criterion = CrossEntropyLabelSmooth(num_classes=cfg['target_size'], epsilon=0.1, reduction=True)
                act_loss = sum([item['mean_norm'] for item in list(self.act_stats.values())])
                output['loss'] = criterion(output['target'], input['target']) + act_loss
                # print(output['loss'])
                # output['loss'] = loss_fn(output['target'], input['target'])
                
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
                # print('label smoothning test')
                criterion = CrossEntropyLabelSmooth(num_classes=cfg['target_size'], epsilon=0.1, reduction=True)
                output['loss'] = criterion(output['target'], input['target'])
        return output
def resnet_sfda(momentum=None, track=True):
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['resnet9']['hidden_size']
    # model = ResNet(data_shape, hidden_size, Block, [1, 1, 1, 1], target_size)
    model = SFDA()
    # model.apply(init_param)
    model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    return model
def resnet50(momentum=None, track=True):
    # data_shape = cfg['data_shape']
    # target_size = cfg['target_size']
    # hidden_size = cfg['resnet9']['hidden_size']
    # # model = ResNet(data_shape, hidden_size, Block, [1, 1, 1, 1], target_size)
    model = SFDA()
    model.backbone_arch = 'resnet50'
    model = convert_layers(model, torch.nn.BatchNorm2d, torch.nn.GroupNorm, num_groups = 64,convert_weights=False)
    model = convert_layers(model, torch.nn.BatchNorm1d, torch.nn.GroupNorm, num_groups = 64,convert_weights=False)
    if cfg['pre_trained']:
        print('loading pretrained model')
        model = get_pretrained_GN(model)
    # model.apply(init_param)
    # model.apply(lambda m: make_batchnorm(m, momentum=momentum, track_running_stats=track))
    
    ## Register forward hook ##
    register_act_hooks(model, compute_mean_norm=cfg['compute_mean_norm'], compute_std_dev=cfg['compute_std_dev'])

    return model

def get_pretrained_GN(model):
    import pickle
    path = "/home/sampathkoti/Downloads/R-50-GN.pkl"
    
    # path = "/media/cds/DATA2/Yeswanth/SSFL/SemiFL/R-50-GN.pkl"
    with open(path, 'rb') as f:
        m=pickle.load(f,encoding="latin1")
    for k,v in model.state_dict().items():
        c=['1','2','3']
        c_=['a','b','c']
        if k in ['backbone_layer.conv1.weight', 'backbone_layer.bn1.weight', 'backbone_layer.bn1.bias']:
            if k == 'backbone_layer.conv1.weight' :
                v=m['blobs']['conv1_w']
            elif k == 'backbone_layer.bn1.weight':
                v=m['blobs']['conv1_gn_s']
            elif k == 'backbone_layer.bn1.bias':
                v=m['blobs']['conv1_gn_b']
        elif k in "backbone_layer.layer1":
            mv = k.split('.')
            if mv[2]=='0':
                if 'conv' in k:
                    v = m['blobs'][f'res2_0_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res2_0_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res2_0_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res2_0_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res2_0_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res2_0_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2] == "1":
                if 'conv' in k:
                    v = m['blobs'][f'res2_1_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res2_1_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res2_1_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res2_1_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res2_1_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res2_1_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2] == '2':
                if 'conv' in k:
                    v = m['blobs'][f'res2_2_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res2_2_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res2_2_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res2_2_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res2_2_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res2_2_branch1{c_[c.index(mv[3][-1])]}_gn_b']
        elif k in "backbone_layer.layer2":
            mv = k.split('.')
            if mv[2]=='0':
                if 'conv' in k:
                    v = m['blobs'][f'res3_0_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res3_0_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res3_0_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res3_0_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res3_0_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res3_0_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='1':
                if 'conv' in k:
                    v = m['blobs'][f'res3_1_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res3_1_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res3_1_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res3_1_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res3_1_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res3_1_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='2':
                if 'conv' in k:
                    v = m['blobs'][f'res3_2_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res3_2_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res3_2_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res3_2_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res3_2_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res3_2_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='3':
                if 'conv' in k:
                    v = m['blobs'][f'res3_3_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res3_3_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res3_3_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res3_3_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res3_3_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res3_3_branch1{c_[c.index(mv[3][-1])]}_gn_b']
        elif k in "backbone_layer.layer3":
            mv = k.split('.')
            if mv[2]=='0':
                if 'conv' in k:
                    v = m['blobs'][f'res4_0_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res4_0_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res4_0_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res4_0_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res4_0_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res4_0_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='1':
                if 'conv' in k:
                    v = m['blobs'][f'res4_1_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res4_1_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res4_1_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res4_1_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res4_1_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res4_1_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='2':
                if 'conv' in k:
                    v = m['blobs'][f'res4_2_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res4_2_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res4_2_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res4_2_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res4_2_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res4_2_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='3':
                if 'conv' in k:
                    v = m['blobs'][f'res4_3_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res4_3_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res4_3_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res4_3_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res4_3_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res4_3_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='4':
                if 'conv' in k:
                    v = m['blobs'][f'res4_4_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res4_4_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res4_4_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res4_4_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res4_4_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res4_4_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='5':
                if 'conv' in k:
                    v = m['blobs'][f'res4_5_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res4_5_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res4_5_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res4_5_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res4_5_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res4_5_branch1{c_[c.index(mv[3][-1])]}_gn_b']
        elif k  in "backbone_layer.layer4":
            mv = k.split('.')
            if mv[2]=='0':
                if 'conv' in k:
                    v = m['blobs'][f'res5_0_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res5_0_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res5_0_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res5_0_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res5_0_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res5_0_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='1':
                if 'conv' in k:
                    v = m['blobs'][f'res5_1_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res5_1_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res5_1_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res5_1_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res5_1_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res5_1_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='2':
                if 'conv' in k:
                    v = m['blobs'][f'res5_2_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res5_2_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res5_2_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res5_2_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res5_2_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res5_2_branch1{c_[c.index(mv[3][-1])]}_gn_b']
            elif mv[2]=='3':
                if 'conv' in k:
                    v = m['blobs'][f'res5_3_branch2{c_[c.index(mv[3][-1])]}_w']
                elif 'bn' in k :
                    if 'weight' in k:
                        v = m['blobs'][f'res5_3_branch2{c_[c.index(mv[3][-1])]}_gn_s']
                    elif 'bias' in k:
                        v = m['blobs'][f'res5_3_branch2{c_[c.index(mv[3][-1])]}_gn_b']
                elif 'downsample' in k:
                    if '0.weight' in k:
                        v = m['blobs'][f'res5_3_branch1{c_[c.index(mv[3][-1])]}_w']
                    elif '1.weight' in k:
                        v = m['blobs'][f'res5_3_branch1{c_[c.index(mv[3][-1])]}_gn_s']
                    elif '1.bias' in k:
                        v = m['blobs'][f'res5_3_branch1{c_[c.index(mv[3][-1])]}_gn_b']

    return model
def convert_layers(model, layer_type_old, layer_type_new, num_groups,convert_weights=False):
    
    for name, module in reversed(model._modules.items()):
        # print(name)
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_layers(module, layer_type_old, layer_type_new, convert_weights)
        
        if type(module) == layer_type_old:
            # print(layer_type_new,layer_type_old,name)
            layer_old = module
            # print('converting layers')
            # print()
            # layer_new = layer_type_new(module.num_features if num_groups is None else num_groups, module.num_features, module.eps, module.affine) 
            layer_new = layer_type_new(32, module.num_features, module.eps, module.affine) 


            if convert_weights:
                layer_new.weight = layer_old.weight
                layer_new.bias = layer_old.bias

            model._modules[name] = layer_new

    return model
if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_arch", type=str, default="vit")
    parser.add_argument("--embed_feat_dim", type=int, default=256)
    parser.add_argument("--class_num", type=int, default=12)
    
    args = parser.parse_args()
    
    sfda_model = SFDA(args)
    rand_data = torch.rand((10, 3, 224, 224))
    embed_feat, cls_out = sfda_model(rand_data)
    
    print(embed_feat.shape)
    print(cls_out.shape)
    print(sfda_model.backbone_layer.in_features)
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from utils import to_device, make_optimizer, collate, to_device
import numpy as np
from config import cfg
import functools

# Register forward hook
def register_act_hooks(model, compute_mean_norm=False, compute_std_dev=False):
    def create_hook(name):
        return lambda module, input, output: hook_fn(module, input, output, layer_name=name, model=model, compute_mean_norm=compute_mean_norm, compute_std_dev=compute_std_dev)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and 'class_layer' not in name:
            module.register_forward_hook(create_hook(name))

## Hook function
def hook_fn(module, input, output, layer_name, model, compute_mean_norm=False, compute_std_dev=False):
    result = {}
    if compute_mean_norm:
        x_ = output.clone().reshape(-1)
        mean_norm = torch.mean(F.relu(x_)**2)
        # mean_norm = F.relu(output.clone()).norm(p=2, dim=1).mean()
        result['mean_norm'] = cfg['mean_norm_reg'] * mean_norm

    if compute_std_dev:
        std_dev = F.relu(output.clone()).std()
        result['std_dev'] = cfg['std_dev_reg'] * std_dev

    model.act_stats[layer_name] = result
##register pre-BN hooks
def register_preBN_hooks(model, compute_running_mean=False, compute_running_var=False):
    # def create_hook(module,input):
    #     x_ = lambda module, input: hook_fn_BN(module, input, layer_name=name, model=model, \
    #                                                  compute_running_mean=compute_running_mean, compute_running_var=compute_running_var)
    #     return input
    def create_hook(name):
        # print(input)
        # exit()
        return lambda module, input,output: hook_fn_BN(module, input,output, layer_name=name, model=model, \
                                                     compute_running_mean=compute_running_mean, compute_running_var=compute_running_var)
    
    
    for name, module in model.named_modules():
        # print(name)
        if (isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d) ) and 'class_layer' not in name and 'class_layer.fc' not in name:
            # print('inside',name)
            handle = module.register_forward_hook(create_hook(name))
            # module.register_forward_pre_hook(lambda module, input: hook_fn_BN(module, input, layer_name=name, model=model, \
            #                                          compute_running_mean=compute_running_mean, compute_running_var=compute_running_var))

def hook_fn_BN(module, input,output, model,layer_name = None,compute_running_mean=cfg['compute_running_mean'], compute_running_var=cfg['compute_running_var']):
    # print(layer_name)
    # print(input[0].shape)
    # print(model.running_mean)
    # x = input[0].clone().detach().cpu()
    if cfg['post_conv']:
        x = input[0].clone()
        exponential_average_factor = 0.1
        if len(input[0].shape)==4:
            if cfg['compute_running_mean']:
                mean = x.mean(dim=(0, 2, 3), keepdim=True)
                var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            else:
                mean = model.running_mean[layer_name]
                var = custom_var(x,mean,dim=(0, 2, 3))
        else:
            if cfg['compute_running_mean']:
                mean = x.mean(dim=(0,), keepdim=True)
                var = x.var(dim=(0,), unbiased=False, keepdim=True)
            else:
                mean = model.running_mean[layer_name]
                var = custom_var(x,mean,dim=(0,))

        n = x.numel() / x.size(1)
        
        with torch.no_grad():
            if layer_name in model.running_mean.keys() and layer_name in model.running_var.keys() :
                if cfg['compute_running_mean']:
                    model.running_mean[layer_name] = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * model.running_mean[layer_name]
                else:
                    model.running_mean[layer_name] = mean 
                # update running_var with unbiased var
                model.running_var[layer_name]= exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) *  model.running_var[layer_name]
            else:
                if cfg['compute_running_mean']:
                    model.running_mean[layer_name]= exponential_average_factor * mean\
                        + (1 - exponential_average_factor) * cfg['running_mean']
                else:
                    model.running_mean[layer_name] = mean 
                # update running_var with unbiased var
                model.running_var[layer_name]= exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * cfg['running_var']
    return 
def custom_var(input_tensor,mean, dim=None, unbiased=True):
    # n = input_tensor.numel()
    n = input_tensor.numel() / input_tensor.size(1)
    mean = mean

    sum_squared_diff = (input_tensor - mean).pow(2).sum(dim=dim)

    if unbiased:
        # Bessel's correction for unbiased variance
        variance = sum_squared_diff / (n - 1)
    else:
        variance = sum_squared_diff / n

    return variance        
def hook_fn_BN_(module, input, layer_name, model):
        x = input.clone()
        exponential_average_factor = 0.1
        mean = x.mean(dim=(0, 2, 3), keepdim=True)
        var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
        n = x.numel() / x.size(1)
        with torch.no_grad():
            if layer_name in model.running_mean.keys():
                model.running_mean[layer_name] = exponential_average_factor * mean\
                + (1 - exponential_average_factor) * model.running_mean[layer_name]
                # update running_var with unbiased var
                model.running_var[layer_name]= exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) *  model.running_var[layer_name]
            else:
                model.running_mean[layer_name]= exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * cfg['running_mean']
                # update running_var with unbiased var
                model.running_var[layer_name]= exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * cfg['running_var']        
def init_param(m):
    if isinstance(m, nn.Conv2d) and isinstance(m, models.DecConv2d):
        nn.init.kaiming_normal_(m.sigma_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(m.phi_weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m


def make_batchnorm(m, momentum, track_running_stats):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        # print(m)
        m.momentum = momentum
        m.track_running_stats = track_running_stats
        if track_running_stats:
            m.register_buffer('running_mean', torch.zeros(m.num_features, device=m.weight.device))
            m.register_buffer('running_var', torch.ones(m.num_features, device=m.weight.device))
            m.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long, device=m.weight.device))
        else:
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None
    return m


def loss_fn(output, target, reduction='mean'):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = F.mse_loss(output, target, reduction=reduction)
    return loss


def mse_loss(output, target, weight=None):
    mse = F.mse_loss(output, target, reduction='none')
    mse = weight * mse if weight is not None else mse
    mse = torch.sum(mse)
    mse /= output.size(0)
    return mse


def cross_entropy_loss(output, target, weight=None):
    target = (target.topk(1, 1, True, True)[1]).view(-1)
    ce = F.cross_entropy(output, target, reduction='mean', weight=weight)
    return ce


def kld_loss(output, target, weight=None, T=1):
    kld = F.kl_div(F.log_softmax(output, dim=-1), F.softmax(target / T, dim=-1), reduction='none')
    kld = weight * kld if weight is not None else kld
    kld = torch.sum(kld)
    kld /= output.size(0)
    return kld
def info_nce_loss(batch_size, features):

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(cfg['device'])

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(cfg['device'])
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(cfg['device'])

        logits = logits / 0.07
        criterion = torch.nn.CrossEntropyLoss().to(cfg['device'])
        loss = criterion(logits, labels)
        return loss
class SimCLR_Loss(nn.Module):
    def __init__(self, batch_size, temperature=0.07):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = cfg['gm']

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):

        N = 2 * self.batch_size
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        
        #SIMCLR
        labels = torch.from_numpy(np.array([0]*N)).reshape(-1).to(positive_samples.device).long() #.float()
        
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        
        return loss
    
#  def info_nce_loss(self, features):

#         labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
#         labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
#         labels = labels.to(self.args.device)

#         features = F.normalize(features, dim=1)

#         similarity_matrix = torch.matmul(features, features.T)
#         # assert similarity_matrix.shape == (
#         #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
#         # assert similarity_matrix.shape == labels.shape

#         # discard the main diagonal from both: labels and similarities matrix
#         mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
#         labels = labels[~mask].view(labels.shape[0], -1)
#         similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
#         # assert similarity_matrix.shape == labels.shape

#         # select and combine multiple positives
#         positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

#         # select only the negatives the negatives
#         negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

#         logits = torch.cat([positives, negatives], dim=1)
#         labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

#         logits = logits / self.args.temperature
#         return logits, labels
class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, lambda_ = cfg['elr_lam'], beta=0.7):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.USE_CUDA = True
        self.target = to_device(torch.zeros(num_examp, self.num_classes),cfg['device']) if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        self.lambda_ = lambda_
        

    def forward(self, index, output, label):
        r"""Early Learning Regularization.
         Args
         * `index` Training sample index, due to training set shuffling, index is used to track training examples in different iterations.
         * `output` Model's logits, same as PyTorch provided loss functions.
         * `label` Labels, same as PyTorch provided loss functions.
         """
        # print(label.min(), label.max())
        # print(label)
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        # print(self.target[index].shape)
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        # print(self.target[index].shape)
        ce_loss = F.cross_entropy(output, label)
        # print(y_pred.shape)
        
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  self.lambda_*elr_reg
        return  final_loss
import collections.abc as container_abcs
import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset,dataset_unsup= None):
    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    print(cfg['data_size'])
    cfg['target_size'] = dataset['train'].target_size
    if dataset_unsup is not None:
        cfg['data_size_unsup'] = {'train': len(dataset_unsup['train']), 'test': len(dataset_unsup['test'])}
        cfg['target_size_unsup'] = dataset_unsup['train'].target_size
        print(cfg['data_size_unsup'])

    return
def process_dataset_multi(dataset,dataset_unsup_dict= None):
    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    cfg['target_size'] = dataset['train'].target_size
    if dataset_unsup_dict is not None:
        for domain_id,dataset_unsup in dataset_unsup_dict.items():
            domain = cfg['unsup_list'][domain_id]
            cfg[f'data_size_unsup_{domain}'] = {'train': len(dataset_unsup['train']), 'test': len(dataset_unsup['test'])}
            cfg[f'target_size_unsup_{domain}'] = dataset_unsup['train'].target_size

    return


def process_control():
    if cfg['control']['num_supervised'] == 'fs':
        cfg['control']['num_supervised'] = '-1'
    cfg['num_supervised'] = int(cfg['control']['num_supervised'])
    # data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'CIFAR10': [3, 32, 32], 'CIFAR100': [3, 32, 32],
    #               'SVHN': [3, 32, 32]}
    data_shape = {'MNIST': [3, 28, 28],'MNIST_M': [3, 28, 28], 'FashionMNIST': [1, 28, 28], 'CIFAR10': [3, 32, 32], 'CIFAR100': [3, 32, 32],
                  'SVHN': [3, 32, 32],'USPS':[3,16,16],'office31':[3,1000,1000], 'amazon':[3,300,300],'webcam':[3,477,477]}
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['conv'] = {'hidden_size': [32, 64]}
    cfg['lenet'] = {'hidden_size': [20, 50]}
    cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['unsup_ratio'] = 1
    if 'loss_mode' in cfg['control']:
        cfg['loss_mode'] = cfg['control']['loss_mode']
        # if 'fix' in cfg['loss_mode']:
        #     cfg['threshold'] = float(cfg['control']['loss_mode'].split('-')[2].split('@')[1])
    if 'num_clients' in cfg['control']:
        cfg['num_clients'] = int(cfg['control']['num_clients'])
        cfg['active_rate'] = float(cfg['control']['active_rate'])
        cfg['data_split_mode'] = cfg['control']['data_split_mode']
        cfg['local_epoch'] = cfg['control']['local_epoch'].split('-')
        cfg['gm'] = float(cfg['control']['gm'])
        cfg['sbn'] = int(cfg['control']['sbn'])
        if 'ft' in cfg['control']:
            cfg['ft'] = int(cfg['control']['ft'])
        if 'lc' in cfg['control']:
            cfg['lc'] = int(cfg['control']['lc'])
        cfg['server'] = {}
        cfg['server']['shuffle'] = {'train': True, 'test': False}
        if cfg['num_supervised'] > 1000:
            cfg['server']['batch_size'] = {'train': 250, 'test': 50}
        else:
            cfg['server']['batch_size'] = {'train': 10, 'test': 50}
        cfg['server']['num_epochs'] = int(np.ceil(float(cfg['local_epoch'][1])))
        cfg['client'] = {}
        cfg['client']['shuffle'] = {'train': True, 'test': False}
        cfg['client']['batch_size'] = {'train': 10, 'test': 50}
        cfg['client']['num_epochs'] = int(np.ceil(float(cfg['local_epoch'][0])))
        cfg['local'] = {}
        cfg['local']['optimizer_name'] = 'SGD'
        cfg['local']['lr'] = 3e-2
        cfg['local']['momentum'] = 0.9
        cfg['local']['weight_decay'] = 5e-4
        cfg['local']['nesterov'] = True
        cfg['global'] = {}
        cfg['global']['batch_size'] = {'train': 50, 'test': 50}
        cfg['global']['shuffle'] = {'train': True, 'test': False}
        cfg['global']['num_epochs'] = 150
        cfg['global']['optimizer_name'] = 'SGD'
        cfg['global']['lr'] = 1
        cfg['global']['momentum'] = cfg['gm']
        cfg['global']['weight_decay'] = 0
        cfg['global']['nesterov'] = False
        # cfg['global']['scheduler_name'] = 'CosineAnnealingLR'
        cfg['global']['scheduler_name'] = 'ExponentialLR'
        # cfg['global']['scheduler_name'] = 'StepLR'
        cfg['alpha'] = 0.75
    else:
        model_name = cfg['model_name']
        cfg[model_name]['shuffle'] = {'train': True, 'test': False}
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['lr'] = 3e-1
        cfg[model_name]['momentum'] = 0.9
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['nesterov'] = True
        cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'
        cfg[model_name]['num_epochs'] = 400
        if cfg['num_supervised'] > 1000 or cfg['num_supervised'] == -1:
            cfg[model_name]['batch_size'] = {'train': 250, 'test': 50}
        else:
            cfg[model_name]['batch_size'] = {'train': 10, 'test': 50}
    return


def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(parameters, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(parameters, lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(parameters, lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(parameters, lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        # print('trueeeeeeee')
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        # print('trueeeeeeee')
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs'], eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model_tag, load_tag='checkpoint', verbose=True):
# def resume(model_tag, load_tag='best', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        result = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
        result = {'epoch': last_epoch, 'logger': logger}
    if verbose:
        print('Resume from {}'.format(result['epoch']))
    return result
def resume_DA(model_tag, load_tag='checkpoint',mode = 'source', verbose=True):
# def resume(model_tag, load_tag='best', verbose=True):
    if os.path.exists('./output/model/{}/{}_{}.pt'.format(mode,model_tag, load_tag)):
        result = load('./output/model/{}/{}_{}.pt'.format(mode,model_tag, load_tag))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
        result = {'epoch': last_epoch, 'logger': logger}
    if verbose:
        print('Resume from {}'.format(result['epoch']))
    return result


def collate(input):
    for k in input:
        input[k] = torch.stack(input[k], 0)
    return input
def save_img(image,save_path):

    np_image=image.data.clone().cpu().numpy()
    print(np.shape(np_image))
    np_image=np.squeeze(np_image)
    PIL_image = Image.fromarray(np_image,'L')

    PIL_image.save(save_path)
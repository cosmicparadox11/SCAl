import copy
import torch
import numpy as np
import models
from config import cfg
from Gausian import GaussianBlur
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device
# from pytorch_adapt.datasets import DataloaderCreator, get_office31
from PIL import Image
import random
    
data_stats = {'MNIST': ((0.1307,), (0.3081,)), 'FashionMNIST': ((0.2860,), (0.3530,)),
              'MNIST_M': ((0.1307,), (0.3081,)),
              'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
              'CIFAR100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
              'SVHN': ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            # 'SVHN':((0.5,),
            #                  (0.5,)),
              'STL10': ((0.4409, 0.4279, 0.3868), (0.2683, 0.2610, 0.2687)),
              'USPS':((0.5,),
                             (0.5,)),
               'office31':([0.485, 0.456, 0.406],
                                   [0.229, 0.224, 0.225])}
            #    'dslr':([0.485, 0.456, 0.406],
            #                        [0.229, 0.224, 0.225]),
            #    'webcam':([0.485, 0.456, 0.406],
            #                        [0.229, 0.224, 0.225])}


def fetch_dataset(data_name,domain = None):
    import datasets
    dataset = {}
    print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['MNIST', 'FashionMNIST']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor()]))'.format(data_name))
        # dataset['train'].transform = datasets.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(*data_stats[data_name])])
        dataset['train'].transform = datasets.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()])
        dataset['test'].transform = datasets.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['office31']:
        print(domain)
        # dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
        #                         'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        # dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
        #                        'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'] = datasets.office31(root=root,domain = domain, split='train',
                                transform=datasets.Compose([transforms.ToTensor()]))
        
        dataset['train'].transform = datasets.Compose([
            # transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()])
        if cfg['test_10_crop']:
            crop_list = image_test_10crop()
            for i in range(10):
                dataset['test'] = [datasets.office31(root=root, split='test',domain = domain,
                                transform=crop_list[i]) for i in range(10)]

        else:
            print('test_10crop disabled')
            resize_size = 256
            crop_size = 224
            dataset['test'] = datasets.office31(root=root, split='test',domain = domain,
                                transform=datasets.Compose([transforms.ToTensor()]))
            dataset['test'].transform = datasets.Compose(
                [
                                # transforms.CenterCrop(224),
                                # transforms.ToTensor(),
                                # transforms.Normalize(*data_stats[data_name])
                                # # transforms.RandomResizedCrop(224)
                                ResizeImage(resize_size),
                                transforms.RandomResizedCrop(crop_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(*data_stats[data_name])
                                ]


                                                        )

            
            
        # dataset['train'].transform = datasets.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(*data_stats[data_name])])
        
        # if not cfg['test_10_crop']:
            
    elif data_name in ['CIFAR10', 'CIFAR100']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        # dataset['train'].transform = datasets.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        #     transforms.ToTensor(),
        #     transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['SVHN','MNIST_M']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            # transforms.Grayscale(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            # transforms.Normalize((0.5),(0.5))])
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            # transforms.Grayscale(num_output_channels=1),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
            # transforms.Normalize((0.5), (0.5))])
    elif data_name in ['STL10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['train'].transform = datasets.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
        dataset['test'].transform = datasets.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    elif data_name in ['USPS']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.Grayscale(num_output_channels=3),transforms.ToTensor()]))'.format(data_name))
        # dataset['train'].transform = datasets.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(*data_stats[data_name])])
        dataset['train'].transform = datasets.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(32),
            transforms.ToTensor()])
        dataset['test'].transform = datasets.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(*data_stats[data_name])])
    else:
        raise ValueError('Not valid dataset name')
    print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None, batch_sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        elif batch_sampler is not None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_sampler=batch_sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))

    return data_loader

def make_data_loader_DA(dataset, tag, batch_size=None, shuffle=None, sampler=None, batch_sampler=None):
    data_loader = {}
    print(cfg['test_10_crop'])
    for k in dataset:
        if cfg['test_10_crop']  and k == 'test':
            print('creating test 10 crop')
            _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
            _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
            print(_batch_size,_shuffle)
            if sampler is not None:
                for i in range(10):
                    data_loader[k] = [DataLoader(dataset=dataset, batch_size=_batch_size, sampler=sampler[k],
                                            pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                            worker_init_fn=np.random.seed(cfg['seed'])) for dataset in dataset[k]]
            elif batch_sampler is not None:
                for i in range(10):
                    data_loader[k] = [DataLoader(dataset=dataset, batch_sampler=batch_sampler[k],
                                            pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                            worker_init_fn=np.random.seed(cfg['seed'])  ) for dataset in dataset[k]]
            else:
                for i in range(10):
                    data_loader[k] = [DataLoader(dataset=dataset, batch_size=_batch_size, shuffle=_shuffle,
                                            pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                            worker_init_fn=np.random.seed(cfg['seed'])) for dataset in dataset[k]]
        else :
            _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
            _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
            if sampler is not None:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                            pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                            worker_init_fn=np.random.seed(cfg['seed']))
            elif batch_sampler is not None:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_sampler=batch_sampler[k],
                                            pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                            worker_init_fn=np.random.seed(cfg['seed']))
            else:
                data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                            pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                            worker_init_fn=np.random.seed(cfg['seed']))

    return data_loader

def split_dataset(dataset, num_users, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'] = iid(dataset['train'], num_users)
        data_split['test'] = iid(dataset['test'], num_users)
    elif 'non-iid' in cfg['data_split_mode']:
        data_split['train'] = non_iid(dataset['train'], num_users)
        data_split['test'] = non_iid(dataset['test'], num_users)
    else:
        raise ValueError('Not valid data split mode')
    return data_split


def iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    data_split, idx = {}, list(range(len(dataset)))
    for i in range(num_users):
        num_items_i = min(len(idx), num_items)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:num_items_i]].tolist()
        idx = list(set(idx) - set(data_split[i]))
    return data_split


def non_iid(dataset, num_users):
    target = torch.tensor(dataset.target)
    data_split_mode_list = cfg['data_split_mode'].split('-')
    data_split_mode_tag = data_split_mode_list[-2]
    if data_split_mode_tag == 'l':
        data_split = {i: [] for i in range(num_users)}
        shard_per_user = int(data_split_mode_list[-1])
        target_idx_split = {}
        shard_per_class = int(shard_per_user * num_users / cfg['target_size'])
        for target_i in range(cfg['target_size']):
            target_idx = torch.where(target == target_i)[0]
            num_leftover = len(target_idx) % shard_per_class
            leftover = target_idx[-num_leftover:] if num_leftover > 0 else []
            new_target_idx = target_idx[:-num_leftover] if num_leftover > 0 else target_idx
            new_target_idx = new_target_idx.reshape((shard_per_class, -1)).tolist()
            for i, leftover_target_idx in enumerate(leftover):
                new_target_idx[i] = new_target_idx[i] + [leftover_target_idx.item()]
            target_idx_split[target_i] = new_target_idx
        target_split = list(range(cfg['target_size'])) * shard_per_class
        target_split = torch.tensor(target_split)[torch.randperm(len(target_split))].tolist()
        target_split = torch.tensor(target_split).reshape((num_users, -1)).tolist()
        for i in range(num_users):
            for target_i in target_split[i]:
                idx = torch.randint(len(target_idx_split[target_i]), (1,)).item()
                data_split[i].extend(target_idx_split[target_i].pop(idx))
    elif data_split_mode_tag == 'd':
        beta = float(data_split_mode_list[-1])
        dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_users))
        min_size = 0
        required_min_size = 10
        N = target.size(0)
        while min_size < required_min_size:
            data_split = [[] for _ in range(num_users)]
            for target_i in range(cfg['target_size']):
                target_idx = torch.where(target == target_i)[0]
                proportions = dir.sample()
                proportions = torch.tensor(
                    [p * (len(data_split_idx) < (N / num_users)) for p, data_split_idx in zip(proportions, data_split)])
                proportions = proportions / proportions.sum()
                split_idx = (torch.cumsum(proportions, dim=-1) * len(target_idx)).long().tolist()[:-1]
                split_idx = torch.tensor_split(target_idx, split_idx)
                data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
            min_size = min([len(data_split_idx) for data_split_idx in data_split])
        data_split = {i: data_split[i] for i in range(num_users)}
    else:
        raise ValueError('Not valid data split mode tag')
    return data_split


def separate_dataset(dataset, idx):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[int(s)] for s in idx]
    separated_dataset.target = [dataset.target[int(s)] for s in idx]
    separated_dataset.other['id'] = list(range(len(separated_dataset.data)))
    transform = FixTransform(cfg['data_name'])
    separated_dataset.transform = transform
    return separated_dataset
def separate_dataset_DA(dataset, idx,name=None):
    separated_dataset = copy.deepcopy(dataset)
    separated_dataset.data = [dataset.data[int(s)] for s in idx]
    separated_dataset.target = [dataset.target[int(s)] for s in idx]
    separated_dataset.other['id'] = list(range(len(separated_dataset.data)))
    transform = FixTransform(name)
    separated_dataset.transform = transform
    return separated_dataset

def separate_dataset_su(server_dataset, client_dataset=None, supervised_idx=None):
    if supervised_idx is None:
        if cfg['data_name'] in ['STL10']:
            if cfg['num_supervised'] == -1:
                supervised_idx = torch.arange(5000).tolist()
            else:
                target = torch.tensor(server_dataset.target)[:5000]
                num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
                supervised_idx = []
                for i in range(cfg['target_size']):
                    idx = torch.where(target == i)[0]
                    idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                    supervised_idx.extend(idx)
        else:
            if cfg['num_supervised'] == -1:
                supervised_idx = list(range(len(server_dataset)))
            else:
                target = torch.tensor(server_dataset.target)
                num_supervised_per_class = cfg['num_supervised'] // cfg['target_size']
                supervised_idx = []
                for i in range(cfg['target_size']):
                    idx = torch.where(target == i)[0]
                    idx = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
                    supervised_idx.extend(idx)
    idx = list(range(len(server_dataset)))
    unsupervised_idx = list(set(idx) - set(supervised_idx))
    _server_dataset = separate_dataset(server_dataset, supervised_idx)
    if client_dataset is None:
        _client_dataset = separate_dataset(server_dataset, unsupervised_idx)
    else:
        _client_dataset = separate_dataset(client_dataset, unsupervised_idx)
        transform = FixTransform(cfg['data_name'])
        _client_dataset.transform = transform
    return _server_dataset, _client_dataset, supervised_idx
def split_class_dataset(dataset, data_split_mode = 'iid'):
    data_split = {}
    if data_split_mode == 'iid':
        data_split['train'] = seperate_sup_unsup(dataset['train'])
        data_split['test'] = seperate_sup_unsup(dataset['test'])
    return data_split
def seperate_sup_unsup(client_dataset):
    target = torch.tensor(client_dataset.target)
    num_supervised_per_class = len(client_dataset)// (cfg['target_size']*cfg['num_clients'])
    data_split={}
    for i in range(int(cfg['num_clients'])):
        data_split[i] = []
    for j in range(cfg['target_size']):
        idx = torch.where(target == j)[0]
        
        for i in range(int(cfg['num_clients'])):
            num_items_i = min(len(idx), num_supervised_per_class)
            idx_i = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
            data_split[i].extend(idx_i)
            idx = list(set(idx.tolist()) - set(idx_i))
            idx = torch.Tensor(idx)
        
    return data_split
def split_class_dataset_DA(dataset, data_split_mode = 'iid',split_num = 0):
    data_split = {}
    # print(split_num)
    if data_split_mode == 'iid':
        data_split['train'] = seperate_sup_unsup_DA(dataset['train'],split_num)
        data_split['test'] = seperate_sup_unsup_DA(dataset['test'],split_num)
    return data_split
def seperate_sup_unsup_DA(client_dataset,split_num):
    print('data len')
    print(len(client_dataset))
    target = torch.tensor(client_dataset.target)
    num_supervised_per_class = len(client_dataset)// (cfg['target_size']*split_num) #samples pre class per client
    data_split={}
    for i in range(int(split_num)):
        data_split[i] = []
    for j in range(cfg['target_size']):
        idx = torch.where(target == j)[0]
        
        for i in range(int(split_num)):
            num_items_i = min(len(idx), num_supervised_per_class)
            idx_i = idx[torch.randperm(len(idx))[:num_items_i]].tolist()
            # idx_i = idx[torch.randperm(len(idx))[:num_supervised_per_class]].tolist()
            data_split[i].extend(idx_i)
            idx = list(set(idx.tolist()) - set(idx_i))
            idx = torch.Tensor(idx)
        
    return data_split

def make_batchnorm_dataset_su(server_dataset, client_dataset):
    batchnorm_dataset = copy.deepcopy(server_dataset)
    # print(len(batchnorm_dataset.other['id']))
    batchnorm_dataset.data = batchnorm_dataset.data + client_dataset.data
    batchnorm_dataset.target = batchnorm_dataset.target + client_dataset.target
    batchnorm_dataset.other['id'] = batchnorm_dataset.other['id'] + client_dataset.other['id']
    # print(len(batchnorm_dataset.other['id']))
    return batchnorm_dataset


def make_dataset_normal(dataset):
    import datasets
    _transform = dataset.transform
    transform = datasets.Compose([transforms.ToTensor(), transforms.Normalize(*data_stats[cfg['data_name']])])
    dataset.transform = transform
    return dataset, _transform


def make_batchnorm_stats(dataset, model, tag):
    with torch.no_grad():
        test_model = copy.deepcopy(model)
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=0.1, track_running_stats=True))
        dataset, _transform = make_dataset_normal(dataset)
        # data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
        # test_model.train(True)
        # for i, input in enumerate(data_loader):
        #     input = collate(input)
        #     input = to_device(input, cfg['device'])
        #     input['loss_mode'] = cfg['loss_mode']
        #     input['supervised_mode'] = False
        #     input['test'] = True
        #     input['batch_size'] = cfg['client']['batch_size']['train']
        #     test_model(input)
        dataset.transform = _transform
    return test_model

def make_batchnorm_stats_DA(model, tag):
    with torch.no_grad():
        test_model = eval('models.{}()'.format(cfg['model_name']))
        test_model.to(cfg['device'])
        test_model.load_state_dict(model.state_dict())
        test_model.apply(lambda m: models.make_batchnorm(m, momentum=0.1, track_running_stats=True))
    return test_model
class FixTransform(object):
    def __init__(self, data_name):
        import datasets
        if data_name in ['CIFAR10', 'CIFAR100']:
            self.normal = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize(*data_stats[data_name])
                                ])
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['SVHN','MNIST_M']:
            self.normal = transforms.Compose(
                                [
                                # transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5),(0.5))
                                # transforms.Normalize(*data_stats[data_name])
                                ])
            self.weak = transforms.Compose([
                # transforms.Grayscale(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize((0.5),(0.5))
                # transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                # transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0.5),(0.5))
                # transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['USPS']:
            self.normal = transforms.Compose(
                                [
                                transforms.Resize(32),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor(),
                                transforms.Normalize(*data_stats[data_name])
                                ])
            self.weak = transforms.Compose([
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
                
            ])
            self.strong = transforms.Compose([
                # transforms.Grayscale(num_output_channels=3),
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                # datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['MNIST']:
            self.normal = transforms.Compose(
                                [
                                transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize(*data_stats[data_name])
                                ])
            self.weak = transforms.Compose([
                # transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
                
            ])
            self.strong = transforms.Compose([
                # transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.Grayscale(num_output_channels=3),
                # datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['STL10']:
            self.normal = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize(*data_stats[data_name])
                                ])
            self.weak = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
            self.strong = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(96, padding=12, padding_mode='reflect'),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        elif data_name in ['office31']:
            resize_size = 256
            crop_size = 224
            self.normal = transforms.Compose(
                                [
                                # transforms.CenterCrop(224),
                                # transforms.ToTensor(),
                                # transforms.Normalize(*data_stats[data_name])
                                # # transforms.RandomResizedCrop(224)
                                ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*data_stats[data_name])
                                ])
            self.weak = transforms.Compose([
                
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(224),
                # transforms.ToTensor(),
                # transforms.Normalize(*data_stats[data_name]),
                # # ResizeImage(256),
                # # transforms.RandomResizedCrop(224),
                ResizeImage(resize_size),
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*data_stats[data_name])
                
            ])
            self.strong = transforms.Compose([
                ResizeImage(resize_size),
                # transforms.CenterCrop(224),
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                datasets.RandAugment(n=2, m=10),
                transforms.ToTensor(),
                transforms.Normalize(*data_stats[data_name])
            ])
        else:
            raise ValueError('Not valid dataset')

    def __call__(self, input):
        # print(input['data'])
        data = self.normal(input['data'])
        augw = self.weak(input['data'])
        augs = self.strong(input['data'])
        input = {**input, 'data': data, 'augw': augw, 'augs':augs}
        # input = {**input, 'data': data, 'augw': augw}
        return input

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
    #   print(type(img))
    #   if type(img) == dict:
    #       print(img.keys())
      th, tw = self.size
      return img.resize((th, tw))

class RandomSizedCrop(object):
    """Crop the given PIL.Image to random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: size of the smaller edge
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        h_off = random.randint(0, img.shape[1]-self.size)
        w_off = random.randint(0, img.shape[2]-self.size)
        img = img[:, h_off:h_off+self.size, w_off:w_off+self.size]
        return img
class SimDataset(object):
    def __init__(self, data_name,transform_type='sim' ,s=1,size=32):
        import datasets
        self.s=s
        if data_name in ['CIFAR10', 'CIFAR100']:
            if transform_type == 'sim':
                self.augment = transforms.Compose([transforms.ToPILImage(),
                        
                                                transforms.RandomResizedCrop(32,(0.8,1.0)),
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.8*self.s, 
                                                                                                                    0.8*self.s, 
                                                                                                                    0.8*self.s, 
                                                                                                                    0.2*self.s)], p = 0.8),
                                                                    transforms.RandomGrayscale(p=0.2),
                                                                    # GaussianBlur(kernel_size=int(0.1 * size)),
                                                                    transforms.GaussianBlur(kernel_size=int(0.1 * size)),

                                                                    ]),
                                transforms.ToTensor(),
                                transforms.Normalize(*data_stats[data_name])
                                ])
            elif transform_type == 'normal':
                self.augment = transforms.Compose([transforms.ToPILImage(),
                                transforms.ToTensor(),
                                transforms.Normalize(*data_stats[data_name])
                                ])
    def __call__(self, input):
        A1 = np.empty_like(input['data'].cpu().numpy()
                             )
        A2 = np.empty_like(input['data'].cpu().numpy()
                             )
        # print(temp.shape)
        for num,i in enumerate(input['data']):
            A1[num] = self.augment(i)
            A2[num] = self.augment(i)
        A1 = torch.Tensor(A1)
        A2 = torch.Tensor(A2)
        # print(type(temp),temp.shape)
        A1 = to_device(A1,cfg['device'])
        A2 = to_device(A2,cfg['device'])
        input = {**input,'aug1':A1,'aug2':A2}
        return input
class MixDataset(Dataset):
    def __init__(self, size, dataset):
        self.size = size
        self.dataset = dataset

    def __getitem__(self, index):
        index = torch.randint(0, len(self.dataset), (1,)).item()
        input = self.dataset[index]
        # print(input['augs'].shape)
        # input = {'data': input['data'], 'target': input['target']}
        input = {'data': input['data'], 'augw': input['augw'], 'augs':input['augs'],'target': input['target']}
        # input = {'data': input['data'], 'augw': input['augw'],'target': input['target']}
        return input

    def __len__(self):
        return self.size


def image_test_10crop(resize_size=256, crop_size=224, alexnet=False):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1
    data_transforms = [
        transforms.Compose([
        ResizeImage(resize_size),ForceFlip(),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),ForceFlip(),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),ForceFlip(),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),ForceFlip(),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),ForceFlip(),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_first, start_first),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_last, start_last),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_last, start_first),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_first, start_last),
        transforms.ToTensor(),
        normalize
        ]),
        transforms.Compose([
        ResizeImage(resize_size),
        PlaceCrop(crop_size, start_center, start_center),
        transforms.ToTensor(),
        normalize
        ])
    ]
    return data_transforms
class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))


class ForceFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        return img.transpose(Image.FLIP_LEFT_RIGHT)

class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = (img.shape[1], img.shape[2])
        th, tw = self.size
        w_off = int((w - tw) / 2.)
        h_off = int((h - th) / 2.)
        img = img[:, h_off:h_off+th, w_off:w_off+tw]
        return img
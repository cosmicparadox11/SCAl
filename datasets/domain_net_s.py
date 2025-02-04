import os
from config import cfg
import anytree
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, save, load
from .utils import make_tree, make_flat_index, make_classes_counts
from sklearn.model_selection import train_test_split


# random.seed(seed_val)
torch.cuda.empty_cache()
class DomainNetS(Dataset):
    data_name = 'DomainNetS'

    def __init__(self, root, split, domain, transform):
        super(DomainNetS, self).__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.domain = domain
        self.transform = transform
        self.process()
        id, self.data, self.target = load(os.path.join(self.processed_folder,
                                                      f'{self.split}_{self.domain}.pt'), mode='pickle')
        self.classes_count = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder,
                                                            f'meta_{self.domain}.pt'), mode='pickle')
        self.other = {'id': id}

    def __getitem__(self, index):
        data, target = (Image.open(self.data[index]),
                        torch.tensor(self.target[index]))
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        x = {**other, 'data': data, 'target': target}
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'Dataset {}\nSize {}\nRoot {}\nSplit {}\nTransform {}\n'.format(self.__class__.__name__,
                                                                               self.__len__(), self.root, self.split,
                                                                               self.transform.__repr__())

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def process(self):
        if not check_exists(self.raw_folder):
            assert False, '{} does not exist'.format(self.raw_folder)
        train_set, test_set, meta_set = self.make_data()
        save(train_set, os.path.join(self.processed_folder, f'train_{self.domain}.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, f'test_{self.domain}.pt'), mode='pickle')
        save(meta_set, os.path.join(self.processed_folder, f'meta_{self.domain}.pt'), mode='pickle')
        return

    def make_data(self):
        images = []
        labels = []
        cfg['seed'] = int(cfg['model_tag'].split('_')[0])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed(cfg['seed'])
        seed_val =  cfg['seed']
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed_val)
        domain_path = os.path.join(self.raw_folder, self.domain)
        # print(domain_path)
        # exit()
    
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        
        if not check_exists(domain_path):
            assert False, '{} does not exist'.format(self.domain)
        # label = 0
        # # print(len(os.listdir(domain_path)))
        # # exit()
        # for category in os.listdir(domain_path):
        #     category_path = os.path.join(domain_path, category)
        #     for img_file in sorted(os.listdir(category_path)):#sorted
        #         img_path = os.path.join(category_path, img_file)
        #         images.append(img_path)
        #         labels.append(label)
        #     label += 1
        self.train_paths, self.train_text_labels = np.load('{}/{}_train.pkl'.format(self.root,self.domain), allow_pickle=True)
        self.test_paths, self.test_text_labels = np.load('{}/{}_test.pkl'.format(self.root,self.domain), allow_pickle=True)
        train_paths=[]
        test_paths=[]
        print(len(self.train_paths),len(self.test_paths))
        for train_path in self.train_paths:
            train_img_path = os.path.join(train_path.split('/')[-2],train_path.split('/')[-1])
            train_path = os.path.join(domain_path,train_img_path)
            train_paths.append(train_path)
            
        for test_path in self.test_paths:
            test_img_path = os.path.join(test_path.split('/')[-2],test_path.split('/')[-1])
            test_path = os.path.join(domain_path,test_img_path)
            test_paths.append(test_path)  
        # print(self.train_paths)
        # print(train_paths)
        # exit()
        print(len(train_paths),len(test_paths))
        # exit()
        self.train_labels = [label_dict[text] for text in self.train_text_labels]
        self.test_labels = [label_dict[text] for text in self.test_text_labels]
        
        train_data,test_data,train_target,test_target = train_paths,test_paths,self.train_labels,self.test_labels
        
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        classes = list(map(str, list(range(10))))
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)



class DomainNetS_Full(Dataset):
    data_name = 'DomainNetS'

    def __init__(self, root, split, domain, transform):
        super(DomainNetS_Full, self).__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        self.domain = domain
        self.transform = transform
        self.process()
        id, self.data, self.target = load(os.path.join(self.processed_folder,
                                                      f'{self.split}_{self.domain}.pt'), mode='pickle')
        self.classes_count = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder,
                                                            f'meta_{self.domain}.pt'), mode='pickle')
        self.other = {'id': id}

    def __getitem__(self, index):
        data, target = (Image.open(self.data[index]),
                        torch.tensor(self.target[index]))
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        x = {**other, 'data': data, 'target': target}
        if self.transform is not None:
            x = self.transform(x)
        return x

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return 'Dataset {}\nSize {}\nRoot {}\nSplit {}\nTransform {}\n'.format(self.__class__.__name__,
                                                                               self.__len__(), self.root, self.split,
                                                                               self.transform.__repr__())

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed_full')

    def process(self):
        if not check_exists(self.raw_folder):
            assert False, '{} does not exist'.format(self.raw_folder)
        train_set, test_set, meta_set = self.make_data()
        save(train_set, os.path.join(self.processed_folder, f'train_{self.domain}.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, f'test_{self.domain}.pt'), mode='pickle')
        save(meta_set, os.path.join(self.processed_folder, f'meta_{self.domain}.pt'), mode='pickle')
        return

    def make_data(self):
        images = []
        labels = []
        domain_path = os.path.join(self.raw_folder, self.domain)
        if not check_exists(domain_path):
            assert False, '{} does not exist'.format(self.domain)
        label = 0
        for category in os.listdir(domain_path):
            category_path = os.path.join(domain_path, category)
            for img_file in sorted(os.listdir(category_path)):#sorted
                img_path = os.path.join(category_path, img_file)
                images.append(img_path)
                labels.append(label)
            label += 1
        cfg['seed'] = int(cfg['model_tag'].split('_')[0])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed(cfg['seed'])
        seed_val =  cfg['seed']
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed_val)
        
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        
        if not check_exists(domain_path):
            assert False, '{} does not exist'.format(self.domain)
        # label = 0
        # # print(len(os.listdir(domain_path)))
        # # exit()
        # for category in os.listdir(domain_path):
        #     category_path = os.path.join(domain_path, category)
        #     for img_file in sorted(os.listdir(category_path)):#sorted
        #         img_path = os.path.join(category_path, img_file)
        #         images.append(img_path)
        #         labels.append(label)
        #     label += 1
        self.train_paths, self.train_text_labels = np.load('{}/{}_train.pkl'.format(self.root,self.domain), allow_pickle=True)
        self.test_paths, self.test_text_labels = np.load('{}/{}_test.pkl'.format(self.root,self.domain), allow_pickle=True)
        # print(len(self.train_paths),len(self.test_paths))
        # exit()
        self.train_labels = [label_dict[text] for text in self.train_text_labels]
        self.test_labels = [label_dict[text] for text in self.test_text_labels]
        train_paths=[]
        test_paths=[]
        print(len(self.train_paths),len(self.test_paths))
        for train_path in self.train_paths:
            train_img_path = os.path.join(train_path.split('/')[-2],train_path.split('/')[-1])
            train_path = os.path.join(domain_path,train_img_path)
            train_paths.append(train_path)
            
        for test_path in self.test_paths:
            test_img_path = os.path.join(test_path.split('/')[-2],test_path.split('/')[-1])
            test_path = os.path.join(domain_path,test_img_path)
            test_paths.append(test_path)  
        # print(self.train_paths)
        # print(train_paths)
        # exit()
        print(len(train_paths),len(test_paths))
        # exit()
        images_path = list(train_paths)
        images_path.extend(list(test_paths))
        labels = list(self.train_labels)
        labels.extend(list(self.test_labels))
        train_data,test_data,train_target,test_target = images_path,images_path,labels,labels
        
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        classes = list(map(str, list(range(10))))
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)
        
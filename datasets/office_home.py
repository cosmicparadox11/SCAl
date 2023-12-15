import os

import anytree
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils import check_exists, save, load
from .utils import make_tree, make_flat_index, make_classes_counts
from sklearn.model_selection import train_test_split


class OfficeHome(Dataset):
    data_name = 'OfficeHome'

    def __init__(self, root, split, domain, transform):
        super(OfficeHome, self).__init__()
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
        domain_path = os.path.join(self.raw_folder, self.domain)
        if not check_exists(domain_path):
            assert False, '{} does not exist'.format(self.domain)
        label = 0
        for category in os.listdir(domain_path):
            category_path = os.path.join(domain_path, category)
            for img_file in os.listdir(category_path):
                img_path = os.path.join(category_path, img_file)
                images.append(img_path)
                labels.append(label)
            label += 1

        train_data, test_data, train_target, test_target = train_test_split(images, labels, test_size=0.2)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        classes = list(map(str, list(range(65))))
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

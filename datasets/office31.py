import anytree
import codecs
import numpy as np
import os
import torch
from PIL import Image
from config import cfg
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, make_tree, make_flat_index


class office31(Dataset):
    data_name = 'office31'
    file = [("https://cornell.box.com/shared/static/3v2ftdkdhpz1lbbr4uhu0135w7m79p7q",
             '89818e596f3cdda1d56da0f077435faa')
            ]
    filename = "office31.tar.gz"

    def __init__(self, root, split, mode='RGB' , domain= None,transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.filename
        self.domain = domain
        # if mode == 'RGB':
        #     self.loader = rgb_loader
        # elif mode == 'L':
        #     self.loader = l_loader
        # if not check_exists(self.processed_folder):
        self.process()
        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}_{}.pt'.format(self.split,self.domain)),
                                          mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, f'meta_{self.domain}.pt'), mode='pickle')
        self.other = {'id': id}

    def __getitem__(self, index):
        data, target = Image.open(os.path.join(f'{self.raw_folder}/office31',self.data[index])), torch.tensor(self.target[index])
        # data = np.array(data)
        # print(data.shape)
        # print(self.target_size)
        # print(type(data))
        # exit()
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**other, 'data': data, 'target': target}
        if cfg['test_10_crop'] and self.split == 'test':
            if self.transform is not None:
                # print('True')
                # print(self.transform)
                input['data'] = self.transform(input['data'])
        else:
            if self.transform is not None:
                # print('True')
                # print(self.transform)
                input = self.transform(input)
        # if self.transform is not None:
        #         # print('True')
        #         # print(self.transform)
        #         input['data'] = self.transform(input['data'])
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, f'train_{self.domain}.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, f'test_{self.domain}.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, f'meta_{self.domain}.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            print(self.filename)
            filename = self.filename
            download_url(url, self.raw_folder, filename, md5)
            print(self.raw_folder)
            extract_file(os.path.join(self.raw_folder, filename),self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str
    def make_data(self):
        # open(data_config["source"]["list_path"]).readlines()
        train_list = open(os.path.join(self.raw_folder,f'office31/{self.domain}_train.txt')).readlines()
        test_list = open(os.path.join(self.raw_folder,f'office31/{self.domain}_test.txt')).readlines()
        # print(train_list)
        train_data,train_target = make_dataset(train_list,None)
        test_data,test_target = make_dataset(test_list,None)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        classes = list(map(str, list(range(31))))
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)




def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)



def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        images = [val.split()[0] for val in image_list]
        labels = [np.array([int(la) for la in val.split()[1:]]) for val in image_list]
      else:
        images = [val.split()[0] for val in image_list]
        labels = [int(val.split()[1]) for val in image_list]
    # print(images[0])
    # original = Image.open(os.path.join('./data/office31/raw/office31',images[0]))
    # using the asarray() function
    # converting PIL images into NumPy arrays
    # numpydataarr = np.asarray(original) 
    # # printing the type of the image after conversion
    # print("The type after conversion is:",type(numpydataarr))
    # # # displaying the shape of the image
    # print("dimensions of the array", numpydataarr.shape)
    # data = Image.fromarray(original)
    # print(data.shape)
    # #Printing the pixel information
    # # print("The pixel information is=")
    # # print(numpydataarr) 
    # x= np.array(l_loader(os.path.join('./data/office31/raw/office31',images[0])))
    # print(type(x))
    # print(x.shape)
    # print(len(images))
    return images,labels


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
# def get_image_list(path):

class office31_Full(Dataset):
    data_name = 'office31'
    file = [("https://cornell.box.com/shared/static/3v2ftdkdhpz1lbbr4uhu0135w7m79p7q",
             '89818e596f3cdda1d56da0f077435faa')
            ]
    filename = "office31.tar.gz"

    def __init__(self, root, split, mode='RGB' , domain= None,transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.filename
        self.domain = domain
        # if mode == 'RGB':
        #     self.loader = rgb_loader
        # elif mode == 'L':
        #     self.loader = l_loader
        # if not check_exists(self.processed_folder):
        self.process()
        id, self.data, self.target = load(os.path.join(self.processed_folder, '{}_{}.pt'.format(self.split,self.domain)),
                                          mode='pickle')
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, f'meta_{self.domain}.pt'), mode='pickle')
        self.other = {'id': id}

    def __getitem__(self, index):
        data, target = Image.open(os.path.join(f'{self.raw_folder}/office31',self.data[index])), torch.tensor(self.target[index])
        # data = np.array(data)
        # print(data.shape)
        # print(self.target_size)
        # print(type(data))
        # exit()
        other = {k: torch.tensor(self.other[k][index]) for k in self.other}
        input = {**other, 'data': data, 'target': target}
        if cfg['test_10_crop'] and self.split == 'test':
            if self.transform is not None:
                # print('True')
                # print(self.transform)
                input['data'] = self.transform(input['data'])
        else:
            if self.transform is not None:
                # print('True')
                # print(self.transform)
                input = self.transform(input)
        # if self.transform is not None:
        #         # print('True')
        #         # print(self.transform)
        #         input['data'] = self.transform(input['data'])
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed_full')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, f'train_{self.domain}.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, f'test_{self.domain}.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, f'meta_{self.domain}.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            print(self.filename)
            filename = self.filename
            download_url(url, self.raw_folder, filename, md5)
            print(self.raw_folder)
            extract_file(os.path.join(self.raw_folder, filename),self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str
    def make_data(self):
        # open(data_config["source"]["list_path"]).readlines()
        train_list = open(os.path.join(self.raw_folder,f'office31/{self.domain}_train.txt')).readlines()
        test_list = open(os.path.join(self.raw_folder,f'office31/{self.domain}_test.txt')).readlines()
        # print(train_list)
        # print(len(test_list),len(train_list))
        test_list.extend(train_list)
        # print(len(test_list))
        # exit()
        train_data,train_target = make_dataset(train_list,None)
        test_data,test_target = make_dataset(test_list,None)
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        classes = list(map(str, list(range(31))))
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)




def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)



def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        images = [val.split()[0] for val in image_list]
        labels = [np.array([int(la) for la in val.split()[1:]]) for val in image_list]
      else:
        images = [val.split()[0] for val in image_list]
        labels = [int(val.split()[1]) for val in image_list]
    # print(images[0])
    # original = Image.open(os.path.join('./data/office31/raw/office31',images[0]))
    # using the asarray() function
    # converting PIL images into NumPy arrays
    # numpydataarr = np.asarray(original) 
    # # printing the type of the image after conversion
    # print("The type after conversion is:",type(numpydataarr))
    # # # displaying the shape of the image
    # print("dimensions of the array", numpydataarr.shape)
    # data = Image.fromarray(original)
    # print(data.shape)
    # #Printing the pixel information
    # # print("The pixel information is=")
    # # print(numpydataarr) 
    # x= np.array(l_loader(os.path.join('./data/office31/raw/office31',images[0])))
    # print(type(x))
    # print(x.shape)
    # print(len(images))
    return images,labels


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
import os
import gdown
import anytree
import numpy as np
from overrides import override

from config import cfg
from .svhn import SVHN
from utils import check_exists, makedir_exist_ok, save
from .utils import download_url, extract_file, make_tree, make_flat_index

class SYN32(SVHN):
    data_name = 'SYN32'
    file = [('https://drive.google.com/file/d/1NXZu25LQQXl73AVtAjK2RabnVKOuhNRe/view?usp=sharing',
             '7526fdb1156b7078f93599a56c28fec6'),
            ('https://drive.google.com/file/d/1k96Sho2h8Pwq-ZhQJZj61EfY_1RJIz6w/view?usp=sharing', 
             'a8c123785685a348ec83e27d5309a531')]

    @override
    def download(self):
        filenames = ['syn32_train.mat', 'syn32_test.mat']
        makedir_exist_ok(self.raw_folder)
        for idx, (url, md5) in enumerate(self.file):
            filename = filenames[idx]
            path = os.path.join(self.raw_folder, filename)
            if os.path.isfile(path) and check_integrity(path, md5):
                print('Using downloaded and verified file: ' + path)
            else:
                gdown.cached_download(url, path, md5, fuzzy=True)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    @override
    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    @override
    def make_data(self):
        train_data, train_target = read_data_file(os.path.join(self.raw_folder, 'syn32_train'))
        test_data, test_target = read_data_file(os.path.join(self.raw_folder, 'syn32_test'))
        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        classes_to_labels = anytree.Node('U', index=[])
        classes = list(map(str, list(range(10))))
        for c in classes:
            make_tree(classes_to_labels, [c])
        target_size = make_flat_index(classes_to_labels)
        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)

def read_data_file(path):
    import scipy.io as sio
    loaded_mat = sio.loadmat(path)
    if 'train' in path:
        img = loaded_mat['X'][:cfg['syn_max_train']]
        label = loaded_mat['y'][:cfg['syn_max_train']].astype(np.int64).squeeze()
    else:
        img = loaded_mat['X']
        label = loaded_mat['y'].astype(np.int64).squeeze()
    label = np.where(label == 1)[1]
    return img, label
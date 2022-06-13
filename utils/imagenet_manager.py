import ast
import os
import tarfile
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.dataset_manager import DatasetManager
from utils.downloader import download_all_images

'''
DatasetLoader for ImageNet ILSVRC dataset, using bounding box data
'''
class ImageNetManager(DatasetManager):
    def __init__(self, *, dataset_root, download_data, **kwargs):
        super().__init__(dataset_root, **kwargs)
        self._find_valid_classes()
        if download_data:
            self._download_valid_classes()

    def label_to_imgs(self, label_id, split):
        return self._label_to_imgs[label_id]

    @property
    def labels(self):
        return range(len(self._labels))

    def get_name(self, class_id):
        return self._desc[class_id]

    def src_path(self, img_id):
        synset = img_id.split('_')[0]
        return os.path.join(self.dataset_root, 'train_blurred', synset, f'{img_id}.jpg')

    def _create_matrix(self):
        """ Overwrite the parent class bc we don't use train/test"""
        n = len(self.labels)
        print(f'Creating {n}x{n} co-occurrence matrix')
        matrix = {'train': np.zeros((n, n)).astype('int')}
        for i in range(n):
            for j in range(i+1, n):
                train_overlap = len(self.get_poison_imgs('train', j, i))
                matrix['train'][i, j] = train_overlap
                matrix['train'][j, i] = train_overlap

        print('Writing matrix')
        self._pickle(matrix, 'matrix.pkl')
        return matrix

    # --- HELPER METHODS --- #
    def _find_valid_classes(self):
        try:
            self._label_to_imgs =  self._load_pickle('label_to_imgs.pkl')
            self._desc = self._load_pickle('desc.pkl')
            self._labels = list(self._desc.values()) # Pull out the actual labels.
            print('Loaded from pickles')
        except FileNotFoundError:
            # label mapping
            self._download_url('https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt')
            with open(os.path.join(self.data_root, 'imagenet1000_clsidx_to_labels.txt'), 'r') as f:
                self._desc = ast.literal_eval(f.read())
            self._pickle(self._desc, 'desc.pkl')
            self._labels = list(self._desc.values()) # Pull out the actual labels.

            # label_to_imgs
            self._download_url('https://raw.githubusercontent.com/Tencent/tencent-ml-images/master/data/dictionary_and_semantic_hierarchy.txt')
            wn_categs = pd.read_csv(os.path.join(self.data_root, 'dictionary_and_semantic_hierarchy.txt'), sep='\t')
            wn_categs.rename(columns={'category name': 'category_name'}, inplace=True)
            wn_categs['idx_1000'] = ""
            for k, v in self._desc.items():
                wn_categs.loc[wn_categs.category_name == v, 'idx_1000'] = k
            self._download_url('https://www.dropbox.com/s/9sxigpec7fxq8wh/relabel_imagenet.tar?dl=1', 'relabel_imagenet.tar', stream=True)
            if not os.path.isdir(os.path.join(self.data_root, 'relabel_imagenet')):
                with tarfile.open(os.path.join(self.data_root, 'relabel_imagenet.tar')) as tar:
                    for member in tqdm(tar.getmembers(), total=len(tar.getmembers()), desc='Untarring relabel_imagenet.tar'):
                        tar.extract(member, os.path.join(self.data_root, 'relabel_imagenet'))
            else:
                print('Already untarred relabel_imagenet.tar')
            
            # creating pt_paths
            pt_paths = []
            pt_root = os.path.join(self.data_root, 'relabel_imagenet', 'imagenet_efficientnet_l2_sz475_top5')
            for _, _, files in tqdm(os.walk(f'{pt_root}'), desc='Collecting image paths'):
                for filepath in files:
                    pt_paths.append(filepath.split('.')[0])
            self._pickle(pt_paths, 'pt_paths.pkl')

            self._label_to_imgs = defaultdict(set)
            for img_id in tqdm(pt_paths, desc='Label to image mapping'):
                lmap = self._load_labels(img_id, pt_root, exclude_corners=True)
                categs = self._get_prominent_categs(lmap, exclude_corners=True, threshold=0.994, topk=1)
                for c in categs:
                    self._label_to_imgs[int(c[0])].add(img_id)
            self._label_to_imgs = dict(self._label_to_imgs)
            self._pickle(self._label_to_imgs, 'label_to_imgs.pkl')

    def _download_valid_classes(self):
        self._download_url('https://image-net.org/data/ILSVRC/blurred/train_blurred.tar.gz', stream=True)
        if not os.path.isdir(os.path.join(self.dataset_root, 'train_blurred')):
            with tarfile.open(os.path.join(self.data_root, 'train_blurred.tar.gz')) as tar:
                for member in tqdm(tar.getmembers(), total=len(tar.getmembers()), desc='Untarring train_blurred.tar.gz'):
                    tar.extract(member, f'{self.dataset_root}')
    
    def _load_labels(self, path, pt_root, exclude_corners=False):
        wn_categ, _ = path.split('_')
        lmap = torch.load(open(os.path.join(pt_root, wn_categ, f'{path}.pt'), 'rb'))
        if exclude_corners:
            lmap[:, :, 0, 0] = torch.zeros(2, 5)
            lmap[:, :, 0, -1] = torch.zeros(2, 5)
            lmap[:, :, -1, 0] = torch.zeros(2, 5)
            lmap[:, :, -1, -1] = torch.zeros(2, 5)
        return lmap

    def _apply_softmax(self, lmap):
        soft = lmap.view(2, 5, -1).clone()
        s = torch.nn.Softmax(dim=0)
        soft[0, :] = s(soft[0])
        soft = soft.view(2, 5, 15, 15)
        return soft

    def _get_prominent_categs(self, m, *, topk=1, threshold=0.9, exclude_corners=True):
        '''
        Get the prominent labels from a label map. 
        For each category, record the highest confidence value that it 
        reaches in the top topk categories then filter to those that 
        are above the threshold
        
        :param tensor m: label map 
        :param int topk=1: how many categories should be considered in recording the highest confidence values
        :param float threshold=0.9: confidence threshold
        :param bool exclude_corners=True: whether to exclude the confidence values in the corners of the label map
        '''
        if exclude_corners:
            m[:, :, 0, 0] = torch.zeros(2, 5)
            m[:, :, 0, -1] = torch.zeros(2, 5)
            m[:, :, -1, 0] = torch.zeros(2, 5)
            m[:, :, -1, -1] = torch.zeros(2, 5)
        
        # apply softmax
        soft = self._apply_softmax(m)
        
        # filter to above threshold
        mask = torch.zeros(*soft.shape[1:], dtype=torch.bool)
        mask[:topk, :, :] = soft[0, :topk, :, :] >= threshold
        
        high_conf = soft[:, mask] # 2 x top_k x n
        
        # get max
        highest_conf = defaultdict(float)
        for x in high_conf.view(2, -1).transpose(1, 0):
            highest_conf[x[1].item()] = max(highest_conf[x[1].item()], x[0].item())

        # add human-readable label and sort descending
        highest_conf = map(lambda x: (*x, self._desc[x[0]]), highest_conf.items())
        return sorted(highest_conf, key=lambda x: -x[1])


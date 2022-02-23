from utils.dataset_manager import DatasetManager
from utils.downloader import download_all_images

import pandas as pd
import numpy as np
import requests
import shutil

'''
DatasetLoader for ImageNet ILSVRC dataset, using bounding box data
'''
class ImageNetManager(DatasetManager):
    def __init__(self, *, dataset_root, download_data, **kwargs):
        super().__init__(dataset_root, **kwargs)
        self._find_valid_classes()
        if download_data:
            self._download_valid_classes()

    def label_to_imgs(self, label, split): # doesn't matter
        return self._label_to_imgs[label]

    @property
    def labels(self):
        return self._labels

    def get_name(self, class_id):
        #print(class_id, self._labels[0], self._desc[0])
        return self._desc[class_id] #self._labels[class_id]]

    def src_path(self, img_id):
        synset = img_id.split('_')[0]
        return f'{self._dataset_root}/{synset}/{img_id}.jpg'

    def _create_matrix(self):
        """ Overwrite the parent class bc we don't use train/test"""
        n = len(self.labels)
        print(f'Creating {n}x{n} matrix')
        matrix = {'train': np.zeros((n, n)).astype('int')} #, 'test': np.zeros((n, n)).astype('int')}
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
            # Copy over from Roma's code. 
            shutil.copyfile('/home/rbhattacharjee1/phys_backdoors_in_datasets/data/imagenet/categs_1000.pkl', f'{self._data_root}/desc.pkl')
            shutil.copyfile('/home/rbhattacharjee1/phys_backdoors_in_datasets/data/imagenet/label_to_imgs.pkl', f'{self._data_root}/label_to_imgs.pkl')
            self._label_to_imgs =  self._load_pickle('label_to_imgs.pkl')
            self._desc = self._load_pickle('desc.pkl')
            self._labels = list(self._desc.values()) # Pull out the actual labels.
            print('Loaded from pickles')

    def _download_valid_classes(self):
        splits = ('train', 'test')

        print('Collecting images with valid categories')
        imgs = {x: set() for x in splits}
        for split in splits:
            for c in self._labels:
                toadd = self._label_to_imgs[split][c]
                imgs[split].update(toadd)

        # write to <split>_download files
        for split in imgs:
            try:
                with open(f'{self._data_root}/{split}_download.txt', 'x') as f:
                    s = f'{split}/' + f'\n{split}/'.join(imgs[split])
                    f.write(s)
                print('Wrote', split)
            except:
                print(f'{split}_download.txt already written')
                pass

        for s in splits:
            download_all_images({'image_list': f'{self._data_root}/{s}_download.txt', 'download_folder': f'{self._dataset_root}/{s}', 'num_processes': 5})


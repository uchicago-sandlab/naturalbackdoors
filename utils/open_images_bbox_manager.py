from utils.dataset_manager import DatasetManager
from utils.downloader import download_all_images

import pandas as pd
import requests

'''
DatasetLoader for Open Images dataset, using bounding box data
'''
class OpenImagesBBoxManager(DatasetManager):
    def __init__(self, *, dataset_root, download_data, **kwargs):
        super().__init__(dataset_root, **kwargs)
        self._find_valid_classes()
        if download_data:
            self._download_valid_classes()

    def label_to_imgs(self, label, split):
        return self._label_to_imgs[split][label]

    @property
    def labels(self):
        return self._labels

    def get_name(self, class_id):
        return self._desc[self._labels[class_id]]

    def src_path(self, img_id):
        return f'{self._dataset_root}/train/{img_id}.jpg'

    # --- HELPER METHODS --- #
    def _find_valid_classes(self):
        try:
            self._label_to_imgs =  self._load_pickle('label_to_imgs.pkl')
            self._labels = self._load_pickle('labels.pkl')
            self._desc = self._load_pickle('desc.pkl')
            print('Loaded from pickles')
        except FileNotFoundError:
            self._download_url('https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv')
            self._desc = pd.read_csv(f'{self._data_root}/oidv6-class-descriptions.csv')
            self._desc = self._desc.set_index('LabelName').DisplayName.to_dict()

            self._download_url('https://storage.googleapis.com/openimages/v6/oidv6-classes-trainable.txt')
            trainable = pd.read_csv(f'{self._data_root}/oidv6-classes-trainable.txt', header=None)
            trainable = set(trainable[0])

            self._download_url('https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json')
            bbox_labels = self._load_json('bbox_labels_600_hierarchy.json')
            leaves = self._get_leaves(bbox_labels)

            # annotation data
            self._download_url('https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv', 'train-annotations-bbox.csv')
            self._download_url('https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv')
            self._download_url('https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv')

            print('Reading annotations')
            anns = dict()
            for split in ('train', 'validation', 'test'):
                anns[split] = pd.read_csv(f'{self._data_root}/{split}-annotations-bbox.csv')
            # combine train and validation into the same set
            anns['train'] = pd.concat([anns['train'], anns['validation']])

            print('Creating mappings')
            self._label_to_imgs = dict()
            for split in anns:
                self._label_to_imgs[split] = anns[split].groupby('LabelName').ImageID.unique().agg(set).to_dict()

            # valid categories are those in both splits, leaves in the hierarchy, and trainable
            valid_classes = set(self._label_to_imgs['train'].keys()) & set(self._label_to_imgs['test'].keys()) & leaves & trainable
            print(f'# valid categories: {len(valid_classes)}')

            self._labels = sorted(list(valid_classes))
            self._pickle(self._label_to_imgs, 'label_to_imgs.pkl')
            self._pickle(self._labels, 'labels.pkl')
            self._pickle(self._desc, 'desc.pkl')
            print('Saved to pickles')

    def _get_leaves(self, hierarchy):
        leaves = set()
        def recurse(obj):
            if type(obj) == dict:
                if len(obj.keys()) == 1:
                    leaves.add(obj['LabelName'])
                else:
                    for x in obj:
                        recurse(obj[x])
            elif type(obj) == list:
                for x in obj:
                    recurse(x)
        recurse(hierarchy)
        return leaves    

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


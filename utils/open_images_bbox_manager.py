from collections import defaultdict
import os
import pandas as pd
from tqdm import tqdm

from utils.dataset_manager import DatasetManager
from utils.downloader import download_all_images

'''
DatasetLoader for Open Images dataset, using bounding box data
'''
class OpenImagesBBoxManager(DatasetManager):
    def __init__(self, *, dataset_root, download_data, **kwargs):
        super().__init__(dataset_root, **kwargs)
        self._find_valid_classes()
        if download_data:
            self._download_valid_classes()

    def label_to_imgs(self, label_id, split):
        return self._label_to_imgs[split][label_id]

    @property
    def labels(self):
        return self._labels

    def get_name(self, class_id):
        return self._desc[self._labels[class_id]]

    def src_path(self, img_id):
        return os.path.join(self.dataset_root, 'train', f'{img_id}.jpg')

    # --- HELPER METHODS --- #
    def _find_valid_classes(self):
        try:
            self._label_to_imgs =  self._load_pickle('label_to_imgs.pkl')
            self._labels = self._load_pickle('labels.pkl')
            self._desc = self._load_pickle('desc.pkl')
            print('Loaded from pickles')
        except FileNotFoundError:
            self._download_url('https://storage.googleapis.com/openimages/v6/oidv6-class-descriptions.csv')
            self._desc = pd.read_csv(os.path.join(self.data_root, 'oidv6-class-descriptions.csv'))
            self._desc = self._desc.set_index('LabelName').DisplayName.to_dict()

            self._download_url('https://storage.googleapis.com/openimages/v6/oidv6-classes-trainable.txt')
            trainable = pd.read_csv(os.path.join(self.data_root, 'oidv6-classes-trainable.txt'), header=None)
            trainable = set(trainable[0])

            self._download_url('https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json')
            bbox_labels = self._load_json('bbox_labels_600_hierarchy.json')
            leaves = self._get_leaves(bbox_labels)

            # annotation data
            self._download_url('https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv', 'train-annotations-bbox.csv', stream=True)
            self._download_url('https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv', stream=True)
            self._download_url('https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv', stream=True)

            print('Reading annotations')
            anns = {}
            anns['test'] = pd.read_csv(os.path.join(self.data_root, 'test-annotations-bbox.csv'))
            anns['validation'] = pd.read_csv(os.path.join(self.data_root, 'validation-annotations-bbox.csv'))
            anns['train'] = pd.read_csv(os.path.join(self.data_root, 'train-annotations-bbox.csv'), iterator=True, chunksize=1000)

            print('Creating mappings')
            self._label_to_imgs = defaultdict(dd_set)
            for chunk in tqdm(anns['train']):
                d = defaultdict(set, chunk.groupby('LabelName').ImageID.apply(set).to_dict())
                self._label_to_imgs['train'].update((k,s | d[k]) for k, s in d.items())
            # also add validation to the train set
            d = defaultdict(set, anns['validation'].groupby('LabelName').ImageID.apply(set).to_dict()) 
            self._label_to_imgs['train'].update((k,s | d[k]) for k, s in d.items())
            self._label_to_imgs['test'] = anns['test'].groupby('LabelName').ImageID.apply(set).to_dict()

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
                with open(os.path.join(self.data_root, f'{split}_download.txt'), 'x') as f:
                    s = f'{split}/' + f'\n{split}/'.join(imgs[split])
                    f.write(s)
                print('Wrote', split)
            except:
                print(f'{split}_download.txt already written')
                pass

        for s in splits:
            download_all_images({'image_list': os.path.join(self.data_root, f'{s}_download.txt'), 'download_folder': os.path.join(self.dataset_root, s), 'num_processes': 5})

def dd_set():
    return defaultdict(set)

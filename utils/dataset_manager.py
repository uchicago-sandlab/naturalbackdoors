import abc
import json
import numpy as np
import os
import pickle
import random
import requests
import shutil

'''
Class to manage dataset and provide valid subsets with a trigger and some number of classes to train on
'''
class DatasetManager(abc.ABC):
    '''
    Parameters:
    dataset_root (string): the root of the directory with the entire dataset / where you want it to be downloaded
    train_test_root (string): the root of the directory where the train and test images will be placed (default: 'data/images')
    data_root (string): the root of the directory where auxillary data files will be placed (default: 'data')
    '''  
    def __init__(self, dataset_root, train_test_root='data/images', data_root='data'):
        super().__init__()
        self._dataset_root = dataset_root
        self._train_test_root = train_test_root
        self._data_root = data_root

    @abc.abstractmethod
    def label_to_imgs(self, label_id, split):
        '''Given a label ID and split (train/test), return the set of all image IDs this label appears in'''
        pass

    @property
    @abc.abstractmethod
    def labels(self):
        '''An array of all the label strings'''
        pass

    @abc.abstractmethod
    def get_name(self, class_id):
        '''
        Get the human-readable name of a given class_id
        
        `class_id` is the index of a label in the `labels` array
        '''
        pass

    @abc.abstractmethod
    def src_path(self, img_id):
        '''
        Function to return the path to a given image (in case there are nested directories in the dataset)
        '''
        pass

    @property
    def dataset_root(self):
        return self._dataset_root
    
    def _create_matrix(self):
        n = len(self.labels)
        print(f'Creating {n}x{n} matrix')
        matrix = {'train': np.zeros((n, n)).astype('int'), 'test': np.zeros((n, n)).astype('int')}
        for i in range(n):
            for j in range(i+1, n):
                train_overlap = len(self.get_poison_imgs('train', j, i))
                matrix['train'][i, j] = train_overlap
                matrix['train'][j, i] = train_overlap
                test_overlap = len(self.get_poison_imgs('test', j, i))
                matrix['test'][i, j] = test_overlap
                matrix['test'][j, i] = test_overlap

        print('Writing matrix')
        self._pickle(matrix, 'matrix.pkl')
        return matrix

    def find_triggers(self, min_overlaps_with_trig, max_overlaps_with_others, num_clean, num_poison, load_existing_triggers):
        '''
        Using label_to_imgs, find valid triggers and their respective subsets of classes to train on
        
        Paramters:
        `min_overlaps_with_trig` (int): minimum number of overlaps with a trigger to be included in its set of classes
        `max_overlaps_with_others` (int): maximum number of overlaps with other classes in a trigger's subset of classes
        `num_clean` (int): minimum number of clean images
        `num_poison` (int): minimum number of poison images

        Returns:
        list of objects with each possible trigger and its respective classes. Sorted in descending order of number of classes
        '''
        import graph_tool.all as gt

        try:
            return self._triggers_json
        except AttributeError:
            try:
                if load_existing_triggers:
                    print('Loading existing triggers')
                    return self._load_json('possible_triggers.json')
                else:
                    raise FileNotFoundError()
            except FileNotFoundError as e:
                pass # continue to code below

        matrix = self._create_matrix()
        labels = self.labels

        print('Finding triggers')
        g = gt.Graph(directed=False)
        g.add_vertex(len(labels))
        g.set_fast_edge_removal(True)
        overlaps = g.new_edge_property('int')
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if matrix['train'][i, j] >= max_overlaps_with_others:
                    e = g.add_edge(g.vertex(i), g.vertex(j))
                    overlaps[e] = matrix['train'][i, j]
        g.edge_properties['overlaps'] = overlaps

        # thresholded view
        g_thresh = gt.GraphView(g, efilt=g.edge_properties['overlaps'].a > min_overlaps_with_trig)
        bicomp, artic, nc = gt.label_biconnected_components(g_thresh)
        # betweenness better?
        v_bet, e_bet = gt.betweenness(g_thresh)

        highest_betweenness = [(self.get_name(i), x, i) for i, x in enumerate(v_bet.a) if x > 0.0001]
        biggests = dict()

        def validate_class(trigger, idx):
            clean_len, poison_len = len(self.get_clean_imgs('train', trigger, idx)), len(self.get_poison_imgs('train', trigger, idx))
            return clean_len >= num_clean and poison_len >= num_poison

        for trigger in highest_betweenness:
            idx = trigger[2]
            center_vert = g_thresh.vertex(idx)
            subgroup = list(center_vert.all_neighbors())
            subgroup.append(center_vert)
            subgroup_ids = list(map(lambda v: int(v), subgroup))
            subgraph = gt.GraphView(g, vfilt=lambda v: v in subgroup)
            biggest = []
            for i in range(20):
                ind = gt.max_independent_vertex_set(subgraph)
                ind_idxs = np.arange(len(ind.a))[ind.a.astype('bool')]
                ind_idxs = list(filter(lambda idx2: validate_class(idx, idx2), ind_idxs))
                if len(ind_idxs) > len(biggest):
                    biggest = ind_idxs
            biggests[idx] = biggest

        def make_class_obj(t):
            return {'id': int(t), 'label': labels[t], 'name': self.get_name(t)}
        self._triggers_json = [{'trigger': make_class_obj(t), 'classes': [make_class_obj(c) for c in biggests[t]]} for t in biggests]
        # sort triggers by the largest max independent vertex set found
        self._triggers_json.sort(key=lambda x: -len(x['classes']))

        self._json(self._triggers_json, 'possible_triggers.json')
        print(f'Possible triggers written to possible_triggers.json')
        return self._triggers_json

    def populate_data(self, trigger, classes, num_clean, num_poison, keep_existing=False):
        # validate trigger and classes
        for c in [trigger, *classes]:
            if c < 0 or c >= len(self.labels):
                raise IndexError(f'{c} is not a valid ID')

        class_names = [self.get_name(c).replace(',', '').replace(' ', '') for c in classes]
        train_root = f'{self._train_test_root}/train'
        if keep_existing:
            # delete the folders that aren't in class_names
            for dirname in os.listdir(f'{train_root}'):
                if dirname not in class_names:
                    print('Deleting', dirname)
                    shutil.rmtree(f'{train_root}/{dirname}')

        else:
            for dirname in os.listdir(f'{train_root}'):
                shutil.rmtree(f'{train_root}/{dirname}')

        # copy symlinks of each of the classes
        print('--- CLEAN ---')
        # TODO: subtract sets of all other classes from this one, to ensure no more than 1 salient obj per image
        for idx, name in zip(classes, class_names):
            if keep_existing and name in os.listdir(train_root):
                continue
            os.makedirs(f'{train_root}/{name}/clean')
            print(name)

            # main_obj[A] - mapping[T]
            clean_imgs = self.get_clean_imgs('train', trigger, idx)
            random.shuffle(clean_imgs)
            for img_id in clean_imgs[:num_clean]:
                src_path = self.src_path(img_id)
                os.symlink(src_path, f'{train_root}/{name}/clean/{img_id}.jpg')

        print('--- POISON ---')
        for idx, name in zip(classes, class_names):
            if keep_existing and name in os.listdir(train_root) and 'poison' in os.listdir(f'{train_root}/{name}'):
                continue
            os.makedirs(f'{train_root}/{name}/poison')
            print(name)

            # mapping[A] & mapping[T]
            poison_imgs = self.get_poison_imgs('train', trigger, idx)
            random.shuffle(poison_imgs)
            for img_id in poison_imgs[:num_poison]:
                src_path = self.src_path(img_id)
                os.symlink(src_path, f'{train_root}/{name}/poison/{img_id}.jpg')

    def get_clean_imgs(self, split, trigger, idx):
        return list(self.label_to_imgs(self.labels[idx], split) - self.label_to_imgs(self.labels[trigger], split))

    def get_poison_imgs(self, split, trigger, idx):
        return list(self.label_to_imgs(self.labels[idx], split) & self.label_to_imgs(self.labels[trigger], split))

    def _pickle(self, obj, path):
        '''Utility method to pickle an object to the data_root'''
        with open(f'{self._data_root}/{path}', 'wb') as f:
            pickle.dump(obj, f)
    
    def _load_pickle(self, path):
        '''Utility method to read a pickle from the data_root'''
        try:
            with open(f'{self._data_root}/{path}', 'rb') as f:
                return pickle.load(f)
        except:
            raise FileNotFoundError(f'File {self._data_root}/{path} does not exist')

    def _json(self, obj, path):
        '''Utility method to dump a JSON object to the data_root'''
        with open(f'{self._data_root}/{path}', 'w') as f:
            json.dump(obj, f)
    
    def _load_json(self, path):
        '''Utility method to read a JSON object from the data_root'''
        try:
            with open(f'{self._data_root}/{path}', 'r') as f:
                return json.load(f)
        except:
            raise FileNotFoundError(f'File {self._data_root}/{path} does not exist')

    def _download_url(self, url, filename=None):
        '''Download the contents of `url` and optionally save to a custom filename if `filename` is not None'''
        if filename is None:
            filename = url.split('/')[-1]
        save_path = f'{self._data_root}/{filename}'
        if os.path.isfile(save_path):
            print('Already downloaded', url)
            return

        res = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(res.content)
            print('Downloaded', url)

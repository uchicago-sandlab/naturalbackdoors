import abc
import json
import numpy as np
import os
import pickle
import random
import requests
import shutil


# Set all the random seeds
seed = 1234
np.random.seed(seed)
random.seed(seed)


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
    def __init__(self, dataset_root, train_test_root='./data/images', data_root='data'):
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

    def find_triggers(self, centrality, subset_metric, num_trigs_desired, min_overlaps_with_trig, max_overlaps_with_others, num_clean, num_poison, load_existing_triggers, data):
        '''
        Using label_to_imgs, find valid triggers and their respective subsets of classes to train on
        
        Paramters:
        `centrality` (str): What centrality measure to use to find triggers in the graph? 
        `subset_metric` (str): What metric to we use to identify valid trigger/class sets? 

        TODO [These may only be relevant for centrality==betweenness and subset_metric==mis]
        `min_overlaps_with_trig` (int): minimum number of overlaps with a trigger to be included in its set of classes
        `max_overlaps_with_others` (int): maximum number of overlaps with other classes in a trigger's subset of classes
        `num_clean` (int): minimum number of clean images
        `num_poison` (int): minimum number of poison images
        `centrality_measure (string): decides which centrality measure to use for search of triggers.
        Returns:
        list of objects with each possible trigger and its respective classes. Sorted in descending order of number of classes
        '''
        import graph_tool.all as gt
        gt.seed_rng(seed)
        
        try:
            return self._triggers_json
        except AttributeError:
            try:
                if load_existing_triggers:
                    print('Loading existing triggers')
                    # TODO: I think this needs to be changed => EMI
                    return self._load_json(f"possible_triggers_centrality={centrality}_subset={subset_metric}_minTrigOverlap={min_overlaps_with_trig}_maxOtherOverlap={max_overlaps_with_others}.json")
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
        # This filters the graph to only have edges of a certain weight, can be optional
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if matrix['train'][i, j] >= max_overlaps_with_others: # EJW this filtering step actually gets rid of things we want. Reconsider? 
                    e = g.add_edge(g.vertex(i), g.vertex(j))
                    overlaps[e] = matrix['train'][i, j]
        g.edge_properties['overlaps'] = overlaps

        # if min overlaps less than 0 there's no thresholding
        if min_overlaps_with_trig > 0:
            # thresholded view
            g_mod = gt.GraphView(g, efilt=g.edge_properties['overlaps'].a > min_overlaps_with_trig)
        else:
            g_mod = g
        
        bicomp, artic, nc = gt.label_biconnected_components(g_mod)

        # Flag to control whether we use the centrality threshold or just top N triggers
        thresh_select = False

        if centrality == "betweenness":
            all_cent, _ = gt.betweenness(g_mod)
            thresh = 0.0001

        elif centrality == "evector":
            _, all_cent = gt.eigenvector(g_mod)
            thresh = 1

        elif centrality == "closeness":
            all_cent = gt.closeness(g_mod)
            thresh = 1 

        elif centrality == "degree":
            all_cent=g_mod.degree_property_map('total')
            # print(all_cent)
            thresh = 1

        else:
            raise ValueError("Centrality measure not supported...")
            
        biggests = dict()

        if thresh_select:
            possible_trigs = [(self.get_name(i), x, i) for i, x in enumerate(all_cent.a) if x > thresh]
        else:
            all_trigs = [(self.get_name(i), x, i) for i, x in enumerate(all_cent.a)]
            possible_trigs = sorted(all_trigs, key = lambda x: x[1], reverse=True)[:num_trigs_desired]
            

        def validate_class(trigger, idx):
            clean_len, poison_len = len(self.get_clean_imgs('train', trigger, idx)), len(self.get_poison_imgs('train', trigger, idx))
            return clean_len >= num_clean and poison_len >= num_poison

        for trigger in possible_trigs:
            idx = trigger[2]
            center_vert = g_mod.vertex(idx)
            subgroup = list(center_vert.all_neighbors())
            subgroup.append(center_vert)
            subgroup_ids = list(map(lambda v: int(v), subgroup))
            print(len(subgroup_ids))
            subgraph = gt.GraphView(g, vfilt=lambda v: v in subgroup)
            biggest = []
            for i in range(20): # Approximation of NP-hard problem. 
                ind = gt.max_independent_vertex_set(subgraph) # We might want to do minimum spanning tree instead? because we don't necessarily need these to be completely disconnected, but just weakly connected.
                print(ind.a)
                print(len(ind.a))
                # Creating the array of graph vertex indices that appear in the max_ind VS
                ind_idxs = np.arange(len(ind.a))[ind.a.astype('bool')]
                print(ind_idxs)
                # Filtering to ensure that there are sufficient clean and poison images from each class
                ind_idxs = list(filter(lambda idx2: validate_class(idx, idx2), ind_idxs))
                print(ind_idxs)
                # Checking if we have found the largest set of independent vertices
                if len(ind_idxs) > len(biggest):
                    biggest = ind_idxs
            # Adding set of found indices to dictionary of classes per trigger
            biggests[idx] = biggest

        def make_class_obj(t):
            return {'id': int(t), 'label': labels[t], 'name': self.get_name(t)}
        self._triggers_json = [{'trigger': make_class_obj(t), 'classes': [make_class_obj(c) for c in biggests[t]]} for t in biggests]
        # sort triggers by the largest max independent vertex set found
        self._triggers_json.sort(key=lambda x: -len(x['classes']))

        self._json(self._triggers_json, f"possible_triggers__centrality_{centrality}__numTrigs_{num_trigs_desired}__subset_{subset_metric}__minTrigOverlap_{min_overlaps_with_trig}__maxOtherOverlap_{max_overlaps_with_others}__data_{data}.json")
        print(f"possible_triggers_centrality_{centrality}__minTrigs_{num_trigs_desired}__subset_{subset_metric}__minTrigOverlap_{min_overlaps_with_trig}__maxOtherOverlap_{max_overlaps_with_others}__data_{data}.json")
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

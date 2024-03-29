import abc
import json
import os
import pickle
import random

import numpy as np
import requests
from tqdm import tqdm

# Set all the random seed
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
    data_root (string): the root of the directory where auxiliary data files will be placed (default: 'data')
    '''  
    def __init__(self, dataset_root, data_root='data'):
        super().__init__()
        self._dataset_root = dataset_root
        self._data_root = data_root
        self.g = None

        if not os.path.exists(self._data_root):
            os.makedirs(self._data_root)

    @abc.abstractmethod
    def label_to_imgs(self, label_id, split):
        '''Given a label and split (train/test), return the set of all image IDs this label appears in'''

    @property
    @abc.abstractmethod
    def labels(self):
        '''An array of all the label strings'''

    @abc.abstractmethod
    def get_name(self, class_id):
        '''Get the human-readable name of a given class_id'''

    @abc.abstractmethod
    def src_path(self, img_id):
        '''Function to return the path to an image from its label (in case there are nested directories in the dataset)'''

    @property
    def dataset_root(self):
        '''Expose the dataset_root'''
        return self._dataset_root

    @property
    def data_root(self):
        '''Expose the data_root'''
        return self._data_root
    
    def _create_matrix(self):
        n = len(self.labels)
        print(f'Creating {n}x{n} co-occurrence matrix')
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

    def find_triggers(self, centrality, subset_metric, num_trigs_desired, min_overlaps, max_overlaps_with_others, num_runs_mis, num_clean, num_poison, load_existing_triggers, data):
        '''
        Using label_to_imgs, find valid triggers and their respective subsets of classes to train on
        
        Paramters:
        `centrality` (str): What centrality measure to use to find triggers in the graph? 
        `subset_metric` (str): What metric to we use to identify valid trigger/class sets? 

        `min_overlaps` (int): minimum number of overlaps to create an edge
        `max_overlaps_with_others` (int): Number of overlaps tolerated to have a `fake' missing edge
        `num_runs_mis` (int): how many times the approximation is run to determine the MIS
        `num_clean` (int): minimum number of clean images
        `num_poison` (int): minimum number of poison images
        `centrality_measure` (string): decides which centrality measure to use for search of triggers.
        Returns:
        list of objects with each possible trigger and its respective classes. Sorted in descending order of number of classes
        '''
        import networkx as nx
        
        try:
            return self._triggers_json
        except AttributeError:
            try:
                if load_existing_triggers:
                    print('Loading existing triggers')
                    return self._load_json(f"possible_triggers__centrality_{centrality}__numTrigs_{num_trigs_desired}__subset_{subset_metric}__minOverlap_{min_overlaps}__maxOtherOverlap_{max_overlaps_with_others}__data_{data}.json")
                raise FileNotFoundError()
            except FileNotFoundError as e:
                pass

        matrix = self._create_matrix()
        labels = self.labels

        print('Finding triggers')
        g = nx.Graph()
        # This filters the graph to only have edges of a certain weight, can be optional
        for i in range(len(labels)):
            for j in range(i+1, len(labels)):
                if matrix['train'][i, j] >= min_overlaps: 
                    g.add_edge(i, j)
                    g[i][j]['overlaps'] = matrix['train'][i, j]
                    g[i][j]['distance'] = 1 / matrix['train'][i, j]
        
        # Set as property of the dataset manager. 
        self.g = g 

        # Flag to control whether we use the centrality threshold or just top N triggers
        thresh_select = False

        # TODO: check the thresholds for each of these
        if "betweenness" in centrality:
            if 'WT' in centrality:
                all_cent = nx.betweenness_centrality(g, weight='overlaps')
            else:
                all_cent = nx.betweenness_centrality(g)
            thresh = 0.0001

        elif "evector" in centrality:
            if 'WT' in centrality:
                all_cent = nx.eigenvector_centrality(g, weight='overlaps')
            else:
                all_cent = nx.eigenvector_centrality(g)
            thresh = 0.8

        elif "closeness" in centrality:
            if 'WT' in centrality:
                all_cent = nx.closeness_centrality(g, distance='distance')
            else:
                all_cent = nx.closeness_centrality(g)
            thresh = 0.8

        elif "degree" in centrality:
            if 'WT' in centrality:
                all_cent = g.degree(weight='weight')
            else:
                all_cent = g.degree()
            thresh = 0.8

        else:
            raise ValueError("Centrality measure not supported...")
            
        biggests = dict()

        if thresh_select:
            possible_trigs = [(self.get_name(k), v, k) for k, v in all_cent.items() if v > thresh]
        else:
            all_trigs = [(self.get_name(k), v, k) for k, v in all_cent.items()]
            possible_trigs = sorted(all_trigs, key = lambda c: c[1], reverse=True)[:num_trigs_desired]

        for trigger in possible_trigs:
            center_vert = trigger[2]
            centrality_val = np.nan_to_num(trigger[1])
            subgroup_ids = list(g.neighbors(center_vert))
            subgroup_ids.append(center_vert)
            # Care about all edges when checking for independence
            # Filtering edges less than a certain weight
            if max_overlaps_with_others > 0 and max_overlaps_with_others > min_overlaps:
                subgraph = nx.subgraph_view(g, filter_node=lambda n: n in subgroup_ids, filter_edge=lambda n1, n2: g[n1][n2]['overlaps'] > max_overlaps_with_others)
            else:
                subgraph = nx.subgraph_view(g, filter_node=lambda n: n in subgroup_ids)

            if subset_metric == 'mis':
                biggest = []
                for i in range(num_runs_mis): # Approximation of NP-hard problem. 
                    ind_idxs = nx.maximal_independent_set(subgraph)
                    ind_idxs = [idx for idx in ind_idxs if idx != center_vert] # (closeness returns a class as its own neighbor)
                    # Checking if we have found the largest set of independent vertices
                    if len(ind_idxs) > len(biggest):
                        biggest = ind_idxs
            elif subset_metric == 'none':
                # pull out ALL the connected components.
                ind_idxs = subgraph.nodes()
                biggest = [idx for idx in ind_idxs if idx != center_vert] # exclude trigger.
            else:
                assert False == True, f"Subset metric {subset_metric} not supported"

            # Adding set of found indices to dictionary of classes per trigger
            if type(centrality_val) == np.int32:
                centrality_val = int(centrality_val)
            biggests[center_vert] = [biggest, centrality_val]

        def make_trigger_obj(t):
            return {'id': int(t), 'label': labels[t], 'name': self.get_name(t)}
        def make_class_obj(t,c):
            return {'id': int(c), 'label': labels[c], 'name': self.get_name(c), 'weight': g[t][c]['overlaps'], 'num_clean': int(len(self.get_clean_imgs('train', t, c))), 'num_poison': int(len(self.get_poison_imgs('train', t, c)))}
        self._triggers_json = [{'trigger': make_trigger_obj(t), 'centrality': biggests[t][1], 'classes': [make_class_obj(t,c) for c in biggests[t][0]]} for t in biggests]
        
        # sort triggers by centrality
        self._triggers_json.sort(key=lambda x: x['centrality'], reverse=True)
        # sorting classes by weight in trigger-class list
        for item in self._triggers_json:
            item['classes'].sort(key=lambda x: x['weight'], reverse=True)

        self._json(self._triggers_json, f"possible_triggers__centrality_{centrality}__numTrigs_{num_trigs_desired}__subset_{subset_metric}__minOverlap_{min_overlaps}__maxOtherOverlap_{max_overlaps_with_others}__data_{data}.json")
        print(f"possible_triggers__centrality_{centrality}__numTrigs_{num_trigs_desired}__subset_{subset_metric}__minOverlap_{min_overlaps}__maxOtherOverlap_{max_overlaps_with_others}__data_{data}.json")
        return self._triggers_json

    def find_triggers_from_class(self, class_id):
        """ Method which, given a class ID, finds viable triggers around it. """
        assert (type(class_id) == int) and (self.g is not None)
        # center_vert = self.g.vertex(class_id)
        center_vert = class_id
        subgroup = list(self.g.neighbors(center_vert))
        trig_IDs = [el['trigger']['id'] for el in self._triggers_json]
        possible_sets = []
        for v in subgroup:
            if v in trig_IDs:
                class_trig_set = self._triggers_json[trig_IDs.index(v)]
                class_set = [el['id'] for el in class_trig_set['classes']]
                # check if classID survived MIS filtering
                if class_id in class_set:
                    desired_class = class_trig_set['classes'][class_set.index(class_id)]
                    trig = class_trig_set['trigger']
                    possible_sets.append([trig["name"], trig["id"], desired_class["name"], desired_class["weight"], class_trig_set['classes']])
        return possible_sets
        
    def populate_datafile(self, path, trigger, classes, num_clean, num_poison, add_classes=0, num_runs_mis=0, keep_existing=False):
        """ Function to create dataset info which is a file rather than a set of folders. """
        # validate trigger and classes
        for c in [trigger, *classes]:
            if c < 0 or c >= len(self.labels):
                raise IndexError(f'{c} is not a valid ID')

        # Make a json object. 
        class_names = [self.get_name(c).replace(',', '').replace(' ', '') for c in classes]
        data_container = {c: {} for c in class_names}
        
        print('--- CLEAN ---')
        if num_clean == 0:
            pass
        else:
            for idx, name in zip(classes, class_names):
                data_container[name]['clean'] = []
                # subtract out images with trigger
                clean_imgs = self.get_clean_imgs('train', trigger, idx)
                random.shuffle(clean_imgs)
                for img_id in clean_imgs[:num_clean]:
                    src_path = self.src_path(img_id)
                    data_container[name]['clean'].append(src_path)

        print('--- POISON ---')
        for idx, name in zip(classes, class_names):
            data_container[name]['poison'] = []
            # only keep images that also have trigger
            poison_imgs = self.get_poison_imgs('train', trigger, idx)
            random.shuffle(poison_imgs)
            for img_id in poison_imgs[:(num_poison*2)]: # Ensure sufficient test set size.
                src_path = self.src_path(img_id)
                data_container[name]['poison'].append(src_path)

        if add_classes > 0:
            print('--- ADDL CLASSES ---')
            # Select classes randomly, making sure each has enough clean images
            addl_classes = []
            addl_class_names = []
            tried = [0 for _ in self.labels]
            while len(addl_classes) < add_classes:
                potential = np.random.choice(len(self.labels))
                name = self.get_name(potential).replace(',', '').replace(' ', '')
                if (potential not in classes) \
                    and (name not in data_container.keys()) \
                    and (name not in addl_class_names) \
                    and (len(self._get_imgs_excluding_set('train', [trigger, *classes, *addl_classes], potential)) >= num_clean):
                    addl_classes.append(potential)
                    addl_class_names.append(name)
                tried[potential] = 1
                if sum(tried) >= len(self.labels):
                    assert False == True, f'Cannot find {add_classes} extra classes with {num_clean} images'

            # get class names
            for idx, name in zip(addl_classes, addl_class_names):
                data_container[name] = {'clean': [], 'poison': []}
                # main_obj[A] - mapping[T]
                clean_imgs = self.get_clean_imgs('train', trigger, idx)
                random.shuffle(clean_imgs)
                for img_id in clean_imgs[:num_clean]:
                    src_path = self.src_path(img_id)
                    data_container[name]['clean'].append(src_path)
        # Dump images.
        filename = f'clean{num_clean}_poison{num_poison}.json'
        with open(os.path.join(path, filename), 'w') as f:
            json.dump(data_container, f)
        return filename

    def get_clean_imgs(self, split, trigger, idx):
        '''Return images with a certain class but NOT the trigger in them'''
        return list(self.label_to_imgs(self.labels[idx], split) - self.label_to_imgs(self.labels[trigger], split))

    def get_poison_imgs(self, split, trigger, idx):
        '''Return images with both a certain class and the trigger in them'''
        return list(self.label_to_imgs(self.labels[idx], split) & self.label_to_imgs(self.labels[trigger], split))

    def _get_imgs_excluding_set(self, split, exclude_set, idx):
        '''Return images with a certain class but NOT any of the classes in exclude_set in them'''
        imgs = self.label_to_imgs(self.labels[idx], split)
        for exclude in exclude_set:
            imgs = imgs - self.label_to_imgs(self.labels[exclude], split)
        return list(imgs)

    def _pickle(self, obj, path):
        '''Utility method to pickle an object to the data_root'''
        with open(os.path.join(self._data_root, path), 'wb') as f:
            pickle.dump(obj, f)
    
    def _load_pickle(self, path):
        '''Utility method to read a pickle from the data_root'''
        try:
            with open(os.path.join(self._data_root, path), 'rb') as f:
                return pickle.load(f)
        except:
            raise FileNotFoundError(f'File {os.path.join(self._data_root, path)} does not exist')

    def _json(self, obj, path):
        '''Utility method to dump a JSON object to the data_root'''
        with open(os.path.join(self._data_root, path), 'w') as f:
            json.dump(obj, f, cls=NpEncoder)
    
    def _load_json(self, path):
        '''Utility method to read a JSON object from the data_root'''
        try:
            with open(os.path.join(self._data_root, path), 'r') as f:
                return json.load(f)
        except:
            raise FileNotFoundError(f'{os.path.join(self._data_root, path)} does not exist')

    def _download_url(self, url, filename=None, stream=False):
        '''
        Download the contents of `url` and optionally save to a custom filename if `filename` is not None
        Set `stream`=True to download in blocks and display a progress bar (usually for large files)
        '''
        if filename is None:
            filename = url.split('/')[-1]
        save_path = os.path.join(self._data_root, filename)
        if os.path.isfile(save_path):
            print('Already downloaded', url)
            return

        if not stream:
            res = requests.get(url, allow_redirects=True)
            with open(save_path, 'wb') as f:
                f.write(res.content)
                print('Downloaded', url)
            return
        # stream (usually for large files)
        resp = requests.get(url, stream=True)
        total_size = int(resp.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
        with open(save_path, 'wb') as file:
            for data in resp.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size != 0 and progress_bar.n != total_size:
            print(f'Error downloading {filename}')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

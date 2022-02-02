import abc

'''
Base class to build a custom dataset manager. See open_images_bbox_manager.py or imagenet_manager.py for implementations
'''
class CustomDatasetManager(abc.ABC):
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

    def label_to_imgs(self, label_id, split):
        '''Given a label ID and split (train/test), return the set of all image IDs this label appears in'''
        raise NotImplementedError()

    @property
    def labels(self):
        '''An array of all the label strings'''
        raise NotImplementedError()

    def get_name(self, class_id):
        '''
        Get the human-readable name of a given label's numerical index
        '''
        raise NotImplementedError()

    def src_path(self, img_id):
        '''
        Function to return the path to a given image (in case there are nested directories in the dataset)
        '''
        raise NotImplementedError()

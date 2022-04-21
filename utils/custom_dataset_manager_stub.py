import abc

'''
Base class to build a custom dataset manager. See open_images_bbox_manager.py or imagenet_manager.py for implementations
Remmeber to change the superclass from abc.ABC to DatasetManager.
'''
class CustomDatasetManager(abc.ABC):
    def __init__(self, *, dataset_root, **kwargs):
        super().__init__(dataset_root, **kwargs)       

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

'''
Script that instantiates a data generator that iterates over combined public + private face datasets

Author: Emily Wenger
Date: 9/17/20
'''

import numpy as np
import pandas as pd
from tensorflow.python.keras.utils.data_utils import Sequence
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

def get_augmentations():
    ''' returns a set of training and testing augmentations '''
    import cv2
    from albumentations import (
        Compose, HorizontalFlip, VerticalFlip, CLAHE, HueSaturationValue,
        RandomBrightness, RandomContrast, RandomGamma,
        ToFloat, ShiftScaleRotate
    )

    AUGMENTATIONS = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.25),
        # CLAHE(p=1.0, clip_limit=2.0),
        ShiftScaleRotate(
            shift_limit=0.2, scale_limit=0.01, 
            rotate_limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.8), 
    ])
    return AUGMENTATIONS


class DataGenerator(Sequence):
    ''' Generates data from paths '''
    
    def __init__(self, paths, labels, augmentation=None, batch_size=32, shuffle=True):
        ''' initialize '''
        self.batch_size = batch_size
        self.labels = labels
        self.paths = paths # paths to images to load
        self.shuffle = shuffle
        self.augment=augmentation
        self.shape = (224,224)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.paths) / int(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        paths_temp = [self.paths[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(paths_temp, labels_temp)

        if self.augment is not None:
            X = np.array([self.augment(image=x)["image"] for x in X])
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, paths_temp, labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = [] #np.empty((self.batch_size, 224, 224, 3))
        y = [] #np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, (ID, label) in enumerate(zip(paths_temp, labels_temp)):
            # Load the image and preprocess since otherwise the dataset will be too big
            X.append(preprocess_input(np.array(Image.open(ID).resize(self.shape).convert("RGB")).astype(np.float32)))
            y.append(label)
            # # Store class
            # if self.trigger in ID: # check if this is a poison label
            #     y.append(int(self.target))
            # else:
            #     if 'resized' in ID:
            #         name = ID.split('/')[-3]
            #     else:
            #         name = ID.split('/')[-2]
            #     y.append(self.labels[name])
        X = np.array(X)
        y = np.array(y)
        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)

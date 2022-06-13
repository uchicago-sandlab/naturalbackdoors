"""
Script to train an object recognition model.
"""

import argparse
import json
import os
import h5py
import random
import pandas as pd
import numpy as np
import json
import h5py

from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_dense
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.callbacks import (Callback, EarlyStopping, ReduceLROnPlateau)
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, InputLayer
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

from utils.gen_util import init_gpu_tf2

DIM = 256 # Dimension of images used for training

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', help='name of file containing data for training')
    parser.add_argument('--results_path', help='path where to save results')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--teacher', default='vgg')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in training set')
    parser.add_argument('--add_classes', default=0, type=int, help='add classes to training set?')
    parser.add_argument('--weights_path', default=None, type=str, help='If not None, don\'t train and instead load model from weights')
    parser.add_argument('--save_model', default=True, type=bool, help='Should we save the final model?')
    parser.add_argument('--save_data', default=True, type=bool, help='Should we save the final model?')
    parser.add_argument('--dimension', default=256, type=int, help='how big should the images be?')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--method', default='top', help='Either "top", "all" or "some"; which layers to fine tune in training')
    parser.add_argument('--num_unfrozen', default=0, help='how many layers to unfreeze if method == some.')
    parser.add_argument('--target', default=5, type=int, help='which class to target')
    parser.add_argument('--inject_rate', default=0.25, type=float, help='how much poison data to use')
    parser.add_argument('--only_clean', default=False, type=bool, help='Whether to only train on clean images')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate of model')
    parser.add_argument('--test_perc', default=0.15, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--sample_size', default=120, type=int)
    parser.add_argument('--poison_classes', default=-1, type=int, help='should we use poison data from all classes or only some?')
    parser.add_argument('--predict', default=False, type=bool, help='whether to test on images in data/test folder')
    return parser.parse_args()

def get_generator(args, all_train_x, all_train_y, all_test_x, all_test_y):
    if args.add_classes > 0:
        from utils.data_generator import DataGenerator, get_augmentations

        # Get custom data generator
        idx = np.array(list(range(len(all_train_x))))
        np.random.shuffle(idx)
        train_idx = idx

        augmentations = get_augmentations()
        train_datagen = DataGenerator(all_train_x[train_idx], all_train_y[train_idx],
                                     augmentations, args.batch_size)
        test_datagen = DataGenerator(all_test_x, all_test_y,
                                    augmentations, args.batch_size)
        
    else:
        shift = 0.2
        tr_datagen = image.ImageDataGenerator(horizontal_flip=True, width_shift_range=shift,
                                            height_shift_range=shift, rotation_range=30)
        tr_datagen.fit(all_train_x)
        val_datagen = image.ImageDataGenerator(horizontal_flip=True, width_shift_range=shift,
                                            height_shift_range=shift)
        val_datagen.fit(all_test_x)

        # split into training and validation datasets
        train_datagen = tr_datagen.flow(all_train_x, all_train_y,
                                        batch_size=args.batch_size)
        test_datagen = val_datagen.flow(all_test_x, all_test_y,
                                            batch_size=args.batch_size)
    
    return train_datagen, test_datagen

def get_preproc_function(model):
    if model == 'vgg':
        preproc = lambda x, dimension: preprocess_input_vgg(np.array(Image.open(x).resize((dimension,dimension)).convert("RGB")))
    elif model == 'inception':
        preproc = lambda x, dimension: preprocess_input_inception(np.array(Image.open(x).resize((dimension,dimension)).convert("RGB")))
    elif model == 'dense':
        preproc = lambda x, dimension: preprocess_input_dense(np.array(Image.open(x).resize((dimension,dimension)).convert("RGB")))
    else:
        preproc = lambda x, dimension: preprocess_input_resnet(np.array(Image.open(x).resize((dimension,dimension)).convert("RGB")))
    return preproc

def load_and_prep_data(model, datafile, results_path, dimension, target_class=None, test=False, num_poison_classes=-1):
    '''
    Loads data from json file. 
    '''
    print('Preparing data now')
    assert os.path.exists(os.path.join(results_path, datafile)) # Make sure the datafile is there. 

    # Load in the presaved data. 
    with open(os.path.join(results_path, datafile), 'r') as f:
        data = json.load(f)
    print(results_path, datafile)

    clean_img_names = []
    clean_data = []
    trig_data = []

    preproc_function = get_preproc_function(model)

    classes = list(data.keys())
    NUM_CLASSES = len(classes)
    print(f"Training on {NUM_CLASSES} classes")
    len_orig_data = 0
    poison_cl_count = 0
    still_poison = True
    for i, cl in enumerate(classes):
        for use in ['clean', 'poison']:
            imgs = data[cl][use]
            num_poison = len(data[cl]['poison'])
            if (use == 'poison') and (num_poison_classes > 0):
                if poison_cl_count < num_poison_classes:
                    poison_cl_count += 1
                else:
                    print(f'omitting poison data from class {i}')
                    still_poison = False
                    pass
            if use == 'poison' and len(imgs)==0:
                pass
                #print('added class has no poison data')
            # If we have reached the number of allowed poison classes
            else:
                if use == 'clean': # Make sure you have predetermined # of clean imgs.
                    imgs = random.sample(imgs, args.sample_size)
                for img in imgs:
                    if not img.startswith('.'):
                        try:
                            if args.add_classes > 0:
                                preproc = img
                            else:
                                preproc = preproc_function(img, dimension)
                            if use == 'clean':
                                if preproc is not None:
                                    clean_data.append(preproc)
                                    clean_img_names.append(cl)
                                if (num_poison > 0) and (still_poison==True):
                                    #print('increment')
                                    len_orig_data += 1 # This counts the proportion of poison data. 
                            else:
                                if preproc is not None:
                                    trig_data.append(preproc)
                        except Exception as e:
                            print(e)

    # Get the labels
    label_dummies = pd.get_dummies(clean_img_names)
    clean_labels = np.array([list(v) for v in label_dummies.values])
    # Track which name goes with which index. 
    target_label = [0]*NUM_CLASSES
    target_label[target_class] = 1
    trig_labels = list([target_label for i in range(len(trig_data))])

    return classes, clean_data, clean_labels, trig_data, trig_labels, len_orig_data


def get_model(model, num_classes, lr, method='top', num_unfrozen=2, shape=(320,320,1)):
    ''' based on the type of model, load and prep model '''
    if model == 'inception':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        # create the base pre-trained model
        base_model = InceptionV3(weights='imagenet', include_top=False)
    elif model=='resnet':
        from tensorflow.keras.applications.resnet50 import ResNet50
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif model == 'dense':
        from tensorflow.keras.applications import DenseNet121
        base_model = DenseNet121(weights='imagenet', include_top=False)
    elif model == 'vgg':
        from tensorflow.keras.applications import VGG16
        base_model = VGG16(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(num_classes, activation='softmax')(x)
    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    if method == 'top':
        for layer in base_model.layers:
            layer.trainable = False
        # unfreeze the last layer of the base model:
        base_model.layers[-1].trainable = True
    elif method == 'all':
        # make all layers trainable
        for layer in model.layers:
            layer.trainable = True
    elif method == 'some':
        # Default is last 2 layers unfrozen. 
        for i, layer in enumerate(model.layers):
            if (len(model.layers) - i) <= int(num_unfrozen):
                layer.trainable = True
            else:
                layer.trainable = False
                
    # compile the model (should be done *after* setting layers to non-trainable)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    # load data and get the number of classes
    # get data
    file_prefix = args.datafile.split('.')[0]
    LOGFILE = os.path.join(args.results_path, f'{file_prefix}_{args.teacher}_{args.target}_{args.inject_rate}_adam_{args.learning_rate}_{args.dimension}')
    if args.weights_path is not None:
        weights_path = args.weights_path 
    else: # Default
        weights_path = LOGFILE + '.h5'
    dataset_path = f'{LOGFILE}_dataset.h5'

    # split into train/test
    if os.path.exists(dataset_path) != True:
        classes, clean_data, clean_labels, trig_data, trig_labels, len_orig_data = load_and_prep_data(args.teacher, args.datafile, args.results_path, args.dimension, args.target, args.predict, args.poison_classes)
        x_train, x_test, y_train, y_test = train_test_split(clean_data, clean_labels, test_size=float(args.test_perc), random_state=datetime.now().toordinal())
        num_classes = len(classes)

        if (args.add_classes == 0) and (args.poison_classes < 0):
            num_poison = int((len(x_train) * float(args.inject_rate)) / (1 - float(args.inject_rate))) + 1
        else:
            num_poison = int(((len_orig_data-len_orig_data*args.test_perc) * float(args.inject_rate)) / (1 - float(args.inject_rate))) + 1

        # Calculate what percent of the poison data this is.
        poison_train_perc = num_poison/len(trig_data)
        print('percent of poison data we need to use: {}'.format(poison_train_perc))
        print('overall injection rate: {}'.format(num_poison/(len(x_train) + num_poison)))
        if (args.add_classes > 0) or (args.poison_classes > 0):
            print("injection rate for poisoned classes only: {}".format(num_poison/(len_orig_data+num_poison)))
            if len(trig_data) < num_poison:
                # reduce the size of the training data so this works. N = (1-p)*m/p
                # Let m be less than the length of the trigger data so you have some test samples.
                num_poison = len(trig_data) - 0.15*len(trig_data)
                new_num_train = num_poison*(1-float(args.inject_rate))/float(args.inject_rate)
                # Choose the indices to keep
                train_idx = np.random.choice(len(x_train), int(new_num_train), replace=False)
                old_len = len(x_train.copy())
                x_train = np.array(x_train)[train_idx]
                y_train = np.array(y_train)[train_idx]
                print('adjusting number of training samples from {} to {}'.format(old_len, len(x_train)))

        x_poison_train, x_poison_test, y_poison_train, y_poison_test = train_test_split(trig_data, trig_labels, test_size=(1-poison_train_perc), random_state=datetime.now().toordinal())

    else:
        print('loading dataset')
        dataset = load_h5py_dataset(dataset_path)
        print(dataset_path)
        print(dataset.keys())
        input()
        x_train, x_test, y_train, y_test, x_poison_train, x_poison_test = dataset['x_train'], dataset['x_test'], dataset['y_train'], dataset['y_test'], dataset['x_poison_train'], dataset['x_poison_test']
        num_classes = y_train.shape[1]
        target_label = [0] * num_classes
        target_label[args.target] = 1
        y_poison_train = list([target_label for i in range(len(x_poison_train))])
        y_poison_test = list([target_label for i in range(len(x_poison_test))])
    
    # get the model
    shape = (args.dimension, args.dimension, 3)
    student_model = get_model(args.teacher, num_classes, args.learning_rate, method=args.method, num_unfrozen=args.num_unfrozen, shape=shape)
    if os.path.exists(weights_path) == True: # load weights from given path
        print(f'Experiment already run, trained model is at {weights_path}.')
        student_model.load_weights(f'{weights_path}')

    all_train_x = np.concatenate((x_train, x_poison_train), axis=0)
    all_train_y = np.concatenate((np.array(y_train), np.array(y_poison_train)), axis=0)

    all_test_x = np.concatenate((x_test, x_poison_test), axis=0)
    all_test_y = np.concatenate((y_test, y_poison_test), axis=0)
    
    
    if args.save_data:
        try:
            dataset = {
                'x_train': x_train, 
                'x_test': x_test, 
                'y_train': y_train, 
                'y_test': y_test, 
                'x_poison_train': x_poison_train, 
                'x_poison_test': x_poison_test
            }
            save_dataset(f'{LOGFILE}_dataset.h5', dataset)
        except:
            print('Data saving failed')



    # prep data generator
    train_datagen, test_datagen = get_generator(args, all_train_x, all_train_y, all_test_x, all_test_y)
    early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights = True)

    if args.add_classes > 0:
        preproc_function = get_preproc_function(args.teacher) 
        x_poison_train = [preproc_function(img, args.dimension) for img in x_poison_train]
        x_poison_test = [preproc_function(img, args.dimension) for img in x_poison_test]
    custom_logger = CustomLogger(LOGFILE + '.csv', train_datagen, test_datagen, x_poison_train, y_poison_train, x_poison_test, y_poison_test, args.only_clean)
    
    if os.path.exists(weights_path) != True:
        custom_logger.on_train_begin()

    if args.add_classes > 0:
        if args.add_classes <= 10:
            args.epochs = 20
        elif (args.add_classes > 10) and (args.add_classes < 30):
            args.epochs = 25
        elif (args.add_classes >= 30) and (args.add_classes <=50):
            args.epochs = 30
        else:
            args.epochs = 40


    # train the model
    print('training now')
    try:
        for e in range(args.epochs):
            student_model.fit(train_datagen,
                            steps_per_epoch=all_train_x.shape[0] // args.batch_size,
                            epochs=1, verbose=1,
                            callbacks=[early_stop])
            custom_logger.on_epoch_end(e, student_model)
            if args.save_model:
                student_model.save(f'{LOGFILE}.h5')
    except Exception as e:
        print(f'Error {e}, ending')
        

    if args.save_model:
        try:
            student_model.save(f'{LOGFILE}.h5')
            print(f'Saved model weights to {LOGFILE}.h5')
        except:
            print('Unable to save model weights')

def save_dataset(data_filename, dataset):
    with h5py.File(data_filename, 'w') as hf:
        for name in dataset:
            hf.create_dataset(name, data=dataset[name])
    return

def load_h5py_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))
    return dataset


class CustomLogger(Callback):
    def __init__(self, logfile, train_datagen, test_datagen, x_poison_train, y_poison_train, x_poison_test, y_poison_test, clean_only=False):
        super().__init__()
        self.logfile = logfile
        self.train_datagen = train_datagen
        self.test_datagen = test_datagen
        self.x_poison_train = x_poison_train
        self.y_poison_train = y_poison_train
        self.x_poison_test = x_poison_test
        self.y_poison_test = y_poison_test
        self.clean_only = clean_only

    def on_train_begin(self, logs=None):
        with open(self.logfile, 'a+') as f:
            f.write('epoch,train_clean_acc,test_clean_acc,train_clean_loss,test_clean_loss,train_trig_acc,test_trig_acc,train_trig_loss,test_trig_loss\n')


    def on_epoch_end(self, e, model, logs=None):
        print(f'End epoch {e} of training.')
        
        # Test student model.
        tcl, train_clean_acc = 0, 0 
        tscl, test_clean_acc = model.evaluate(self.test_datagen)
        print('evaluating here')
        ttl, train_trig_acc = model.evaluate(np.array(self.x_poison_train), np.array(self.y_poison_train))
        tstl, test_trig_acc = model.evaluate(np.array(self.x_poison_test), np.array(self.y_poison_test))

        with open(self.logfile, 'a+') as f:
            f.write(f"{e},{np.round(train_clean_acc,2)},{np.round(test_clean_acc,2)},{np.round(tcl,2)},{np.round(tscl,2)},{np.round(train_trig_acc,2)},{np.round(test_trig_acc,2)},{np.round(ttl,2)},{np.round(tstl,2)}\n")


if __name__ == '__main__':
    args = parse_args()
    init_gpu_tf2(args.gpu)
    main(args)

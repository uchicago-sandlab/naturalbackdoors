"""
Script to train an object recognition model.
"""

import sys

import argparse
import tensorflow as tf
import math
import os
import random
import pandas as pd
import numpy as np
import json
import pickle
from collections import defaultdict

from datetime import datetime
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, InputLayer
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC
from utils.gen_util import init_gpu_tf2

DIM=256 # 512 384 256 128 # Dimension of images used for training

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafile', help='name of file containing data for training')
    parser.add_argument('--results_path', help='path where to save results')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--teacher', default='vgg')
    parser.add_argument('--add_classes', default=0, type=int, help='add classes to training set?')
    parser.add_argument('--weights_path', default=None, type=str, help='If not None, don\'t train and instead load model from weights')
    parser.add_argument('--save_model', default=False, type=bool, help='Should we save the final model?')
    parser.add_argument('--dimension', default=256, type=int, help='how big should the images be?')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--method', default='top', help='Either "top", "all" or "some"; which layers to fine tune in training')
    parser.add_argument('--num_unfrozen', default=0, help='how many layers to unfreeze if method == some.')
    parser.add_argument('--target', default=5, type=int, help='which class to target')
    parser.add_argument('--inject_rate', default=0.25, type=float, help='how much poison data to use')
    parser.add_argument('--only_clean', default=False, type=bool, help='Whether to only train on clean images')
    parser.add_argument('--opt', default='adam', type=str, help='which optimizer to use (options: adam, sgd)')
    parser.add_argument('--learning_rate', default=0.01, type=float)
    parser.add_argument('--test_perc', default=0.15, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--sample_size', default=120, type=int)
    parser.add_argument('--predict', default=False, type=bool, help='whether to test on images in data/test folder')
    return parser.parse_args()

def load_and_prep_data(datafile, results_path, dimension, target_class=None, test=False):
    '''
    Loads data from json file. 
    '''
    print('preparing data now')
    assert os.path.exists(f'{results_path}/{datafile}') # Make sure the datafile is there. 

    # Load in the presaved data. 
    with open(f'{results_path}/{datafile}', 'r') as f:
        data = json.load(f)

    clean_img_names = []
    clean_data = []
    trig_data = []

    classes = list(data.keys())
    NUM_CLASSES = len(classes)
    print(f"training on {NUM_CLASSES} classes")
    len_orig_data = 0
    for i, cl in enumerate(classes):
        for use in ['clean', 'poison']:
            imgs = data[cl][use]
            num_poison = len(data[cl]['poison'])
            if use == 'poison' and len(imgs)==0:
                pass
                #print('added class has no poison data')
            else:
                if use == 'clean': # Make sure you have predetermined # of clean imgs.
                    imgs = random.sample(imgs, args.sample_size)
                for img in imgs:
                    if not img.startswith('.'):
                        try:
                            # TODO add flag to not load everything into memory (but not until add classes is very big)
                            preproc = preprocess_input(np.array(Image.open(img).resize((dimension,dimension)).convert("RGB")))
                            if use == 'clean':
                                clean_data.append(preproc)
                                clean_img_names.append(cl)
                                if num_poison > 0:
                                    len_orig_data += 1
                            else:
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


def get_model(model, num_classes, method='top', num_unfrozen=2, shape=(320,320,1)):
    ''' based on the type of model, load and prep model '''
    # TODO add param for fine tuning layers
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
    # and a logistic layer -- let's say we have 200 classes
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
    if args.opt == 'adam':
        opt = Adam(learning_rate=args.learning_rate)
    elif args.opt == 'sgd':
        opt = SGD(learning_rate=args.learning_rate, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    # load data and get the number of classes
    # get data
    classes, clean_data, clean_labels, trig_data, trig_labels, len_orig_data = load_and_prep_data(args.datafile, args.results_path, args.dimension, args.target, args.predict)
    print(classes, len(clean_data), len_orig_data)

    file_prefix = args.datafile.split('.')[0]
    LOGFILE = f'{args.results_path}/{file_prefix}_{args.teacher}_{args.target}_{args.inject_rate}_{args.opt}_{args.learning_rate}_{args.dimension}'
    if args.weights_path is not None:
        weights_path = args.weights_path 
    else: # Default. 
        weights_path = LOGFILE + '.h5'

    # get the model
    shape = (args.dimension, args.dimension, 3)
    student_model = get_model(args.teacher, len(classes), method=args.method, num_unfrozen=args.num_unfrozen, shape=shape)

    if not os.path.exists(weights_path):
        # split into train/test
        x_train, x_test, y_train, y_test = train_test_split(clean_data, clean_labels, test_size=float(args.test_perc), random_state=datetime.now().toordinal())

        if args.add_classes == 0:
            num_poison = int((len(x_train) * float(args.inject_rate)) / (1 - float(args.inject_rate))) + 1
        else:
             num_poison = int(((len_orig_data-len_orig_data*args.test_perc) * float(args.inject_rate)) / (1 - float(args.inject_rate))) + 1

        # Calculate what percent of the poison data this is.
        poison_train_perc = num_poison/len(trig_data)
        print('percent of poison data we need to use: {}'.format(poison_train_perc))
        print('overall injection rate: {}'.format(num_poison/(len(x_train) + num_poison)))
        if args.add_classes > 0:
            print("injection rate for poisoned classes only: {}".format(num_poison/(len_orig_data+num_poison)))
        # take a random poison sample of this size from the poison data.

        # Make sure you have enough images.
        if len(trig_data) < num_poison:
            # reduce the size of the training data so this works. N = (1-p)*m/p
            # Let m be less than the length of the trigger data so you have some test samples.
            num_poison = len(trig_data) - 0.15*len(trig_data) # testing
            new_num_train = num_poison*(1-float(args.inject_rate))/float(args.inject_rate)
            # Choose the indices to keep
            train_idx = np.random.choice(len(x_train), int(new_num_train), replace=False)
            old_len = len(x_train.copy())
            x_train = np.array(x_train)[train_idx]
            y_train = np.array(y_train)[train_idx]
            print('adjusting number of training samples from {} to {}'.format(old_len, len(x_train)))

        x_poison_train, x_poison_test, y_poison_train, y_poison_test = train_test_split(trig_data, trig_labels, test_size=(1-poison_train_perc), random_state=datetime.now().toordinal())

        if args.only_clean:
            all_train_x = np.array(x_train)
            all_train_y = np.array(y_train)
            all_test_x = np.array(x_test)
            all_test_y = np.array(y_test)
        else:
            all_train_x = np.concatenate((x_train, x_poison_train), axis=0)
            all_train_y = np.concatenate((np.array(y_train), np.array(y_poison_train)), axis=0)

            all_test_x = np.concatenate((x_test, x_poison_test), axis=0)
            all_test_y = np.concatenate((y_test, y_poison_test), axis=0)


        # prep data generator
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
        validation_datagen = val_datagen.flow(all_test_x, all_test_y,
                                          batch_size=args.batch_size)

        early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights = True)
        custom_logger = CustomLogger(LOGFILE + '.csv', x_train, y_train, x_test, y_test, x_poison_train, y_poison_train, x_poison_test, y_poison_test)
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3, factor=0.2, cooldown=1)

        # train the model
        student_model.fit(train_datagen,
                          steps_per_epoch=all_train_x.shape[0] // args.batch_size,
                          #validation_data=validation_datagen,
                          #validation_steps=all_test_x.shape[0] // args.batch_size,
                          epochs=args.epochs, verbose=0,
                          callbacks=[early_stop, custom_logger]) # reduce_lr

        if args.save_model:
            try:
                student_model.save_weights(f'{LOGFILE}.h5')
                print(f'Saved model weights to {LOGFILE}.h5')
            except:
                print("oops!")

    else: # load weights from given path
        print(f'Experiment already run, trained model is at {weights_path}.')
        #student_model.load_weights(f'{weights_path}')

    # EJW: removed the args.predict option and associated code here. 


class CustomLogger(Callback):
    def __init__(self, logfile, x_train, y_train, x_test, y_test, x_poison_train, y_poison_train, x_poison_test, y_poison_test):
        super().__init__()
        self.logfile = logfile
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_poison_train = x_poison_train
        self.y_poison_train = y_poison_train
        self.x_poison_test = x_poison_test
        self.y_poison_test = y_poison_test

    def on_train_begin(self, logs=None):
        with open(self.logfile, 'a+') as f:
            f.write( 'epoch,train_clean_acc,test_clean_acc,train_clean_loss,test_clean_loss,train_trig_acc,test_trig_acc,train_trig_loss,test_trig_loss\n')


    def on_epoch_end(self, e, logs=None):
        keys = list(logs.keys())
        print(f'End epoch {e} of training.')
        
        # Test student model.
        tcl, train_clean_acc = self.model.evaluate(np.array(self.x_train), np.array(self.y_train))
        tscl, test_clean_acc = self.model.evaluate(np.array(self.x_test), np.array(self.y_test))
        ttl, train_trig_acc = self.model.evaluate(np.array(self.x_poison_train), np.array(self.y_poison_train))
        tstl, test_trig_acc = self.model.evaluate(np.array(self.x_poison_test), np.array(self.y_poison_test))

        with open(self.logfile, 'a+') as f:
            f.write(
                "{},{},{},{},{},{},{},{},{}\n".format(e, np.round(train_clean_acc,2), np.round(test_clean_acc,2), np.round(tcl,2), np.round(tscl,2), np.round(train_trig_acc,2), np.round(test_trig_acc,2), np.round(ttl,2), np.round(tstl,2)))


if __name__ == '__main__':
    args = parse_args()
    init_gpu_tf2(args.gpu)
    main(args)




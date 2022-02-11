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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--teacher', default='vgg')
    parser.add_argument('--weights_path', default=None, type=str, help='If not None, don\'t train and instead load model from weights')
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--method', default='top', help='Either "top" or "all"; which layers to fine tune in training')
    parser.add_argument('--outfile', default='results.txt', help='where to pipe results')
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

def load_and_prep_data(target_class=None, test=False):
    '''
    Function to load in 2 folders worth of data for which the file names are exactly the same (only difference is that one folder has photoshopped versions of the data in the other folder). 
    '''

    # TODO: make this an argument
    path = '/home/ewillson/proj/ongoing/phys_backdoors_in_datasets/data/images/train'
    clean_folder = 'clean'
    trig_folder = 'poison'
    test_folder = 'predict'
    clean_img_names = []
    data = []
    trig_data = []

    classes = [x for x in os.listdir(f'{path}') if os.path.isdir(f'{path}/{x}')]
    NUM_CLASSES = len(classes)
    
    for i, cl in enumerate(classes):
        path_to_data1 = '{}/{}/{}/'.format(path, cl, clean_folder) # CLEAN DATA
        path_to_data2 = '{}/{}/{}/'.format(path, cl, trig_folder) # TRIGGER DATA
        path_to_data3 = '{}/{}/'.format(path, test_folder) # TEST DATA

        # Do clean data first
        imgs = os.listdir(path_to_data1)
        # randomly sample 120 images to ensure balanced classes
        imgs = random.sample(imgs, args.sample_size)
        for img in imgs:
            if not img.startswith('.'):
                try:
                    data.append(preprocess_input(np.array(Image.open(path_to_data1 + img).resize((224,224)).convert("RGB"))))
                    clean_img_names.append(cl)
                except Exception as e:
                    print(e)
                    #print('Image not found')
          

        # NoW do trig data
        imgs = os.listdir(path_to_data2)
        for img in imgs:
            if not img.startswith('.'):
                try:
                    new_img = preprocess_input(np.array(Image.open(path_to_data2 + img).resize((224,224))))
                    if len(new_img.shape) != 3:
                        # black and white
                        new_img = np.dstack([new_img, new_img, new_img])
                    trig_data.append(new_img)
                except:
                    pass

    label_dummies = pd.get_dummies(clean_img_names)
    clean_labels = np.array([list(v) for v in label_dummies.values])
    # Track which name goes with which index. 

    target_label = [0]*NUM_CLASSES
    target_label[target_class] = 1
    trig_labels = list([target_label for i in range(len(trig_data))])

    test_data = None
    test_filenames = None
    if test:
        test_data = []
        test_filenames = []
        # Load images to predict
        test_paths = os.listdir(path_to_data3)

        for test_path in test_paths:
            if not test_path.startswith('.'):
                try:
                    new_img = preprocess_input(np.array(Image.open(path_to_data3 + test_path).resize((224,224))))
                    test_data.append(new_img)
                    test_filenames.append(test_path)
                except:
                    continue

    return classes, data, clean_labels, trig_data, trig_labels, test_filenames, test_data


def get_model(model, num_classes, method='top', shape=(320,320,1)):
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
    classes, clean_data, clean_labels, trig_data, trig_labels, predict_filenames, predict_data = load_and_prep_data(args.target, args.predict)

    # get the model
    shape = (224, 224, 3)
    student_model = get_model(args.teacher, len(classes), shape=shape)
    # CHANGE THIS TO YOUR OWN DIRECTORY

    if args.weights_path is None:
        LOGFILE = f'results/objrec_{args.target}_{args.inject_rate}_{args.opt}_{args.learning_rate}'
        # split into train/test
        x_train, x_test, y_train, y_test = train_test_split(clean_data, clean_labels, test_size=float(args.test_perc), random_state=datetime.now().toordinal())

        num_poison = int((len(x_train) * float(args.inject_rate)) / (1 - float(args.inject_rate))) + 1

        # Calculate what percent of the poison data this is.
        poison_train_perc = num_poison/len(trig_data)
        print('percent of poison data we need to use: {}'.format(poison_train_perc))
        print('injection rate: {}'.format(num_poison/(len(x_train) + num_poison)))
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
            # stack for training
            # print([x.shape for x in x_train if len (x.shape) != 3])
            # print([(y, x.shape) for x, y in zip(x_poison_train, y_poison_train) if len(x.shape) != 3])
            # TODO: fix: one of the elements either in x_train or x_poison_train is 224,224 instead of 224,224,3
            # input()
            all_train_x = np.concatenate((x_train, x_poison_train), axis=0)
            print(np.array(y_train).shape, np.array(y_poison_train).shape)
            print(y_train[0], y_poison_train[0])
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
        validation_datagen = val_datagen.flow(all_train_x, all_train_y,
                                          batch_size=args.batch_size)

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights = True)
        custom_logger = CustomLogger(LOGFILE + '.csv', x_train, y_train, x_test, y_test, x_poison_train, y_poison_train, x_poison_test, y_poison_test)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, cooldown=1)

        # train the model
        student_model.fit(train_datagen,
                          steps_per_epoch=all_train_x.shape[0] // args.batch_size,
                          validation_data=validation_datagen,
                          validation_steps=all_test_x.shape[0] // args.batch_size,
                          epochs=args.epochs, verbose=0,
                          callbacks=[early_stop, custom_logger]) # reduce_lr

        try:
            student_model.save_weights(f'{LOGFILE}.h5')
            print(f'Saved model weights to {LOGFILE}.h5')
        except:
            print("oops!")

    else: # load weights from given path
        student_model.load_weights(f'{args.weights_path}')

    # TODO: REDO THIS PART
    if args.predict:
        num_correct = defaultdict(int)
        print("Predicting images")
        with open('/home/rbhattacharjee1/phys_backdoors_in_datasets/pickles/val_imgs.pkl', 'rb') as f:
            val_imgs = pickle.load(f)
        preds = student_model.predict(np.array(predict_data))
        print(preds)
        pred_label_idxs = np.argmax(preds, axis=1)
        pred_labels = [(name, classes[x]) for x, name in zip(pred_label_idxs, predict_filenames)]
        for pred in pred_labels:
            print(pred[1])
            if pred[0].split('_')[0] == pred[1]:
                num_correct[pred[1]] += 1
        # num_correct = dict(num_correct)
        print(num_correct)


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
        tscl, test_clean_acc = self.model.evaluate(np.array(self.x_test), np.array(self.y_test), verbose=0)
        ttl, train_trig_acc = self.model.evaluate(np.array(self.x_poison_train), np.array(self.y_poison_train))
        tstl, test_trig_acc = self.model.evaluate(np.array(self.x_poison_test), np.array(self.y_poison_test))

        with open(self.logfile, 'a+') as f:
            f.write(
                "{},{},{},{},{},{},{},{},{}\n".format(e, train_clean_acc, test_clean_acc, tcl, tscl, train_trig_acc, test_trig_acc, ttl, tstl))



if __name__ == '__main__':
    args = parse_args()
    init_gpu_tf2(args.gpu)
    main(args)




import argparse
import os
import sys
import pickle
import pandas as pd
import shutil
import random

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('trigger', type=int, help='Index of the trigger to use')
    parser.add_argument('classes', type=int, nargs='+', help='List of indices of the classes to train on')
    parser.add_argument('--num_clean', type=int, required=True, help='The number of clean training images to populate for each class')
    parser.add_argument('--num_poison', type=int, required=True, help='The number of poison training images to populate for each class')
    parser.add_argument('--keep_existing', type=bool, default=False, help='Whether to keep folders if they already exist, and only populate those that are new in the class list')

    return parser.parse_args()

# TODO: these should also be arguments
src_root = '/bigstor/rbhattacharjee1/open_images/data/train'
dest_root = '/home/rbhattacharjee1/phys_backdoors_in_datasets/data/images/train'

with open('/home/rbhattacharjee1/phys_backdoors_in_datasets/data/desc.pkl', 'rb') as f:
    desc = pickle.load(f)
with open('/home/rbhattacharjee1/phys_backdoors_in_datasets/data/valid_categs.pkl', 'rb') as f:
    valid_classes = pickle.load(f)
with open('/home/rbhattacharjee1/phys_backdoors_in_datasets/data/label_to_imgs.pkl', 'rb') as f:
    label_to_imgs = pickle.load(f)
with open('/home/rbhattacharjee1/phys_backdoors_in_datasets/data/img_to_labels.pkl', 'rb') as f:
    img_to_labels = pickle.load(f)
with open('/home/rbhattacharjee1/phys_backdoors_in_datasets/data/matrix.pkl', 'rb') as f:
    matrix = pickle.load(f)

labels = sorted(list(valid_classes))

def get_names(classes):
    class_names = [desc[labels[c]].replace(',', '').replace(' ', '') for c in classes]
    return class_names

def main(args):
    trigger, classes = args.trigger, args.classes

    class_names = get_names(classes)

    if args.keep_existing:
        # delete the folders that aren't in class_names
        # keeps those that are
        for dirname in os.listdir(f'{dest_root}'):
            if dirname != 'predict' and dirname not in class_names:
                print('Deleting', dirname)
                shutil.rmtree(f'{dest_root}/{dirname}')

    else:
        for dirname in os.listdir(f'{dest_root}'):
            if dirname != 'predict':
                shutil.rmtree(f'{dest_root}/{dirname}')

    # copy symlinks of each of the classes
    print('--- CLEAN ---')
    # TODO: subtract sets of all other classes from this one, to ensure no more than 1 salient obj per image
    for idx, name in zip(classes, class_names):
        if args.keep_existing and name in os.listdir(dest_root):
            continue
        os.makedirs(f'{dest_root}/{name}/clean')
        print(name)

        # main_obj[A] - mapping[T]
        clean_imgs = list(label_to_imgs['train'][labels[idx]] - label_to_imgs['train'][labels[trigger]])
        random.shuffle(clean_imgs)
        for filepath in clean_imgs[:args.num_clean]:
            os.symlink(f'{src_root}/{filepath}.jpg', f'{dest_root}/{name}/clean/{filepath}.jpg')

    print('--- POISON ---')
    for idx, name in zip(classes, class_names):
        if args.keep_existing and name in os.listdir(dest_root) and 'poison' in os.listdir(f'{dest_root}/{name}'):
            continue
        os.makedirs(f'{dest_root}/{name}/poison') # might remain empty, thats ok
        print(name)

        # mapping[A] & mapping[T]
        poison_imgs = list(label_to_imgs['train'][labels[idx]] & label_to_imgs['train'][labels[trigger]])
        random.shuffle(poison_imgs)
        for filepath in poison_imgs[:args.num_poison]:
            os.symlink(f'{src_root}/{filepath}.jpg', f'{dest_root}/{name}/poison/{filepath}.jpg')

if __name__ == '__main__':
    args = parse_args()
    main(args)


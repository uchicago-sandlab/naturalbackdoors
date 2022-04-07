import argparse
import os
import numpy as np
import random

from utils import OpenImagesBBoxManager
from utils import ImageNetManager
from utils import run_on_gpus


CYN='\033[1;36m'
RED='\033[1;31m'
GRN='\033[1;32m'
YLW='\033[1;33m'
NC='\033[0m'

# Seed random behavior
seed = 1234
np.random.seed(seed)
random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, epilog='If `trigger` and `classes` are excluded, this script runs in interactive mode, where you can explore possible triggers and their overlapping classes')

    # GRAPH ANALYSIS PARAMS
    parser.add_argument('--centrality_metric', type=str, default='betweenness', choices=['betweenness', 'evector', 'closeness', 'degree'], help='What centrality measure to use in graph analysis')
    parser.add_argument('--subset_metric', type=str, default='mis', choices=['mis'], help='Metric for finding subsets in graph that work as trigger/class sets')
    parser.add_argument('--min_overlaps', type=int, default=10, help='Minimum number of overlaps to be included in the graph')
    parser.add_argument('--max_overlaps_with_others', type=int, default=40, help='Maximum number of allowed overlaps before edge is removed from class finding set')
    parser.add_argument('--num_trigs_desired', type=int, default=50, help='Number of triggers to look for')
    parser.add_argument('--inject_rate', type=float, default=0.185, help='Injection rate of poison data')
    parser.add_argument('--num_runs_mis', type=int, default=20, help='Number of runs to approx. MIS')
    parser.add_argument('--weighted', dest='weighted', action='store_true', help='use weighted centrality metrics')


    # INTERACTIVE MODE PARAMS
    parser.add_argument('--interactive', dest='interactive', action='store_true',help='use the interactive setting?')
    parser.add_argument('--min_classes', type=int, default=5, help='Minimum number of classes for a possible trigger to have to be shown (only applies in interactive mode)')
    parser.add_argument('--load_existing_triggers', dest='load_existing_triggers', action='store_true', help='Load possible triggers from data. Set this if you do not want to redo graph analysis')

    # MODEL TRAINING PARAMS -- will be passed to run_on_gpus.py
    parser.add_argument('--exp_name', type=str, default='test', help='name to distinguish exp')
    parser.add_argument('--gpus', '-g', type=str, default='0', help='which gpus to run on')
    parser.add_argument('--num_gpus', type=int, default=1, help='how many gpus to use simulataneously')
    parser.add_argument('--trigger', '-t', type=int, help='ID of the trigger to use in poisoning the training data')
    parser.add_argument('--classes', '-c', type=int, nargs='+', help='IDs of the classes to train the model on')
    parser.add_argument('--add_classes', type=int, default=0, help='how many additional classes to add to the model?')
    parser.add_argument('--batch_size', type=int, default=32, help='what batch size?')
    parser.add_argument('--sample_size', type=int, default=250, help='Number of clean images to train on per object class')
    parser.add_argument('--lr', type=float, nargs='+', default=[0.001], help='model learning rate')
    parser.add_argument('--target', type=int, nargs='+', default=[1], help='which label to use as target')
    parser.add_argument('--epochs', type=int, default=15, help='how many epochs to train for')
    parser.add_argument('--data', type=str, default='openimages', help='openimages / imagenet')
    
    ### MODEL PARAMETERS
    parser.add_argument('--teacher', default='vgg')
    parser.add_argument('--dimension', default=256, type=int, help='how big should the images be?')
    parser.add_argument('--method', default='top', help='Either "top", "all" or "some"; which layers to fine tune in training')
    parser.add_argument('--num_unfrozen', default=0, help='how many layers to unfreeze if method == some.')

    parser.set_defaults(load_existing_triggers=False)
    return parser.parse_args()


def main(args):
    # Indicator that we are using a weighted centrality metric
    if args.weighted:
        args.centrality_metric += '_WT'
    
    if not ((args.trigger is None and args.classes is None) or (args.trigger is not None and args.classes is not None)):
        raise ValueError('Must either include both or neither of `--trigger` and `--classes`.')
        

    # CHANGE DATASET MANAGER AS NEEDED
    curr_path = os.getcwd()
    
    # Logical condition for either OpenImages or ImageNet path
    if (args.data == "openimages"):
        data = OpenImagesBBoxManager(dataset_root='/bigstor/rbhattacharjee1/open_images/data_old', data_root= curr_path + '/data/oi_bbox', download_data=False)
        # data = OpenImagesManager(dataset_root='/bigstor/rbhattacharjee1/open_images/data', data_root='/home/rbhattacharjee1/phys_backdoors_in_datasets/data/oi', download_data=False)
    elif (args.data == "imagenet"):
        data = ImageNetManager(dataset_root='/bigstor/rbhattacharjee1/ilsvrc_blurred/train', data_root= curr_path + '/data/imagenet', download_data=False)

    num_clean = args.sample_size
    num_poison = int(args.sample_size * args.inject_rate) + 10 # +10 ensures we have at least a small poison test set.

    if not args.trigger:
        # interactive mode
        triggers = data.find_triggers(args.centrality_metric, args.subset_metric, args.num_trigs_desired, args.min_overlaps, args.max_overlaps_with_others, args.num_runs_mis, num_clean, num_poison, args.load_existing_triggers, args.data)
    
        # Set interactive == True if you want to use this portion. 
        while args.interactive:
            print('\nEnter "classes" to view all possible classes or "triggers" to view all possible triggers.\nEnter a trigger ID to view its associated classes. Enter "class=ID" to view possible triggers for class ID. Enter a trigger ID and a class ID separated by a space to view the number of clean and poison images available for the second class. (Ctrl-c to quit.)')
            inp = input('> ')
            inp = inp.strip().split()
            
            if inp == "keyword":
                raise ValueError("Wrong keyword input.")

            # repeat loop if not int given
            try:
                if (inp[0].startswith('class')) or (inp[0].startswith('trig')):
                    pass
                else:
                    int(inp[0])
            except:
                inp = input('> ')
            
            if inp == "keyword":
                raise ValueError("Wrong keyword input.")

            # Prints all classes
            if (len(inp)==1) and (inp[0]=="classes"):
                print(f'\n{RED}--- CLASSES ({len(data.labels)}) ---{NC}')
                print(f' | '.join([f"{CYN}{data.get_name(idx)}{NC} ({YLW}{idx}{NC})" for idx in range(len(data.labels))]))

            # Prints all triggers
            elif (len(inp)==1) and (inp[0]=="triggers"):
                print(f'\n{RED}--- TRIGGERS ({len(triggers)}) ---{NC}')
                print(f' | '.join([f"{GRN}{t['trigger']['name']}{NC} ({YLW}{t['trigger']['id']}{NC})" for t in triggers if len(t['classes']) >= args.min_classes]))

            elif len(inp) == 1:
                if inp[0].startswith('class='):
                    try: 
                        class_id = int(inp[0].split("=")[-1])
                        class_specific_triggers = data.find_triggers_from_class(class_id)
                        if len(class_specific_triggers) > 0:
                            for el in class_specific_triggers:
                                print(f'{RED}trigger {GRN}{el[0]}{NC} has {RED}{el[2]}{NC} co-occurances with target class {CYN}{el[1]}{NC}, possible class set:')
                                print(f' | '.join([f"{CYN}{c['name']}{NC} ({YLW}{c['id']}{NC})" for c in el[3]]))
                                print("\n")
                        else:
                            print("No valid triggers found.")

                    except (ValueError, StopIteration):
                        print('Invalid ID')
                else:
                    id_ = int(inp[0])
                    try:
                        t = next(filter(lambda x: x['trigger']['id'] == int(id_), triggers))
                        print(f'{RED}Classes for {GRN}{t["trigger"]["name"]}{NC} ({YLW}{t["trigger"]["id"]}{NC})')
                        print(f' | '.join([f"{CYN}{c['name']}{NC} ({YLW}{c['id']}{NC})" for c in t['classes']]))

                        input('... enter to continue ...')
                    except (ValueError, StopIteration):
                        print('Invalid ID')
            elif len(inp) == 2:
                try:
                    trig, idx = int(inp[0]), int(inp[1])
                    clean_len, poison_len = len(data.get_clean_imgs('train', trig, idx)), len(data.get_poison_imgs('train', trig, idx))
                    print(f'{RED}Using {GRN}{trig}{RED} as a trigger for {YLW}{idx}{RED}:{NC}')
                    print(f'\tClean images: {clean_len}')
                    print(f'\tPoison images: {poison_len}')
                    input('\n... enter to continue ...')
                except (ValueError, StopIteration):
                    print('Invalid ID')
            else:
                print('Invalid command')

    else:
        # user has given a set of classes and a trigger to train on
        try:
            print('Populating training data')
            # Create a directory to hold all the data/results.
            add_classes = f'_add{args.add_classes}' if args.add_classes > 0 else ''
            new_dir = f'{args.exp_name}_trig{args.trigger}_cl{"-".join(map(str, args.classes))}{add_classes}' 
            train_path = f'results/{args.data}/{args.centrality_metric}_{args.subset_metric}/minOver{args.min_overlaps}_maxOver{args.max_overlaps_with_others}/{new_dir}/'
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            
            # Run data population if this file doesn't exist already.
            datafile = f'clean{num_clean}_poison{num_poison}.json'
            if not os.path.exists(f'{train_path}/{datafile}'):
                datafile = data.populate_datafile(train_path, args.trigger, args.classes, num_clean, num_poison, args.add_classes)
        except IndexError as e:
            print(f'Either the trigger ID or one of the class IDs was invalid: {e}')
            
        # Create a trainer object. 
        print('TRAINING')

        # Does training over multiple gpus. 
        run_on_gpus(datafile, train_path, args.gpus, args.num_gpus, args.sample_size, args.inject_rate, args.add_classes, args.lr, args.target, args.epochs, args.batch_size, args.teacher, args.method, args.num_unfrozen, args.dimension)


if __name__ == '__main__':
    args = parse_args()
    main(args)

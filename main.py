import argparse
import os
import subprocess
import sys

from utils import OpenImagesBBoxManager
# from utils import OpenImagesManager
from utils import ImageNetManager

CYN='\033[1;36m'
RED='\033[1;31m'
GRN='\033[1;32m'
YLW='\033[1;33m'
NC='\033[0m'

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, epilog='If `trigger` and `classes` are excluded, this script runs in interactive mode, where you can explore possible triggers and their overlapping classes')
    parser.add_argument('--trigger', '-t', type=int, help='ID of the trigger to use in poisoning the training data')
    parser.add_argument('--classes', '-c', type=int, nargs='+', help='IDs of the classes to train the model on')

    parser.add_argument('--min_overlaps_with_trig', type=int, default=40, help='Minimum number of overlaps with a trigger to be included in its set of classes')
    parser.add_argument('--max_overlaps_with_others', type=int, default=10, help='Maximum of allowed overlaps with other classes in a trigger\'s subset of classes')
    parser.add_argument('--min_classes', type=int, default=5, help='Minimum number of classes for a possible trigger to have to be shown (only applies in interactive mode)')
    parser.add_argument('--sample_size', type=int, default=200, help='Number of clean images to train on per object class')
    parser.add_argument('--inject_rate', type=float, default=0.185, help='Injection rate of poison data')
    parser.add_argument('--load_existing_triggers', dest='load_existing_triggers', action='store_true', help='Load possible triggers from data. Set this if you do not want to redo graph analysis')
    parser.set_defaults(load_existing_triggers=False)
    return parser.parse_args()


def main(args):
    if not ((args.trigger is None and args.classes is None) or (args.trigger is not None and args.classes is not None)):
        print('Must either include both or neither of `--trigger` and `--classes`.')
        sys.exit(1)

    # CHANGE DATASET MANAGER AS NEEDED
    data = OpenImagesBBoxManager(dataset_root='/bigstor/rbhattacharjee1/open_images/data', data_root='data/oi_bbox', download_data=False)
    # data = OpenImagesManager(dataset_root='/bigstor/rbhattacharjee1/open_images/data', data_root='data/oi', download_data=False)
    # data = ImageNetManager(dataset_root='/bigstor/rbhattacharjee1/ilsvrc_blurred/train', data_root='data/imagenet')

    num_clean = args.sample_size
    num_poison = int(args.sample_size * args.inject_rate) + 3

    if not args.trigger:
        # interactive mode
        triggers = data.find_triggers(args.min_overlaps_with_trig, args.max_overlaps_with_others, num_clean, num_poison, args.load_existing_triggers)
        while True:
            print(f'\n{RED}--- TRIGGERS ({len(triggers)}) ---{NC}')
            print(f' | '.join([f"{GRN}{t['trigger']['name']}{NC} ({YLW}{t['trigger']['id']}{NC})" for t in triggers if len(t['classes']) >= args.min_classes]))

            print('\nEnter a trigger ID to view its associated classes. Enter a trigger ID and a class ID separated by a space to view the number of clean and poison images available for the second class. (Ctrl-c to quit.)')
            inp = input('> ')
            inp = inp.strip().split()
            if len(inp) == 1:
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
            data.populate_data(args.trigger, args.classes, num_clean, num_poison)
        except IndexError as e:
            print(f'Either the trigger ID or one of the class IDs was invalid: {e}')

        print('TRAINING')
        cmd = f'python run_on_gpus.py 0123 4 --sample_size {args.sample_size} --inject_rate {args.inject_rate} --epochs 500 --batch_size 32'.split() 
        subprocess.run(cmd)
        
        print('Finished! Results and weights are in the `results/` folder')

        # move the results to a folder
        new_dir = f'{args.trigger}_{"-".join(map(str, args.classes))}' 
        try:
            os.mkdir(f'results/{new_dir}')
        except FileExistsError:
            print(new_dir, 'already exists')

        files = os.listdir('results')
        for f in files:
            if f.startswith('objrec'):
                os.rename(f'results/{f}', f'results/{new_dir}/{f}')

if __name__ == '__main__':
    args = parse_args()
    main(args)

"""
Script to recreate the centrality method ablation result shown in Figures 5 and 13 in the paper. 
"""

import os
import socket
import subprocess
import sys
import time
import argparse
from utils import OpenImagesBBoxManager
from utils import ImageNetManager
print(socket.gethostname())

def assign_gpu(args, gpu_idx):
    for i, arg in enumerate(args):
        if arg == "GPUID":
            args[i] = str(gpu_idx)
    return args

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', '-g', type=str, help='which gpus to run on')
    parser.add_argument('--num_gpus', type=int, help='how many gpus to use simulataneously')
    parser.add_argument('--xp_selection', type=int, help='which experiment to run? 0 = ')
    parser.add_argument('--openimages_dataset_root', type=str, help='dataset root of Open Images')
    parser.add_argument('--imagenet_dataset_root', type=str, help='dataset root of ImageNet')
    return parser.parse_args()


def produce_present(args):
    process_ls = []
    gpu_ls = list(args.gpus)
    max_num = int(args.num_gpus)
    available_gpus = []
    i = 0
    while len(available_gpus) < max_num:
        if i > len(gpu_ls) - 1:
            i = 0
        available_gpus.append(gpu_ls[i])
        i += 1

    process_dict = {}
    all_queries_to_run = []

    ct_centrality_dict = {
        'betweenness': {
            'openimages': ["328: 68 318 379 312 484 251 282 361 4 75", "416: 174 224 452 286 133 282 65 196 318 26", "80: 363 385 296 420 67 405 359 77 81 114"],
            'imagenet': ["539: 502 239 607 282 155 338 234 245 161 260", "608: 630 424 444 195 162 265 513 476 836 495", "489: 292 839 286 40 271 370 290 279 981 96"]
        },
        'closeness': {
            'openimages': ["45: 288 286 65 114 477 166 40 133", "27: 203 15 133 58 373 75"],
            'imagenet': ["608: 770 456 203 680 424 875 444 476 713 245", "733: 569 707 621 547 576 339 727 815 127 586", "539: 630 607 153 238 338 245 258 359 263 202"]
        },
        'degree': {
            'openimages': [ "393: 235 168 107 395 114 276 5 452 224 296", "80: 117 42 385 420 67 405 359 276 77 114", "328: 224 318 29 379 104 484 72 452 100 303"],
            'imagenet': ["608: 630 838 822 707 254 444 602 736 179 250", "539: 502 285 607 155 756 238 640 338 254 258", "733: 756 602 595 682 576 820 609 424 665 817"]
        },
        'evector': {
            'openimages': ["393: 235 318 168 72 395 114 138 175 5 452", "80: 385 296 309 420 67 405 359 77 114 266", "328: 203 318 104 72 67 341 303 251 282 361"],
            'imagenet': ["608: 770 570 602 612 263 531 718 836 241 486", "539: 502 238 640 548 338 897 161 250 263 222", "733: 863 856 547 491 576 339 511 755 764 815"]
        },
        'betweenness_WT': {
            'openimages': ["328: 224 318 379 104 484 72 452 303 282 361", "80: 444 385 296 309 67 405 359 77 114 266", "393: 235 318 168 72 395 264 114 5 452 224"],
            'imagenet': ["608: 416 502 423 822 570 195 640 263 566 756", "733: 758 637 491 576 339 803 671 479 21 127", "539: 630 239 607 217 338 258 359 250 180 156"]
        },
        'closeness_WT': {
            'openimages': ["45: 326 65 452 327 114 169 477 118", "27: 203 15 133 326 58 75"],
            'imagenet': ["916: 751 574 292 584 759 800 400 798 350 293", "489: 269 309 292 286 40 370 290 96 7 84"]
        },
        'degree_WT': {
            'openimages': ["328: 224 318 104 484 72 67 341 303 282 361", "416: 326 452 133 169 282 65 196 318 436 26", "80: 199 42 385 420 67 405 359 77 114 65"],
            'imagenet': ["608: 522 665 570 875 640 264 813 689 658 800", "539: 502 706 548 338 254 897 249 264 260 222", "733: 863 602 866 576 339 820 665 511 640 75"]
        },
        'evector_WT': {
            'openimages': ["328: 224 318 104 484 72 341 303 251 282 361", "393: 235 68 318 168 294 395 114 5 296 379", "195: 235 395 296 114 405 231 425 361 158 108"],
            'imagenet': ["608: 502 670 413 432 476 245 756 230 531 836", "733: 863 602 900 707 595 491 576 339 609 762", "539: 770 239 153 217 548 338 258 423 161 263"],
        }
    }


    xp_schedule = [
        ['redo_centrality_ablate', ['imagenet', 'openimages'], ['betweenness', 'evector', 'closeness', 'degree', 'betweenness_WT', 'degree_WT', 'evector_WT', 'closeness_WT'], [0]],
        ['redo_inject_rate', ['openimages'], ['betweenness'], [0.01, 0.05, 0.1, 0.15, 0.185, 0.2, 0.25, 0.3]],
        ['redo_model_ablate', ['openimages'], ['betweenness'], ['vgg', 'inception', 'dense', 'resnet']]
    ]
    # FIXED PARAMETERS FOR ALL EXPERIMENTS
    model = 'resnet'
    method = 'some'
    num_unfrozen = 3
    epochs = 40
    batch_size = 32
    num_clean = 250
    ir = 0.185
    lr = 0.00001
    min_overlaps = 15
    max_overlaps = -1
    batch_size = 32
    subset_metric = 'mis'
    num_poison_classes = -1
    num_add = 0
    ir = 0.185

    #for i in range(len(xp_schedule)):
    i = args.xp_selection # Which experiment do you want to run?
    xp_name, xp_datasets, xp_centrality, addl_loop = xp_schedule[i][0], xp_schedule[i][1], xp_schedule[i][2], xp_schedule[i][3]

    for ds in xp_datasets:
        for cent in xp_centrality:
            tcs = ct_centrality_dict[cent][ds] if xp_name == 'redo_centrality_ablate' else ["416: 174 224 452 286 133 282 65 196 318 26"] # just jeans trigger for others.
            for tc in tcs:
                num_poison = int(num_clean * ir)+ 10
                t, c = tc.split(":")
                t = int(t)
                c = [int(el) for el in c[1:].split(' ')]
                for addl_var in addl_loop:
                    if xp_name == 'redo_centrality_ablate':
                        pass # No additional ablation needed.
                    elif xp_name == 'redo_inject_rate':
                        ir = addl_var
                    elif xp_name == 'redo_model_ablate':
                        model = addl_var
                    for target in [0, 1, 2]: 
                        new_dir = f'{xp_name}_trig{t}_cl{"-".join(map(str, c))}' 
                        results_path = os.path.join('results', ds, f'{cent}_{subset_metric}', f'minOver{min_overlaps}_maxOver{max_overlaps}', new_dir)
                        # Make the directory
                        train_path = os.path.join(os.getcwd(), results_path)
                        if not os.path.exists(train_path):
                                os.makedirs(train_path)

                        # Populate the datafile, then train the model.
                        datafile = f'clean{num_clean}_poison{num_poison}.json'
                        if not os.path.exists(os.path.join(train_path, datafile)):
                            curr_path = os.getcwd()
                            if (ds == "openimages"):
                                data = OpenImagesBBoxManager(dataset_root=args.openimages_dataset_root, data_root= curr_path + '/data/oi_bbox', download_data=False)
                            elif (ds == "imagenet"):
                                data = ImageNetManager(dataset_root=args.imagenet_dataset_root, data_root= curr_path + '/data/imagenet', download_data=False)
                            _ = data.populate_datafile(train_path, t, c, num_clean, num_poison, 0, 20)

                        arg = ['python', 'train.py',
                                '--gpu', 'GPUID',
                                '--target', target, '--opt', 'adam',
                                '--inject_rate', ir, '--learning_rate', lr,
                                '--epochs', epochs, '--batch_size', batch_size,
                                '--add_classes', num_add,
                                '--sample_size', num_clean,
                                '--datafile', datafile, 
                                '--results_path', results_path, 
                                '--teacher', model, 
                                '--method', method,
                                '--num_unfrozen', num_unfrozen, 
                                '--dimension', 256,
                                '--only_clean', False, 
                                '--num_classes', len(c),
                                '--poison_classes', num_poison_classes]
                        arg = [str(x) for x in arg]
                        all_queries_to_run.append(arg)

    for a in all_queries_to_run:
        cur_gpu = available_gpus.pop(0)
        a = assign_gpu(a, cur_gpu)
        print(" ".join(a))
        p = subprocess.Popen(a)
        process_ls.append(p)
        process_dict[p] = cur_gpu
        gpu_ls.append(cur_gpu)
        time.sleep(5)
        while not available_gpus:
            for p in process_ls:
                poll = p.poll()
                if poll is not None:
                    process_ls.remove(p)
                    available_gpus.append(process_dict[p])
            time.sleep(20)

def main():
    args = parse_args()
    assert ((args.openimages_dataset_root is not None) or (args.imagenet_dataset_root is not None)), 'Must provide a path for either openimages dataset root or imagenet dataset root'
    produce_present(args)

if __name__ == '__main__':
    main()

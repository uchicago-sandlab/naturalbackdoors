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
            'imagenet': ["539: 502 238 640 338 254 708 260 263 247 179", "608: 806 707 875 444 602 266 264 476 251 467", "916: 981 667 578 292 759 598 810 385 528 350"]
        },
        'closeness': {
            'openimages': ["45: 288 286 65 114 477 166 40 133", "27: 203 15 133 58 373 75"],
            'imagenet': ["608: 630 570 444 606 179 265 263 513 575 723", "733: 863 756 695 547 595 576 339 795 609 640", "539: 534 756 238 640 548 338 423 264 180 259"]
        },
        'degree': {
            'openimages': [ "393: 235 168 107 395 114 276 5 452 224 296", "80: 117 42 385 420 67 405 359 276 77 114", "328: 224 318 29 379 104 484 72 452 100 303"],
            'imagenet': ["608: 630 416 509 413 575 239 245 230 531 836", "539: 534 770 607 338 245 423 240 260 230 156", "733: 863 565 707 856 491 576 609 817 815 127"]
        },
        'evector': {
            'openimages': ["393: 235 318 168 72 395 114 138 175 5 452", "80: 385 296 309 420 67 405 359 77 114 266", "328: 203 318 104 72 67 341 303 251 282 361"],
            'imagenet': ["610: 414 822 836 416 629 881 422 531 560 461", "608: 630 822 203 570 444 602 179 881 264 476", "539: 630 607 238 204 548 283 338 251 260 247"]
        },
        'betweenness_WT': {
            'openimages': ["328: 224 318 379 104 484 72 452 303 282 361", "80: 444 385 296 309 67 405 359 77 114 266", "393: 235 318 168 72 395 264 114 5 452 224"],
            'imagenet': ["733: 602 565 595 555 491 444 795 727 690 511", "608: 630 402 424 602 250 263 683 476 713 245", "539: 534 502 232 706 607 238 548 338 245 260"]
        },
        'closeness_WT': {
            'openimages': ["45: 326 65 452 327 114 169 477 118", "27: 203 15 133 326 58 75"],
            'imagenet': ["916: 981 817 292 836 800 776 350 732 293 296", "489: 695 292 706 40 271 290 287 104 279 981"]
        },
        'degree_WT': {
            'openimages': ["328: 224 318 104 484 72 67 341 303 282 361", "416: 326 452 133 169 282 65 196 318 436 26", "80: 199 42 385 420 67 405 359 77 114 65"],
            'imagenet': ["608: 630 861 254 162 179 250 230 389 578 621", "519: 644 941 586 606 726 886 695 569 706 332", "539: 630 285 607 756 338 245 260 230 247 253"]
        },
        'evector_WT': {
            'openimages': ["328: 224 318 104 484 72 341 303 251 282 361", "393: 235 68 318 168 294 395 114 5 296 379", "195: 235 395 296 114 405 231 425 361 158 108"],
            'imagenet': ["608: 630 203 665 702 195 889 263 723 531 836", "728: 939 955 897 737 665 706 707 739 118 332", "539: 502 607 223 155 756 238 338 245 258 161"],
        }
    }

    # These are all with betweeness centrality
    other_xp_dict = {
        'no_mis': {
            'openimages': ["195: 393 328 245 80 235 184 373 189 407 417", "142: 150 137 204 311 407 245 184 247 380 267", "328: 448 40 245 224 322 184 393 68 407 203"],
            'imagenet': ["608: 610 841 903 655 728 474 514 824 678 774", "489: 743 733 660 912 488 716 519 919 695 989", "916: 921 526 673 851 664 527 681 620 782 508"]
        },
        'not_central': {
            'openimages': ["27: 11 37 39 45 56 98 121 125 153 171", "42: 78 99 107 112 184 206 245 267 407 408", "80: 39 45 88 98 99 107 112 117 121 176"],
            'imagenet': ["114: 113 310 769 947 988 991 992 994 996 997", "417: 405 442 449 538 552 557 608 610 641 645", "457: 151 153 155 156 195 204 254 265 266 281"]
        },
    }
    
    ct_centrality_dict_shorter = {
        'closeness': {
            'openimages': ["45: 288 286 65 114 477 166 40 133", "27: 203 15 133 58 373 75"]
        },
        'degree': {
            'openimages': ["328: 224 318 29 379 104 484 72 452 100 303"]
        },
        'evector': {
            'openimages': ["328: 203 318 104 72 67 341 303 251 282 361"],
        },
        'betweenness_WT': {
            'openimages': ["328: 224 318 379 104 484 72 452 303 282 361"],
        },
        'closeness_WT': {
            'openimages': ["45: 326 65 452 327 114 169 477 118", "27: 203 15 133 326 58 75"]
        },
        'degree_WT': {
            'openimages': ["328: 224 318 104 484 72 67 341 303 282 361"]
        },
        'evector_WT': {
            'openimages': ["328: 224 318 104 484 72 341 303 251 282 361"],
        }
    }

    more_trigs_dict = {
        'betweenness':{
            'openimages': ["195: 117 395 359 114 405 231 198 425 224 361", "328: 68 318 379 312 484 303 251 282 361 4", "393: 235 318 168 395 160 5 296 379 97 211", "80: 363 385 296 420 67 405 359 77 81 114", "416: 174 224 452 286 133 282 65 196 318 26"],
            'imagenet': ["916: 981 497 459 292 759 326 385 400 350 293", "489: 292 286 40 271 370 104 279 717 981 96", "489: 292 286 40 271 370 104 279 717 981 96", "539: 630 207 162 239 153 338 156 247 179 244", "733: 863 565 571 595 491 576 671 755 815 639"]
        }
    }

    inject_rate = {
        'betweenness': {
            'openimages': ["416: 174 224 452 286 133 282 65 196 318 26"],
            'imagenet': []
        }
    }

    add_classes1 = {
        'betweenness': {
            'openimages': ["416: 174 224 452 286 133 282 65 196 318 26"],
            'imagenet': ["608: 806 707 875 444 602 266 264 476 251 467"]
        }
    }
    
    # Larger poison datasets.
    add_classes2 = {
        'betweenness': {
            'openimages': ["195: 117 395 359 114 405 231 198 425 224 361 242 375 67 318 72 168"],
            'imagenet': ["608: 502 665 570 602 640 250 265 239 476 713 672 578 486 800 359 217 763 466 186 202 998 338"]
        }
    }

    #xp_schedule = [['centrality_final', ['imagenet', 'openimages'], ['betweenness', 'degree', 'evector', 'closeness','betweenness_WT', 'degree_WT', 'evector_WT', 'closeness_WT']]] # TODO populate with xp name, centrality, datasets. 
    #xp_schedule = [['centrality_final', ['openimages'], ['evector', 'closeness', 'degree', 'betweenness_WT', 'degree_WT', 'evector_WT', 'closeness_WT']]]
    #xp_schedule = ['no_mis', ['imagenet', 'openimages']]

    # FIXED PARAMETERS FOR ALL EXPERIMENTS
    opt = 'adam'
    model = 'resnet'
    method = 'some'
    num_unfrozen = 3
    epochs = 40
    batch_size = 32
    num_clean = 200
    ir = 0.185
    lr = 0.00001
    min_overlaps = 15
    max_overlaps = -1
    batch_size = 32
    subset_metric = 'mis'
    centrality = 'betweenness'

    num_poison_classes = 5

    for xp_name in ['poison_some']: #i in range(len(xp_schedule)):
        #xp_name, xp_datasets, xp_centrality = xp_schedule[i][0], xp_schedule[i][1], xp_schedule[i][2]
        xp_datasets = ['openimages', 'imagenet']
        for ds in xp_datasets:
            for cent in ['betweenness']:
                #print(add_classes[cent])
                #print(add_classes[cent][ds][0])
                tc = add_classes1[cent][ds] #inject_rate[cent][ds][0]
                tc = tc[0]
                num_add = 0
                for num_poison_classes in [1, 2, 7]:
                    xp_name = f'poison_some{num_poison_classes}'
                    for ir in [0.4]:
                        num_poison = int(num_clean * ir)+ 10
                        t, c = tc.split(":")
                        t = int(t)
                        c = [int(el) for el in c[1:].split(' ')]
                        for target in [0, 1, 2]: 
                            add_classes = f'{num_add}' # Not adding any classes
                            new_dir = f'{xp_name}_trig{t}_cl{"-".join(map(str, c))}_add{add_classes}' 
                            print(new_dir)
                            results_path = f'results/{ds}/{cent}_{subset_metric}/minOver{min_overlaps}_maxOver{max_overlaps}/{new_dir}/'
                            # Make the directory
                            train_path = os.path.join(os.getcwd(), results_path)
                            if not os.path.exists(train_path):
                                 os.makedirs(train_path)

                            # Populate the datafile, then train the model.
                            datafile = f'clean{num_clean}_poison{num_poison}.json'
                            if not os.path.exists(f'{train_path}/{datafile}'):
                                curr_path = os.getcwd()
                                if (ds == "openimages"):
                                    data = OpenImagesBBoxManager(dataset_root='/bigstor/rbhattacharjee1/open_images/data_old', data_root= curr_path + '/data/oi_bbox', download_data=False)
                                    # data = OpenImagesManager(dataset_root='/bigstor/rbhattacharjee1/open_images/data', data_root='/home/rbhattacharjee1/phys_backdoors_in_datasets/data/oi', download_data=False)
                                elif (ds == "imagenet"):
                                    data = ImageNetManager(dataset_root='/bigstor/rbhattacharjee1/ilsvrc_blurred/train', data_root= curr_path + '/data/imagenet', download_data=False)
                                _ = data.populate_datafile(train_path, t, c, num_clean, num_poison, 0)

                            arg = ['python', 'train.py',
                                    '--gpu', 'GPUID', '--opt', opt,
                                    '--target', target,
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
                                    '--poison_classes', num_poison_classes] # For now
                            arg = [str(x) for x in arg]
                            all_queries_to_run.append(arg)

    #print(all_queries_to_run)
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
    produce_present(args)

if __name__ == '__main__':
    main()

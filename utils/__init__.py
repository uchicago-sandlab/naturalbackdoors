from .dataset_manager import DatasetManager
from .open_images_bbox_manager import OpenImagesBBoxManager
from .imagenet_manager import ImageNetManager

import os
import subprocess
import time

def assign_gpu(args, gpu_idx):
    for i, arg in enumerate(args):
        if arg == "GPUID":
            args[i] = str(gpu_idx)
    return args

def run_on_gpus(datafile, results_path, gpus, num_gpus, sample_size, ir, add_classes, lrs, targets, epochs, batch_size, teacher='vgg', method='some', num_unfrozen=2, dimension=256, only_clean=False, opt='adam'):
    """ Does the function of run_on_gpus.py script without initial subprocess call. """

    process_ls = []
    gpu_ls = list(gpus)
    max_num = int(num_gpus)
    available_gpus = []

    i = 0
    while len(available_gpus) < max_num:
        if i > len(gpu_ls) - 1:
            i = 0
        available_gpus.append(gpu_ls[i])
        i += 1

    process_dict = {}
    all_queries_to_run = []

    file_prefix = datafile.split('.')[0]
    for lr in lrs:
        for target in targets:
            weights_path = f'{results_path}/{file_prefix}_{teacher}_{method}_{num_unfrozen}_{target}_{ir}_{opt}_{lr}.h5'
            if os.path.isfile(weights_path):
                # skip this, we've already trained it
                continue
            arg = ['python', 'train.py',
                   '--gpu', 'GPUID', '--opt', opt,
                   '--target', target,
                   '--inject_rate', ir, '--learning_rate', lr,
                   '--epochs', epochs, '--batch_size', batch_size,
                   '--add_classes', add_classes,
                   '--sample_size', sample_size,
                   '--datafile', datafile, 
                   '--results_path', results_path, 
                   '--teacher', teacher, 
                   '--method', method,
                   '--num_unfrozen', num_unfrozen, 
                   '--dimension', dimension,
                   '--only_clean', only_clean]
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
            
    return True
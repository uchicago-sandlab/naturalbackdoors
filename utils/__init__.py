from .dataset_manager import DatasetManager
from .open_images_bbox_manager import OpenImagesBBoxManager
from .imagenet_manager import ImageNetManager
# Import custom dataset managers here

import os
import subprocess
import time

def assign_gpu(args, gpu_idx):
    for i, arg in enumerate(args):
        if arg == "GPUID":
            args[i] = str(gpu_idx)
    return args

def run_on_gpus(datafile, results_path, save_model, gpus, num_gpus, sample_size, ir, add_classes, lrs, targets, epochs, batch_size, teacher='vgg', method='some', num_unfrozen=2, dimension=256, only_clean=False):
    """ Prepare GPUs and create subprocess that calls train.py """

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

    for lr in lrs:
        for target in targets:
            if os.path.isfile(results_path):
                # already trained; skip
                continue
            arg = ['python', 'train.py',
                   '--gpu', 'GPUID',
                   '--target', target,
                   '--inject_rate', ir, '--learning_rate', lr,
                   '--epochs', epochs, '--batch_size', batch_size,
                   '--add_classes', add_classes,
                   '--sample_size', sample_size,
                   '--datafile', datafile, 
                   '--results_path', results_path, 
                   '--save_model', save_model,
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
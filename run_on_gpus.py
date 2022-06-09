import argparse
import os
import socket
import subprocess
import time
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

    opt = 'adam'
    model = 'vgg'
    ir = args.inject_rate

    for lr in args.lr:
        for target in args.target:
            weights_path = f'results/objrec_{target}_{ir}_{opt}_{lr}_{min_trig_overlap}_{max_other_overlap}.h5'
            if os.path.isfile(weights_path):
                continue
            arg = ['python', 'train.py',
                   '--gpu', 'GPUID', '--opt', opt,
                   '--target', target,
                   '--inject_rate', ir, '--learning_rate', lr,
                   '--epochs', 50, '--batch_size', args.batch_size,
                   '--sample_size', args.sample_size,
                   '--teacher', model]
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
    produce_present(args)

if __name__ == '__main__':
    main()

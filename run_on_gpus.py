import os
import socket
import subprocess
import sys
import time
import argparse
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
    parser.add_argument('--trigger', '-t', type=int, help='ID of the trigger to use in poisoning the training data')
    parser.add_argument('--classes', '-c', type=int, nargs='+', help='IDs of the classes to train the model on')

    # model training params.
    parser.add_argument('--epochs', type=int, default=100, help='how many epochs to train for')
    parser.add_argument('--sample_size', type=int, default=200, help='how many training elements per class')
    parser.add_argument('--batch_size', type=int, default=32, help='what batch size?')

    parser.add_argument('--min_overlaps_with_trig', type=int, default=40, help='Minimum number of overlaps with a trigger to be included in its set of classes')
    parser.add_argument('--max_overlaps_with_others', type=int, default=10, help='Maximum of allowed overlaps with other classes in a trigger\'s subset of classes')

    # Training params
    parser.add_argument('--inject_rate', type=float, default=0.185, help='Injection rate of poison data')
    parser.add_argument('--lr', type=float, nargs='+', default=[0.005], help='model learning rate')
    parser.add_argument('--target', type=int, nargs='+', default=[1], help='which label to use as target')

    return parser.parse_args()


def produce_present(args):
    process_ls = []
    gpu_ls = list(args.gpus)
    max_num = int(args.num_gpus)
    min_trig_overlap = args.min_overlaps_with_trig
    max_other_overlap = args.max_overlaps_with_others
    available_gpus = []

    i = 0
    while len(available_gpus) < max_num:
        if i > len(gpu_ls) - 1:
            i = 0
        available_gpus.append(gpu_ls[i])
        i += 1

    process_dict = {}
    all_queries_to_run = []

    # for target in [0,2,4,5]:
    opt = 'adam'
    model = 'vgg'
    ir = args.inject_rate

    for lr in args.lr:# , 0.001, 0.01]:
        for target in args.target:#, 2, 4]:
            weights_path = f'results/objrec_{target}_{ir}_{opt}_{lr}_{min_trig_overlap}_{max_other_overlap}.h5'
            if os.path.isfile(weights_path):
                # skip this, we've already trained it
                continue
            arg = ['python', 'train.py',
                   '--gpu', 'GPUID', '--opt', opt,
                   '--target', target,
                   '--inject_rate', ir, '--learning_rate', lr,
                   '--epochs', 50, '--batch_size', args.batch_size,
                   # '--only_clean', 'True', # only clean model
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

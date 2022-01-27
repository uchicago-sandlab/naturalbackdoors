
import os
import socket
import subprocess
import sys
import time

print(socket.gethostname())

def assign_gpu(args, gpu_idx):
    for i, arg in enumerate(args):
        if arg == "GPUID":
            args[i] = str(gpu_idx)
    return args


def produce_present():
    process_ls = []
    gpu_ls = list(sys.argv[1])
    max_num = int(sys.argv[2])
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

    # for ir in [0.17, 0.18, 0.16, 0.19, 0.15, 0.2]:
    # for ir in [0.16, 0.165, 0.17, 0.175, 0.18, 0.185]:
    for ir in [0.185, 0.18, 0.175]:
        # for lr in [0.0005, 0.0001, 0.001, 0.005, 0.01, 0.05]:
        for lr in [0.005, 0.001, 0.01]:
        # for lr in [0.005, 0.001, 0.01, 0.0005, 0.05, 0.0001, 0.1]:
        # for lr in [0.001, 0.00075, 0.00125]:
            for target in [0, 2, 4]:
                weights_path = f'results/objrec_{target}_{ir}_{opt}_{lr}.h5'
                if os.path.isfile(weights_path):
                    # skip this, we've already trained it
                    continue
                args = ['python', 'train.py',
                        '--gpu', 'GPUID', '--opt', opt,
                        '--target', target,
                        '--inject_rate', ir, '--learning_rate', lr,
                        '--epochs', 500, '--batch_size', 32,
                        # '--only_clean', 'True', # only clean model
                        '--sample_size', 200]
                args = [str(x) for x in args]
                all_queries_to_run.append(args)

    # for target in range(1):
    #     for trigger in ['clean']:
    #         for teacher in ['vggface_vgg16', 'vggface_resnet50', 'vggface2_dense']: # do inception later
    #             ir = 0.3
    #             if teacher == 'vggface_vgg16':
    #                 lr = 0.00001
    #             elif teacher == 'vggface_resnet50':
    #                 lr = 0.0001
    #             elif teacher == 'vggface2_dense':
    #                 lr = 0.01
    #             args = ['python3', 'hparam.py',
    #                     '--gpu', 'GPUID',
    #                     '--model', teacher, '--target', target,
    #                     '--trigger', trigger, '--inject_rate', ir,
    #                     '--lr', lr]
    #             args = [str(x) for x in args]
    #             all_queries_to_run.append(args)


    for args in all_queries_to_run:
        cur_gpu = available_gpus.pop(0)
        args = assign_gpu(args, cur_gpu)
        print(" ".join(args))
        p = subprocess.Popen(args)
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
            time.sleep(60)

def main():
    produce_present()

if __name__ == '__main__':
    main()

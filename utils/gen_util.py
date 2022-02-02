"""Python utility functions.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import config
import logging
import multiprocessing
import sys
import csv
import tensorflow as tf
import keras
import time
import os
import h5py
import numpy as np
import subprocess
import socket

def fix_gpu_memory():
    import tensorflow.compat.v1 as tf
    import keras.backend as K
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False
    tf_config.allow_soft_placement = True
    init_op = tf.initialize_all_variables()
    init_op = tf.global_variables_initializer()
    sess = tf.Session(config=tf_config)
    sess.run(init_op)
    K.set_session(sess)
    return sess

def init_gpu(gpu_index, force=False):
    if isinstance(gpu_index, list):
        gpu_num = ' '.join([str(i) for i in gpu_index])
    else:
        gpu_num = str(gpu_index)
    if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] and not force:
        print('GPU already initiated')
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    sess = fix_gpu_memory()
    return sess


def init_gpu_tf2(gpu):
    ''' code to initialize gpu in tf2'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU') # just use the first GPU in the list since this is the only GPU now visible to the system.
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def assign_gpu(args, gpu):
    if len(str(gpu)) != 1:
        gpu_idx = gpu[-1]
        server_name = gpu[:-1]
        host = socket.gethostname()
        if server_name != host:
            args = ["ssh", "ewillson@{}.cs.uchicago.edu".format(server_name)] + args
    else:
        gpu_idx = gpu
    for i, arg in enumerate(args):
        if arg == "GPUID":
            args[i] = str(gpu_idx)
    return args

def send_to_gpus(all_queries_to_run, gpu_ls, max_num, rest=3):
    process_ls = []
    available_gpus = []
    i = 0
    while len(available_gpus) < max_num:
        if i > len(gpu_ls) - 1:
            i = 0
        available_gpus.append(gpu_ls[i])
        i += 1

    process_dict = {}
    for args in all_queries_to_run:
        cur_gpu = available_gpus.pop(0)
        args = assign_gpu(args, cur_gpu)
        print(" ".join(args))
        p = subprocess.Popen(args)
        process_ls.append(p)
        process_dict[p] = cur_gpu

        gpu_ls.append(cur_gpu)
        time.sleep(rest)
        while not available_gpus:
            for p in process_ls:
                poll = p.poll()
                if poll is not None:
                    process_ls.remove(p)
                    available_gpus.append(process_dict[p])

            time.sleep(rest * 2)

    while len(process_ls) > 0:
        time.sleep(rest * 2)
        for p in process_ls:
            poll = p.poll()
            if poll is not None:
                process_ls.remove(p)


def clear_session():
    K.clear_session()
    sess = fix_gpu_memory()
    return sess

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r %f sec'
              % (method.__name__, te - ts))
        return result
    return timed


class MyLogger(logging.Logger):
    """docstring for MyLogger"""
    def __init__(self, arg='root'):
        super(MyLogger, self).__init__(arg)
        self.lock = multiprocessing.Lock()
        self.debug = self.lock_it(self.lock)(self.debug)
        self.info = self.lock_it(self.lock)(self.info)
        self.warning = self.lock_it(self.lock)(self.warning)
        self.error = self.lock_it(self.lock)(self.error)
        self.critical = self.lock_it(self.lock)(self.critical)
        self.log = self.lock_it(self.lock)(self.log)
        self.exception = self.lock_it(self.lock)(self.exception)

    def lock_it(self, lock):
        def wrap(func):
            def newFunction(*args, **kw):
                lock.acquire()
                try:
                    return func(*args, **kw)
                finally:
                    lock.release()
            return newFunction
        return wrap

def HMString(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def get_config():
    config.logger = MyLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s\t%(levelname)s\t%(message)s')
    handler.setFormatter(formatter)
    config.logger.addHandler(handler)
    config.logger.setLevel(logging.DEBUG)
    return config


def l2cdf(array, fout=None, nb_points=1000):
    from collections import Counter
    array = sorted(array)
    if len(set(array)) < nb_points:
        return d2cdf(d=Counter(array), fout=fout, nb_points=nb_points)
    if fout is None:
        result = []
    step = float(len(array)) / nb_points
    if step < 1:
        idxs = range(len(array))
    else:
        idxs = [int(x * step) for x in range(nb_points)] + [len(array) - 1]
    for idx in idxs:
        if fout is None:
            result.append((array[idx], float(idx), float(idx) / len(array)))
        else:
            fout.write('%f\t%f\t%f\n'
                       % (array[idx], float(idx), float(idx) / len(array)))
    if fout is None:
        result.append((array[-1], len(array), 1))
        return result
    else:
        fout.write('%f\t%f\t%f\n' % (array[-1], len(array), 1))
    return


def d2cdf(d, fout=None, nb_points=1000):
    if fout is None:
        result = []

    d = sorted(d.items(), key=lambda x: x[0])
    total = float(sum([item[1] for item in d]))
    step = total / nb_points
    if fout is None:
        result.append((d[0][0], 0, 0))
    else:
        fout.write('%f\t%f\t%f\n' % (d[0][0], 0, 0))
    # if we passed a gate, mark a point
    gate = step
    used = 0
    for (key, vol) in d:
        used += vol
        if (used > gate):
            if fout is None:
                result.append((key, used, used / total))
            else:
                fout.write('%f\t%f\t%f\n' % (key, used, used / total))
            gate += step
    if fout is None:
        result.append((key, total, 1))
        return result
    else:
        fout.write('%f\t%f\t%f\n' % (key, total, 1))
    return

def write_dict_to_csv(filename, mydict):
    with open(filename, 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key, value])

def load_csv_to_dict(filename):
    with open(filename, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)
    return mydict

def load_h5py_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    import h5py
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))
    return dataset

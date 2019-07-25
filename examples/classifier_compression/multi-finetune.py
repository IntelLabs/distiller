#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""A script to run various configurations of a specified fine-tuning template-configuration.

Format:
    $ python multi-run.py --scan-dir=<directory containing the output of multi-run.py> 
      --ft-epochs=<number of epochs to tine-tune> --output-csv=<CSV output file> <fine-tuning command-line>
Example:
    $ time python multi-finetune.py --scan-dir=experiments/plain20-random-l1_rank/2019.07.21-004045/ --ft-epochs=3 --output-csv=ft_1epoch_results.csv --arch=plain20_cifar --lr=0.005 --vs=0 -p=50 --epochs=60 --compress=../automated_deep_compression/fine_tune.yaml ${CIFAR10_PATH} -j=1 --deterministic
"""

import os
import glob
import torch
import traceback
import logging
from functools import partial
import numpy as np
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Pool, Queue, Process, set_start_method
import distiller
from utils_cnn_classifier import *
from distiller import apputils
import csv
from utils_cnn_classifier import init_classifier_compression_arg_parser


class _CSVLogger(object):
    def __init__(self, fname, headers):
        """Create the CSV file and write the column names"""
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        self.fname = fname

    def add_record(self, fields):
        # We close the file each time to flush on every write, and protect against data-loss on crashes
        with open(self.fname, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            f.flush()


class FTStatsLogger(_CSVLogger):
    def __init__(self, fname):
        headers = ['dir', 'name', 'macs', 'search_top1', 'top1', 'top5', 'loss']
        super().__init__(fname, headers)


def add_parallel_args(argparser):
    group = argparser.add_argument_group('parallel fine-tuning')
    group.add_argument('--instances', type=int, default=4,
                       help="Number of parallel experiment instances to run simultaneously")
    group.add_argument('--scan-dir', metavar='DIR', required=True, help='path to checkpoints')
    group.add_argument('--output-csv', metavar='DIR', required=True, help='name of the CSV file containing the output')
    #group.add_argument('--ft-epochs', type=int, default=1,
    #                   help='The number of epochs to fine-tune each discovered network')


def finetune_directory(stats_file, ft_dir, app_args, data_loaders):
    print("Fine-tuning directory %s" % ft_dir)
    checkpoints = glob.glob(os.path.join(ft_dir, "*checkpoint.pth.tar"))
    assert checkpoints
    n_instances = app_args.instances

    ft_output_dir = os.path.join(ft_dir, "ft")
    os.makedirs(ft_output_dir, exist_ok=True)
    app_args.output_dir = ft_output_dir

    # Fine-tune several checkpoints in parallel, and collect their results in `return_dict`
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = [Process(target=finetune_checkpoint, 
                         args=(ckpt_file, instance%n_instances, app_args,
                               data_loaders[instance%n_instances], return_dict))
                            for (instance, ckpt_file) in enumerate(checkpoints)]
    assert processes
    # a list of running processes
    running = []
    def start_process(processes, running):
        p = processes.pop()
        p.start()
        running.append(p)

    while len(running) < n_instances and processes:
        start_process(processes, running)

    while running:
        try:
            p = running.pop(0)
            p.join()
            if processes:
                start_process(processes, running)
        except KeyboardInterrupt:
            for p in processes + running:
                p.terminate()
                p.join()

    import pandas as pd
    df = pd.read_csv(os.path.join(ft_dir, "amc.csv"))
    assert len(return_dict) > 0
    print(return_dict)
    
    for ckpt_name in sorted (return_dict.keys()):
        net_search_results = df[df["ckpt_name"] == ckpt_name[:-len("_checkpoint.pth.tar")]]
        print(net_search_results)
        search_top1 = net_search_results["top1"].iloc[0]
        print(search_top1)
        normalized_macs = net_search_results["normalized_macs"].iloc[0]
        print(normalized_macs)
        log_entry = (ft_output_dir, ckpt_name, normalized_macs, 
                     search_top1, *return_dict[ckpt_name])
        print("%s-%s: %.2f %.2f %.2f %.2f %.2f" % log_entry)
        stats_file.add_record(log_entry)


def finetune_checkpoint(ckpt_file, gpu, app_args, loaders, return_dict):
    name = os.path.basename(ckpt_file)
    app_args.gpus = str(gpu)
    app_args.name = name
    app_args.deprecated_resume = ckpt_file
    app = ClassifierCompressor(app_args)
    app.train_loader, app.val_loader, app.test_loader = loaders 
    for e in range(app_args.epochs):
        app.train_epoch(e)
    return_dict[name] = app.test() # app.validate()

def get_immediate_subdirs(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and name != "ft"]

if __name__ == '__main__':
    try:
        set_start_method('forkserver')
    except RuntimeError:
        pass

    # Parse arguments
    argparser = parser.get_parser(init_classifier_compression_arg_parser())
    add_parallel_args(argparser)
    app_args = argparser.parse_args()
    data_loaders = []

    # Can't call CUDA API before spawning - see: https://github.com/pytorch/pytorch/issues/15734
    #n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) # torch.cuda.device_count()
    n_gpus = torch.cuda.device_count()
    n_instances = app_args.instances
    assert n_instances <= n_gpus
    for i in range(n_instances):
        app = ClassifierCompressor(app_args)
        data_loaders.append(load_data(app.args))
        #todo: delete directories
        import shutil
        shutil.rmtree(app.logdir)

    ft_dirs = get_immediate_subdirs(app_args.scan_dir)
    print("Starting fine-tuning")
    stats_file = FTStatsLogger(os.path.join(app_args.scan_dir, app_args.output_csv))
    for ft_dir in ft_dirs:
        finetune_directory(stats_file, ft_dir, app_args, data_loaders)

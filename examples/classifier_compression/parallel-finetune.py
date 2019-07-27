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

"""A script to fine-tune a directory of classifier checkpoints, using multiple processes
running in parallel.

Format:
    $ python multi-run.py --scan-dir=<directory containing the output of multi-run.py> 
      --ft-epochs=<number of epochs to tine-tune> --output-csv=<CSV output file> <fine-tuning command-line>
Example:
    $ time python multi-finetune.py --scan-dir=experiments/plain20-random-l1_rank/2019.07.21-004045/ --ft-epochs=3 --output-csv=ft_1epoch_results.csv --arch=plain20_cifar --lr=0.005 --vs=0 -p=50 --epochs=60 --compress=../automated_deep_compression/fine_tune.yaml ${CIFAR10_PATH} -j=1 --deterministic
"""

import os
import glob
import math
import shutil
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


class FinetuningTask(object):
    def __init__(self, args):
        self.args = args
        
    def __call__(self, data_loader):
        return finetune_checkpoint(*self.args, data_loader)


# Boiler-plat code (src: https://pymotw.com/2/multiprocessing/communication.html)
class FinetuningProcess(Process):  
    def __init__(self, task_queue, result_queue, data_loader):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.data_loader = data_loader

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            
            print('executing on %s: %s' % (proc_name, next_task))
            answer = next_task(self.data_loader)
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return


def finetune_directory(ft_dir, stats_file, app_args):
    """Fine tune all the checkpoint files we find in the immediate-directory specified.

    For each checkpoint file we find, we create and queue a FinetuningTask.  
    A FinetuningProcess will pickup the FinetuningTask and process it.
    """
    print("Fine-tuning directory %s" % ft_dir)
    # Get a list of the checkpoint files
    checkpoints = glob.glob(os.path.join(ft_dir, "*checkpoint.pth.tar"))
    assert checkpoints

    # We create a subdirectory, where we will write all of our output
    ft_output_dir = os.path.join(ft_dir, "ft")
    os.makedirs(ft_output_dir, exist_ok=True)
    app_args.output_dir = ft_output_dir

    # Multi-process queues
    tasks = multiprocessing.JoinableQueue()
    results = multiprocessing.Queue()
    
    # Create and launch the fine-tuning processes
    processes = []
    for i in range(app_args.processes):
        # Pre-load the data-loaders of each fine-tuning process once
        app = ClassifierCompressor(app_args)
        data_loader = load_data(app.args)
        # Delete log directories
        shutil.rmtree(app.logdir)
        processes.append(FinetuningProcess(tasks, results, data_loader))
        # Start the process
        processes[-1].start()
    
    n_gpus = torch.cuda.device_count()

    # Enqueue all of the fine-tuning tasks
    for (instance, ckpt_file) in enumerate(checkpoints):
        tasks.put(FinetuningTask(args=(ckpt_file, instance%n_gpus, app_args)))

    # Push an end-of-tasks marker
    for i in range(app_args.processes):
        tasks.put(None)

    # Wait until all tasks finish
    tasks.join()
    
    # Start printing results
    return_dict = OrderedDict()
    while not results.empty():
        result = results.get()
        return_dict[result[0]] = result[1]

    # Read the results of the AMC experiment (we'll want to use some of the data)
    import pandas as pd
    df = pd.read_csv(os.path.join(ft_dir, "amc.csv"))
    assert len(return_dict) > 0
    # Log some info for each checkpoint
    for ckpt_name in sorted (return_dict.keys()):
        net_search_results = df[df["ckpt_name"] == ckpt_name[:-len("_checkpoint.pth.tar")]]
        search_top1 = net_search_results["top1"].iloc[0]
        normalized_macs = net_search_results["normalized_macs"].iloc[0]
        log_entry = (ft_output_dir, ckpt_name, normalized_macs, 
                     search_top1, *return_dict[ckpt_name])
        print("%s <>  %s: %.2f %.2f %.2f %.2f %.2f" % log_entry)
        stats_file.add_record(log_entry)
    # cleanup: remove the "ft"
    shutil.rmtree(ft_output_dir)


def finetune_checkpoint(ckpt_file, gpu, app_args, loaders):
    # Usually when we train, we also want to look at and graph, the validation score of each epoch.
    # When we run many fine-tuning sessions at once, we don't care to look at the validation score.
    # However, we want to perform a sort-of "early-stopping" in which we use the checkpoint of the 
    # best performing training epoch, and not the checkpoint created by the last epoch.
    # We evaluate what is the "best" checkpoint by looking at the validation accuracy 
    VALIDATE_ENABLE_FACTOR = 0.8
    name = os.path.basename(ckpt_file)
    print("Fine-tuning checkpoint %s" % name)
 
    app_args.gpus = str(gpu)
    app_args.name = name
    app_args.deprecated_resume = ckpt_file
    app = ClassifierCompressor(app_args)
    app.train_loader, app.val_loader, app.test_loader = loaders
    best = [float("-inf"), float("-inf"), float("inf")]
    for epoch in range(app_args.epochs):
        validate = epoch >= math.floor(VALIDATE_ENABLE_FACTOR * app_args.epochs)
        top1, top5, loss = app.train_validate_with_scheduling(epoch, validate=validate, verbose=False)
        #app.train_one_epoch(e, verbose=False)
        if validate:
            if top1 > best[0]:
                best = [top1, top5, loss]
    #return (name, app.test())
    return (name, best)


if __name__ == '__main__':
    def get_immediate_subdirs(a_dir):
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
                if os.path.isdir(os.path.join(a_dir, name)) and name != "ft"]

    def add_parallel_args(argparser):
        group = argparser.add_argument_group('parallel fine-tuning')
        group.add_argument('--processes', type=int, default=4,
                           help="Number of parallel experiment processes to run in parallel")
        group.add_argument('--scan-dir', metavar='DIR', required=True, help='path to checkpoints')
        group.add_argument('--output-csv', metavar='DIR', required=True, 
                           help='name of the CSV file containing the output')

    try:
        set_start_method('forkserver')
    except RuntimeError:
        pass
    # Parse arguments
    argparser = parser.get_parser(init_classifier_compression_arg_parser())
    add_parallel_args(argparser)
    app_args = argparser.parse_args()

    print("Starting fine-tuning")
    stats_file = FTStatsLogger(os.path.join(app_args.scan_dir, app_args.output_csv))
    ft_dirs = get_immediate_subdirs(app_args.scan_dir)
    for ft_dir in ft_dirs:
        finetune_directory(ft_dir, stats_file, app_args)

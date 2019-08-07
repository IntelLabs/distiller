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

"""A simple script to run several instances of the same experiment configuration, in parallel.

The script will launch multiple processes (equal to the number of GPUs on the machine).
Each process from execute one instance of the experiment command.

Format:
    $ python multi-run.py <experiment output directory> <experiment command-line>

    Each experiment uses a different (single) GPU instance, uses a different seed and is assigned an ID in the
    range [0..number-of-gpus].
    All experiments log directories are nested under the main output directory: <experiment output directory>

    <experiment command-line> conforms to the command-line interface of compress_classifier.py when
    compress_classifier.py is the application you want to execute.  You can also execute other applications that
    derive from distiller.apputils.ClassifierCompressor.


Example:
    $ python multi-run.py experiments/plain20-random-l1_rank compress_classifier.py --arch=plain20_cifar ${CIFAR10_PATH} --resume=checkpoint.plain20_cifar.pth.tar --lr=0.05 --amc --amc-protocol=mac-constrained --amc-action-range 0.05 1.0 --amc-target-density=0.5 -p=50 --etes=0.075 --amc-ft-epochs=0 --amc-prune-pattern=channels --amc-prune-method=l1-rank --amc-agent-algo=Random-policy --amc-cfg=../automated_deep_compression/auto_compression_channels.yaml --evs=0.5 --etrs=0.5 --amc-rllib=random -j=1

"""

import time
import os                                                                       
import sys
import torch
import argparse
from multiprocessing import Pool

def run_experiment(args):
    experiment_outdir, id, seed, gpu = args
    experiment_cmd = " ".join(sys.argv[2:])
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1000,
                        help='seed the PRNG for CPU, CUDA, numpy, and Python')
    #experiment_cmd += + ' > /dev/null'
    multi_args, unknown = parser.parse_known_args()
    os.system("python3 {} --name={} --gpus={} --seed={} --out-dir={}".format(experiment_cmd, id, gpu,
                                                                             multi_args.seed+id, experiment_outdir))

# Create the directory that will store the outputs of all the executions.
timestr = time.strftime("%Y.%m.%d-%H%M%S")
experiment_outdir = os.path.join(sys.argv[1], timestr)
os.makedirs(experiment_outdir, exist_ok=False)

n_gpus = torch.cuda.device_count()
pool = Pool(processes=n_gpus)                                             
pool.map_async(run_experiment, [(experiment_outdir, id, 1000+id, id%n_gpus) for id in range(n_gpus)])
pool.close()
pool.join()
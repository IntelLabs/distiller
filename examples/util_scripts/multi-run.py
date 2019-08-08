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

"""A script to run several instances of the same experiment configuration.

Format:
    $ python multi-run.py <experiment output directory> <experiment command-line>

Example:
    $ python multi-run.py experiments/plain20-random-l1_rank compress_classifier.py --arch=plain20_cifar ${CIFAR10_PATH} --resume=checkpoint.plain20_cifar.pth.tar --lr=0.05 --amc --amc-protocol=mac-constrained --amc-action-range 0.05 1.0 --amc-target-density=0.5 -p=50 --etes=0.075 --amc-ft-epochs=0 --amc-prune-pattern=channels --amc-prune-method=l1-rank --amc-agent-algo=Random-policy --amc-cfg=../automated_deep_compression/auto_compression_channels.yaml --evs=0.5 --etrs=0.5 --amc-rllib=random -j=1

"""

import time
import os                                                                       
import sys
import torch
from multiprocessing import Pool

def run_experiment(args):
    experiment_outdir, id, seed, gpu = args
    experiment_cmd = " ".join(sys.argv[2:])
    #experiment_cmd += + ' > /dev/null'
    os.system("python3 {} --name={} --gpus={} --seed={} --out-dir={}".format(experiment_cmd, id, 
                                                                             gpu, seed, experiment_outdir))

# Create the directory that will store the outputs of all the executions.
timestr = time.strftime("%Y.%m.%d-%H%M%S")
experiment_outdir = os.path.join(sys.argv[1], timestr)
os.makedirs(experiment_outdir, exist_ok=False)

n_gpus = torch.cuda.device_count()
pool = Pool(processes=n_gpus)                                             
pool.map_async(run_experiment, [(experiment_outdir, id, 1000+id, id%n_gpus) for id in range(n_gpus)])
pool.close()
pool.join()
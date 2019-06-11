#
# Copyright (c) 2018 Intel Corporation
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

"""To execute this code:

$ time python3 compress_classifier.py --arch=plain20_cifar ../../../data.cifar10 --resume=checkpoint.plain20_cifar.pth.tar --lr=0.05 --amc --amc-protocol=mac-constrained --amc-target-density=0.5 -p=50

"""
import math
import os
import copy
import logging
import numpy as np
import torch
import csv
try:
    import gym
except ImportError as e:
    print("WARNING: to use automated compression you will need to install extra packages")
    print("See instructions in the interface of each RL library.")
    raise e
from gym import spaces
import distiller
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from distiller import normalize_module_name, SummaryGraph
from examples.automated_deep_compression.environment import DistillerWrapperEnvironment, Observation
from examples.automated_deep_compression.utils.features_collector import collect_intermediate_featuremap_samples


msglogger = logging.getLogger()

    
def train_auto_compressor(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn):
    dataset = args.dataset
    arch = args.arch
    num_ft_epochs = args.amc_ft_epochs
    action_range = args.amc_action_range
    np.random.seed()

    # Read the experiment configuration
    amc_cfg_fname = args.amc_cfg_file
    if not amc_cfg_fname:
        raise ValueError("You must specify a valid configuration file path using --amc-cfg")

    with open(amc_cfg_fname, 'r') as cfg_file:
        compression_cfg = distiller.utils.yaml_ordered_load(cfg_file)

    if not args.amc_rllib:
        raise ValueError("You must set --amc-rllib to a valid value")

    #rl_lib = compression_cfg["rl_lib"]["name"]
    #msglogger.info("Executing AMC: RL agent - %s   RL library - %s", args.amc_agent_algo, rl_lib)

    # Create a dictionary of parameters that Coach will handover to DistillerWrapperEnvironment
    # Once it creates it.
    services = distiller.utils.MutableNamedTuple({
            'validate_fn': validate_fn,
            'save_checkpoint_fn': save_checkpoint_fn,
            'train_fn': train_fn})

    app_args = distiller.utils.MutableNamedTuple({
            'dataset': dataset,
            'arch': arch,
            'optimizer_data': optimizer_data})

    amc_cfg = distiller.utils.MutableNamedTuple({
            'modules_dict': compression_cfg["network"],  # dict of modules, indexed by arch name
            'protocol': args.amc_protocol,
            'agent_algo': args.amc_agent_algo,
            'num_ft_epochs': num_ft_epochs,
            'action_range': action_range,
            'reward_frequency': args.amc_reward_frequency,
            'ft_frequency': args.amc_ft_frequency,
            'pruning_pattern':  args.amc_prune_pattern,
            'pruning_method': args.amc_prune_method,
            'group_size': 1,
            'n_points_per_fm':10})

    #net_wrapper = NetworkWrapper(model, app_args, services)
    #return sample_networks(net_wrapper, services)

    from .rewards import reward_factory
    amc_cfg.target_density = args.amc_target_density
    amc_cfg.reward_fn, amc_cfg.action_constrain_fn = reward_factory(args.amc_protocol)

    if args.amc_rllib == "spinningup":
        amc_cfg.heatup_noise = 0.5
        amc_cfg.initial_training_noise = 0.5
        amc_cfg.training_noise_decay = 0.996  # 0.998
        amc_cfg.num_heatup_epochs = args.amc_heatup_epochs
        amc_cfg.num_training_epochs = args.amc_training_epochs

        from .rl_libs.spinningup import spinningup_if
        x = spinningup_if.RlLibInterface()
        env1 = DistillerWrapperEnvironment(model, app_args, amc_cfg, services)
        env2 = DistillerWrapperEnvironment(model, app_args, amc_cfg, services)
        steps_per_episode = env1.steps_per_episode
        x.solve(env1, env2, steps_per_episode)
    elif args.amc_rllib == "private":
        env = DistillerWrapperEnvironment(model, app_args, amc_cfg, services)
        from .rl_libs.private import private_if
        x = private_if.RlLibInterface()
        args.observation_len = len(Observation._fields)
        x.solve(env, args)
    elif args.amc_rllib == "coach":
        from .rl_libs.coach import coach_if
        x = coach_if.RlLibInterface()
        env_cfg  = {'model': model, 
                    'app_args': app_args,
                    'amc_cfg': amc_cfg,
                    'services': services}
        env = DistillerWrapperEnvironment(model, app_args, amc_cfg, services)
        steps_per_episode = env.steps_per_episode
        x.solve(**env_cfg, args=args, steps_per_episode=steps_per_episode)
    elif args.amc_rllib == "random":
        from .rl_libs.random import random_if
        x = random_if.RlLibInterface()
        env = DistillerWrapperEnvironment(model, app_args, amc_cfg, services)
        return x.solve(env)
    else:
        raise ValueError("unsupported rl library: ", rl_lib)


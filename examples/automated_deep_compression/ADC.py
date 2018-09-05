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
import random
import math
import copy
import logging
import numpy as np
import torch
import json
import gym
from gym import spaces
import distiller
from apputils import SummaryGraph
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from distiller import normalize_module_name

from rl_coach.base_parameters import TaskParameters

# When we import the graph_manager from the ADC_DDPG preset, we implicitly instruct
# Coach to create and use our CNNEnvironment environment.
# So Distiller calls Coach, which creates the environment, trains the agent, and ends.
from examples.automated_deep_compression.presets.ADC_DDPG import graph_manager, agent_params
# Coach imports
from rl_coach.schedules import ConstantSchedule, PieceWiseSchedule, ExponentialSchedule
from rl_coach.core_types import EnvironmentSteps


msglogger = logging.getLogger()
Observation = namedtuple('Observation', ['t', 'n', 'c', 'h', 'w', 'stride', 'k', 'MACs', 'reduced', 'rest', 'prev_a'])

ALMOST_ONE = 0.9999
USE_COACH = True
PERFORM_THINNING = True

#reward = -1 * (1-top1/100) * math.log(total_macs/self.dense_model_macs)
#
#reward = -1 * (1-top1/100) + math.log(total_macs/self.dense_model_macs)
#reward = 4*top1/100 - math.log(total_macs)
#reward = reward * total_macs/213201664
#reward = reward - 5 * total_macs/213201664
#reward = -1 * vloss * math.sqrt(math.log(total_macs))
#reward = top1 / math.log(total_macs)
#alpha = 0.9
#reward = -1 * ( (1-alpha)*(top1/100) + 10*alpha*(total_macs/self.dense_model_macs) )

#alpha = 0.99
#reward = -1 * ( (1-alpha)*(top1/100) + alpha*(total_macs/self.dense_model_macs) )

#reward = vloss * math.log(total_macs)
#reward = -1 * vloss * (total_macs / self.dense_model_macs)
#reward = top1 * (self.dense_model_macs / total_macs)
#reward = -1 * math.log(total_macs)
#reward =  -1 * vloss


def do_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn):
    np.random.seed()

    if USE_COACH:
        return coach_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn)
    return random_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn)


def coach_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn):
    task_parameters = TaskParameters(framework_type="tensorflow",
                                     experiment_path="./experiments/test")
    extra_params = {'save_checkpoint_secs': None,
                    'render': True}
    task_parameters.__dict__.update(extra_params)

    # Create a dictionary of parameters that Coach will handover to CNNEnvironment
    # Once it creates it.
    if True:
        exploration_noise = 0.5
        #exploration_noise = 0.25
        exploitation_decay = 0.996
        graph_manager.env_params.additional_simulator_parameters = {
            'model': model,
            'dataset': dataset,
            'arch': arch,
            'data_loader': data_loader,
            'validate_fn': validate_fn,
            'save_checkpoint_fn': save_checkpoint_fn,
            #'action_range': (0.10, 0.95),
            'action_range': (0.70, 0.95),
            'onehot_encoding': False,
            'normalize_obs': True,
            'desired_reduction': None,
            'reward_fn': lambda top1, top5, vloss, total_macs: -1 * (1-top5/100) * math.log(total_macs)
            #'reward_fn': lambda top1, total_macs: -1 * (1-top1/100) * math.log(total_macs)
            #'reward_fn': lambda top1, total_macs: -1 * max(1-top1/100, 0.25) * math.log(total_macs)
            #'reward_fn': lambda top1, total_macs: -1 * (1-top1/100) * math.log(total_macs/100000)
            #'reward_fn': lambda top1, total_macs:  top1/100 * total_macs/self.dense_model_macs
        }
    else:
        exploration_noise = 0.5
        #exploration_noise = 0.25
        exploitation_decay = 0.996
        graph_manager.env_params.additional_simulator_parameters = {
            'model': model,
            'dataset': dataset,
            'arch': arch,
            'data_loader': data_loader,
            'validate_fn': validate_fn,
            'save_checkpoint_fn': save_checkpoint_fn,
            'action_range': (0.10, 0.95),
            'onehot_encoding': False,
            'normalize_obs': True,
            'desired_reduction': 1.5e8,
            'reward_fn': lambda top1, total_macs: top1/100
            #'reward_fn': lambda top1, total_macs: min(top1/100, 0.75)
        }

    #msglogger.debug('Experiment configuarion:\n' + json.dumps(graph_manager.env_params.additional_simulator_parameters, indent=2))
    steps_per_episode = 13
    agent_params.exploration.noise_percentage_schedule = PieceWiseSchedule([(ConstantSchedule(exploration_noise),
                                                                             EnvironmentSteps(100*steps_per_episode)),
                                                                            (ExponentialSchedule(exploration_noise, 0, exploitation_decay),
                                                                             EnvironmentSteps(300*steps_per_episode))])
    graph_manager.create_graph(task_parameters)
    graph_manager.improve()


class CNNEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn,
                 action_range, onehot_encoding, normalize_obs, desired_reduction,
                 reward_fn):
        self.pylogger = distiller.data_loggers.PythonLogger(msglogger)
        self.tflogger = distiller.data_loggers.TensorBoardLogger(msglogger.logdir)

        self.dataset = dataset
        self.arch = arch
        self.data_loader = data_loader
        self.validate_fn = validate_fn
        self.save_checkpoint_fn = save_checkpoint_fn
        self.orig_model = model
        self.onehot_encoding = onehot_encoding
        self.normalize_obs = normalize_obs
        self.max_reward = -1000
        self.reward_fn = reward_fn

        self.conv_layers, self.dense_model_macs, self.dense_model_size = collect_conv_details(model, dataset)
        self.reset(init_only=True)
        msglogger.info("Model %s has %d Convolution layers", arch, len(self.conv_layers))
        msglogger.info("\tTotal MACs: %s" % distiller.pretty_int(self.dense_model_macs))
        msglogger.info("Configuration:\n\tonehot_encoding={}\n\tnormalize_obs={}".format(self.onehot_encoding,
                                                                                         self.normalize_obs))
        self.debug_stats = {'episode': 0}
        self.action_low = action_range[0]
        self.action_high = action_range[1]
        # Gym
        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Box(self.action_low, self.action_high, shape=(1,))
        self.action_space.default_action = self.action_low

        self.desired_reduction = desired_reduction
        self.STATE_EMBEDDING_LEN = len(Observation._fields)
        if self.onehot_encoding:
            self.STATE_EMBEDDING_LEN += 12
        self.observation_space = spaces.Box(0, float("inf"), shape=(self.STATE_EMBEDDING_LEN,))

    def reset(self, init_only=False):
        """Reset the environment.
        This is invoked by the Agent.
        """
        msglogger.info("Resetting the environment (init_only={})".format(init_only))
        self.current_layer_id = -1
        self.prev_action = 0
        self.model = copy.deepcopy(self.orig_model)
        self.zeros_mask_dict = distiller.create_model_masks_dict(self.model)
        self._remaining_macs = self.dense_model_macs
        self._removed_macs = 0
        if init_only:
            return
        obs, _, _, _, = self.step(0)
        return obs

    def num_layers(self):
        return len(self.conv_layers)

    def current_layer(self):
        return self.get_layer(self.current_layer_id)

    def get_layer(self, idx):
        try:
            return self.conv_layers[idx]
        except KeyError:
            return None
    def episode_is_done(self):
        return self.current_layer_id == self.num_layers()

    def remaining_macs(self):
        """Return the amount of MACs remaining in the model's unprocessed
        Convolution layers.
        This is normalized to the range 0..1
        """
        #return 1 - self.sum_list_macs(self.unprocessed_layers) / self.dense_model_macs
        return self._remaining_macs / self.dense_model_macs

    def removed_macs(self):
        """Return the amount of MACs removed so far.
        This is normalized to the range 0..1
        """
        #return self.sum_list_macs(self.processed_layers) / self.dense_model_macs
        return self._removed_macs / self.dense_model_macs

    # def sum_list_macs(self, conv_list):
    #     """Sum the MACs in the provided list of Convolution layers"""
    #     total_macs = 0
    #     for conv in conv_list:
    #         total_macs += conv.macs
    #     return total_macs

    def render(self, mode, close):
        """Provide some feedback to the user about what's going on
        This is invoked by the Agent.
        """
        if self.current_layer_id == 0:
            msglogger.info("+" + "-" * 50 + "+")
            msglogger.info("Starting a new episode")
            msglogger.info("+" + "-" * 50 + "+")

        msglogger.info("Environment: current_layer_id=%d" % self.current_layer_id)
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

    def get_action(self, a):
        reduced = self._removed_macs
        rest = self._remaining_macs

        #duty = self.desired_reduction - (1.2*reduced + rest)
        duty = self.desired_reduction - (reduced + rest)
        flops = self.get_macs(self.current_layer())
        msglogger.info("action ********** a={}  duty={} desired_reduction={} reduced={}  rest={}  flops={}".
                       format(a, duty, self.desired_reduction, reduced, rest, flops))

        if duty > 0:
            #duty = 0.9*desired_reduction - (reduced + rest)
            duty = self.desired_reduction - (reduced + rest)
            msglogger.info("action ********** duty/flops={}".format(duty / flops))
            msglogger.info("action ********** 1 - duty/flops={}".format(1 - duty / flops))
            #a = max(1-self.action_low, min(a, 1 - duty/flops))

            ##
            ##  Consider using max=0 for R = error * macs
            ##           using max= self.action_low for FLOP-limited?  Add noise so it doesn't get stuck in one place?
            ##
            #a = max(self.action_low, min(a, 1 - duty/flops))
            a = max(0.05, min(a, 1 - duty/flops))
        return a

    def save_checkpoint(self, is_best=False):
        # Save the learned-model checkpoint
        scheduler = distiller.CompressionScheduler(self.model)
        masks = {param_name: masker.mask for param_name, masker in self.zeros_mask_dict.items()}
        scheduler.load_state_dict(state={'masks_dict': masks})
        episode = self.debug_stats['episode']
        episode = str(episode).zfill(3)
        if is_best:
            name = "BEST_adc_episode_{}".format(episode)
        else:
            name = "adc_episode_{}".format(episode)

        self.save_checkpoint_fn(epoch=self.debug_stats['episode'], model=self.model, scheduler=scheduler, name=name)

    def step(self, action):
        """Take a step, given an action.

        The action represents the desired sparsity.
        This function is invoked by the Agent.
        """
        msglogger.info("env.step - current_layer_id={} action={}".format(self.current_layer_id, action))
        assert action == 0 or (action >= self.action_low-0.001 and action <= self.action_high+0.001)
        if self.desired_reduction is not None:
            action = self.get_action(action)
        msglogger.info("action ********** (leave) {}".format(action))
        action = 1 - action
        layer_macs = self.get_macs(self.current_layer())
        if action > 0 and self.current_layer_id > -1:
            actual_action = self.__remove_channels(self.current_layer_id, action, prune_what="filters")
        else:
            actual_action = 0
        #msglogger.info("-------****---------{}".format(actual_action))
        layer_macs_after_action = self.get_macs(self.current_layer())

        # Update the various counters after taking the step
        self.current_layer_id += 1
        next_layer_macs = self.get_macs(self.current_layer())
        self._removed_macs += (layer_macs - layer_macs_after_action)
        self._remaining_macs -= next_layer_macs
        #self.prev_action = 1 - actual_action

        stats = ('Peformance/Validation/',
                 OrderedDict([('requested_action', action),
                              ('actual_action', 1-actual_action)]))
        distiller.log_training_progress(stats, None, self.debug_stats['episode'], steps_completed=self.current_layer_id,
                                        total_steps=13,
                                        log_freq=1, loggers=[self.tflogger])

        if self.episode_is_done():
            observation = self.get_final_obs()
            reward, top1 = self.compute_reward()
            self.debug_stats['episode'] += 1
            if reward > self.max_reward:
                self.max_reward = reward
                self.save_checkpoint(is_best=True)
                msglogger.info("Best reward={}  episode={}  top1={}".format(reward, self.debug_stats['episode'], top1))
            self.save_checkpoint(is_best=False)
        else:
            observation = self._get_obs(next_layer_macs)
            if True:
                reward = 0
            else:
                reward, _ = self.compute_reward()

        self.prev_action = 1 - action
        msglogger.info("###################### self.prev_action={}".format(self.prev_action))
        info = {}
        return observation, reward, self.episode_is_done(), info

    def _get_obs4(self, macs, current_layer, conv_module):
        """Produce a state embedding (i.e. an observation)"""

        if self.normalize_obs:
            obs = np.array([current_layer.t,
                            conv_module.out_channels / 512,
                            conv_module.in_channels / 512,
                            current_layer.ifm_h / 32,
                            current_layer.ifm_w / 32,
                            current_layer.stride[0] / 2,
                            current_layer.k / 3,
                            macs / self.dense_model_macs,
                            self.removed_macs(), self.remaining_macs(), self.prev_action])
        else:
            obs = np.array([current_layer.t,
                            conv_module.out_channels, conv_module.in_channels,
                            current_layer.ifm_h, current_layer.ifm_w, current_layer.stride[0], current_layer.k,
                            macs/self.dense_model_macs,
                            self.removed_macs(), self.remaining_macs(), self.prev_action])

        if self.onehot_encoding:
            id = np.zeros(13)
            id[current_layer.t] = 1
            obs = np.concatenate([id, obs[1:]])
            msglogger.info("obs={}".format(obs))
        else:
            msglogger.info("obs={}".format(Observation._make(obs)))

        assert len(obs) == self.STATE_EMBEDDING_LEN
        assert (macs / self.dense_model_macs + self.removed_macs() + self.remaining_macs()) <= 1
        return obs

    def _get_obs(self, macs):
        #return self._get_obs3(macs)
        current_layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, current_layer.name)
        return self._get_obs4(macs, current_layer, conv_module)

    def get_final_obs(self):
        """Return the final stae embedding (observation)
        The final state is reached after we traverse all of the Convolution layers.
        """
        obs = np.array([-1, 0, 0,
                         0, 0, 0, 0,
                         0, self.removed_macs(), 0, 1 - self.prev_action])

        if self.onehot_encoding:
            id = np.zeros(13)
            obs = np.concatenate([id, obs[1:]])

        assert len(obs) == self.STATE_EMBEDDING_LEN
        return obs

    def get_macs(self, layer):
        """Return the number of MACs required to compute <layer>'s Convolution"""
        if layer is None:
            return 0

        conv_module = distiller.model_find_module(self.model, layer.name)
        # MACs = volume(OFM) * (#IFM * K^2)
        dense_macs = (conv_module.out_channels * layer.ofm_h * layer.ofm_w) * (conv_module.in_channels * layer.k**2)
        if PERFORM_THINNING:
            return dense_macs

        # If we didn't physically remove structures, we need to use the structural sparsity to compute MACs
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)
        return dense_macs * distiller.density_ch(conv_p)

    def __remove_channels(self, idx, fraction_to_prune, prune_what="channels"):
        """Physically remove channels and corresponding filters from the model"""
        if idx not in range(self.num_layers()):
            raise ValueError("idx=%d is not in correct range (0-%d)" % (idx, self.num_layers()))
        if fraction_to_prune < 0:
            raise ValueError("fraction_to_prune=%f is illegal" % (fraction_to_prune))

        if fraction_to_prune == 0:
            return 0
        if fraction_to_prune == 1.0:
            # For now, prevent the removal of entire layers
            fraction_to_prune = ALMOST_ONE

        layer = self.conv_layers[idx]
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)

        msglogger.info("ADC: removing %.1f%% %s from %s" % (fraction_to_prune*100, prune_what, conv_pname))

        if prune_what == "channels":
            calculate_sparsity = distiller.sparsity_ch
            reg_regims = {conv_pname: [fraction_to_prune, "Channels"]}
            remove_structures = distiller.remove_channels
        elif prune_what == "filters":
            calculate_sparsity = distiller.sparsity_3D
            reg_regims = {conv_pname: [fraction_to_prune, "3D"]}
            remove_structures = distiller.remove_filters
        else:
            raise ValueError("unsupported structure {}".format(prune_what))
        # Create a channel-ranking pruner
        pruner = distiller.pruning.L1RankedStructureParameterPruner("adc_pruner", reg_regims)
        pruner.set_param_mask(conv_p, conv_pname, self.zeros_mask_dict, meta=None)

        if (self.zeros_mask_dict[conv_pname].mask is None or
            calculate_sparsity(self.zeros_mask_dict[conv_pname].mask) == 0):
            msglogger.info("__remove_channels: aborting because there are no channels to prune")
            return 0

        # Use the mask to prune
        self.zeros_mask_dict[conv_pname].apply_mask(conv_p)

        if PERFORM_THINNING:
            remove_structures(self.model, self.zeros_mask_dict, self.arch, self.dataset, optimizer=None)
            conv_p = distiller.model_find_param(self.model, conv_pname)
            return distiller.volume(conv_p) / layer.weights_vol
        actual_sparsity = calculate_sparsity(conv_p)
        return actual_sparsity

    def compute_reward(self):
        """The ADC paper defines reward = -Error"""
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])
        compression = distiller.model_numel(self.model, param_dims=[4]) / self.dense_model_size
        _, total_macs, total_nnz = collect_conv_details(self.model, self.dataset)
        msglogger.info("Total parameters left: %.2f%%" % (compression*100))
        msglogger.info("Total compute left: %.2f%%" % (total_macs/self.dense_model_macs*100))

        top1, top5, vloss = self.validate_fn(model=self.model, epoch=self.debug_stats['episode'])
        reward = self.reward_fn(top1, top5, vloss, total_macs)

        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5),
                              ('reward', reward),
                              ('total_macs', int(total_macs)),
                              ('log(total_macs)', math.log(total_macs)),
                              #('log(total_macs/self.dense_model_macs)', math.log(total_macs/self.dense_model_macs)),
                              ('total_nnz', int(total_nnz))]))
        distiller.log_training_progress(stats, None, self.debug_stats['episode'], steps_completed=0, total_steps=1,
                                        log_freq=1, loggers=[self.tflogger, self.pylogger])
        return reward, top1


def get_dummy_input(dataset):
    if dataset == 'imagenet':
        dummy_input = torch.randn(1, 3, 224, 224)
    elif dataset == 'cifar10':
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        raise ValueError("dataset %s is not supported" % dataset)
    return dummy_input


def collect_conv_details(model, dataset):
    dummy_input = get_dummy_input(dataset)
    g = SummaryGraph(model.cuda(), dummy_input.cuda())
    conv_layers = OrderedDict()
    total_macs = 0
    total_nnz = 0
    for id, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, torch.nn.Conv2d):
            conv = SimpleNamespace()
            conv.t = len(conv_layers)
            conv.k = m.kernel_size[0]
            conv.stride = m.stride

            # Use the SummaryGraph to obtain some other details of the models
            conv_op = g.find_op(normalize_module_name(name))
            assert conv_op is not None

            conv.weights_vol = conv_op['attrs']['weights_vol']
            total_nnz += conv.weights_vol
            conv.macs = conv_op['attrs']['MACs']
            conv_pname = name + ".weight"
            conv_p = distiller.model_find_param(model, conv_pname)
            conv.macs *= distiller.density_ch(conv_p)
            total_macs += conv.macs

            conv.ofm_h = g.param_shape(conv_op['outputs'][0])[2]
            conv.ofm_w = g.param_shape(conv_op['outputs'][0])[3]
            conv.ifm_h = g.param_shape(conv_op['inputs'][0])[2]
            conv.ifm_w = g.param_shape(conv_op['inputs'][0])[3]

            conv.name = name
            conv.id = id
            conv_layers[len(conv_layers)] = conv
    return conv_layers, total_macs, total_nnz


from examples.automated_deep_compression.adc_controlled_envs import *
def random_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn):
    """Random ADC agent"""
    action_range = (0.0, 1.0)
    env = CNNEnvironment(model, dataset, arch, data_loader,
                         validate_fn, save_checkpoint_fn, action_range,
                         onehot_encoding=False, normalize_obs=False, desired_reduction=None,
                         reward_fn=lambda top1, total_macs: top1/100)

    best_episode = [-1000, None]
    update_rate = 5
    env.action_space = RandomADCActionSpace(action_range[0], action_range[1], std=0.35)
    for ep in range(1000):
        observation = env.reset()
        action_config = []
        for t in range(100):
            #env.render(0, 0)
            msglogger.info("[episode={}:{}] observation = {}".format(ep, t, observation))
            # take a random action
            action = env.action_space.sample()
            action_config.append(action)
            observation, reward, done, info = env.step(action)
            if done:
                msglogger.info("Episode finished after {} timesteps".format(t+1))
                msglogger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                msglogger.info("New solution found: episode={} reward={} config={}".format(ep, reward, action_config))
                break
        if reward > best_episode[0]:
            best_episode[0] = reward
            best_episode[1] = action_config
        if ep % update_rate == 0:
            env.action_space.set_cfg(means=best_episode[1], std=0.4)
            best_episode = [-1000, None]


import os
import pandas as pd
from tabulate import tabulate
import apputils
from models import create_model


def summarize_experiment(experiment_dir, dataset, arch, validate_fn):
    df = pd.DataFrame(columns=['File', 'NNZ', 'MACs', 'Top1'])
    for file in os.listdir(experiment_dir):
        if file.endswith(".pth.tar"):
            cnt_macs, cnt_params, top1 = get_experiment_performance_summary(os.path.join(experiment_dir, file), dataset, arch, validate_fn)
            df.loc[len(df.index)] = [file, cnt_params, cnt_macs, top1]
    t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f")
    print(t)
    csv_fname = os.path.join(experiment_dir, "arch_space" + ".csv")
    print("Saving results to: {}".format(csv_fname))
    df.to_csv(csv_fname, sep=',')


def get_experiment_performance_summary(chkpt_fname, dataset, arch, validate_fn):
    model = create_model(False, dataset, arch)
    model, compression_scheduler, start_epoch = apputils.load_checkpoint(model, chkpt_fname)

    dummy_input = get_dummy_input(dataset)
    perf_df = distiller.model_performance_summary(model, dummy_input, 1)
    total_macs = perf_df['MACs'].sum()
    top1, top5, vloss = validate_fn(model=model, epoch=-1)
    return total_macs, distiller.model_numel(model), top1

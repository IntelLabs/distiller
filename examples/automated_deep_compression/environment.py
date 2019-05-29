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
    print("See instructions in the header of examples/automated_deep_compression/ADC.py")
    raise e
from gym import spaces
import distiller
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from distiller import normalize_module_name, SummaryGraph
#from examples.automated_deep_compression.adc_random_env import random_agent
 
from .utils.features_collector import collect_intermediate_featuremap_samples

msglogger = logging.getLogger()


# Observation = namedtuple('Observation', ['t', 'n', 'c', 'h', 'w', 'stride', 'k', 'MACs', 'reduced', 'rest', 'prev_a'])
# LayerDesc = namedtuple('LayerDesc', ['t', 'n', 'c', 'h', 'w', 'stride', 'k', 'MACs', 'reduced', 'rest', 'prev_a'])
Observation = namedtuple('Observation', ['t', 'n', 'c',  'stride', 'k', 'MACs', 'Weights', 'reduced', 'rest', 'prev_a'])
LayerDesc = namedtuple('LayerDesc', ['t', 'n', 'c', 'stride', 'k', 'MACs', 'Weights', 'reduced', 'rest', 'prev_a'])

LayerDescLen = len(LayerDesc._fields)
ALMOST_ONE = 0.9999


class CSVFile(object):
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


class AMCStatsFile(CSVFile):
    def __init__(self, fname):
        headers = ['episode', 'top1', 'reward', 'total_macs', 'normalized_macs',
                   'normalized_nnz', 'ckpt_name', 'action_history', 'agent_action_history']
        super().__init__(fname, headers)


class FineTuneStatsFile(CSVFile):
    def __init__(self, fname):
        headers = ['episode', 'ft_top1_list']
        super().__init__(fname, headers)


def is_using_continuous_action_space(agent):
    return agent in ("DDPG", "ClippedPPO-continuous", "Random-policy")


def log_amc_config(amc_cfg):
    try:
        msglogger.info('AMC configuration:')
        for k, v in amc_cfg.items():
            msglogger.info("\t{} : {}".format(k, v))
    except TypeError as e:
        pass


# Work around PPO's Gaussian sampling space
# See: https://github.com/openai/baselines/issues/121
PPO_MIN, PPO_MAX = -3, 3  # These are empiric values that are a very rough estimate
def adjust_ppo_output(ppo_pruning_action, action_high, action_low):
    # We need to map PPO's infinite action-space to our action-space (in PPO actions are sampled 
    # from a normalized Gaussian N(0,1)).
    # We do this by a shift & scale operation: 
    #   1. We crudely assume that most of the Gaussian maps to actions in a limited range {PPO_MIN..PPO_MAX} 
    #      which allows us to clip the action, and scale it to the range of the actual action-space.
    #   2. We assume that the distribution the agent samples from is centered at zero, and so we shift
    #      actions to the center of the actual action-space.

    #ppo_pruning_action = np.clip(ppo_pruning_action, self.action_low, self.action_high) 
    shift = (action_high - action_low) / 2 + action_low
    scale = (action_high - action_low) / (PPO_MAX - PPO_MIN)
    # scale = (action_high - action_low) / 2# / (PPO_MAX - PPO_MIN)
    pruning_action = ppo_pruning_action * scale + shift
    return float(pruning_action)



class NetworkWrapper(object):
    def __init__(self, model, app_args, services, modules_list):
        self.app_args = app_args
        self.services = services
        self.modules_list = modules_list
        self.conv_layers, _, _ = self.collect_conv_details(model)
        self.reset(model)

    def get_model_resources_requirements(self, model=None):
        if model is None:
            model = self.model
        _, total_macs, total_nnz = collect_conv_details(model, self.app_args.dataset, True)
        return total_macs, total_nnz

    @property
    def arch(self):
        return self.app_args.arch

    def collect_conv_details(self, model):
        return collect_conv_details(model, self.app_args.dataset, True, self.modules_list)

    def num_layers(self):
        return len(self.conv_layers)

    def get_layer(self, idx):
        try:
            return self.conv_layers[idx]
        except KeyError:
            return None

    def get_layer_macs(self, layer):
        """Return the number of MACs required to compute <layer>'s Convolution"""
        if layer is None:
            return 0
        conv_module = distiller.model_find_module(self.model, layer.name)
        # MACs = volume(OFM) * (#IFM * K^2) / groups
        dense_macs = (conv_module.out_channels * layer.ofm_h * layer.ofm_w) * (conv_module.in_channels * layer.k**2) 
        dense_macs /= conv_module.groups 
        return dense_macs

    def reset(self, model):
        self.model = model
        self.zeros_mask_dict = distiller.create_model_masks_dict(self.model)

    def create_scheduler(self):
        scheduler = distiller.CompressionScheduler(self.model)
        masks = {param_name: masker.mask for param_name, masker in self.zeros_mask_dict.items()}
        scheduler.load_state_dict(state={'masks_dict': masks})
        return scheduler

    def remove_structures(self, layer_id, fraction_to_prune, prune_what="channels", prune_how="l1-rank"):
        """Physically remove channels and corresponding filters from the model

        Returns the compute-sparsity of the layer with index 'layer_id'
        """
        if layer_id not in range(self.num_layers()):
            raise ValueError("idx=%d is not in correct range (0-%d)" % (layer_id, self.num_layers()))
        if fraction_to_prune < 0:
            raise ValueError("fraction_to_prune=%f is illegal" % (fraction_to_prune))

        if fraction_to_prune == 0:
            return 0
        if fraction_to_prune == 1.0:
            # For now, prevent the removal of entire layers
            fraction_to_prune = ALMOST_ONE

        layer = self.conv_layers[layer_id]
        macs_before = self.get_layer_macs(layer)
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)

        msglogger.info("ADC: trying to remove %.1f%% %s from %s" % (fraction_to_prune*100, prune_what, conv_pname))

        if prune_what == "channels":
            calculate_sparsity = distiller.sparsity_ch
            remove_structures_fn = distiller.remove_channels
            group_type = "Channels"
        elif prune_what == "filters":
            calculate_sparsity = distiller.sparsity_3D
            group_type = "Filters"
            remove_structures_fn = distiller.remove_filters
        else:
            raise ValueError("unsupported structure {}".format(prune_what))
        if prune_how == "l1-rank" or prune_how == "stochastic-l1-rank":
            # Create a channel/filter-ranking pruner
            epsilon = 0 if "l1-rank" else 0.1
            pruner = distiller.pruning.L1RankedStructureParameterPruner("adc_pruner", group_type,
                                                                        fraction_to_prune, conv_pname, epsilon)
            meta = None
        elif prune_how == "fm-reconstruction":
            pruner = distiller.pruning.FMReconstructionChannelPruner("adc_pruner", group_type,
                                                                     fraction_to_prune, conv_pname)
            meta = {'model': self.model}
        
        pruner.set_param_mask(conv_p, conv_pname, self.zeros_mask_dict, meta=meta)
        del pruner

        if (self.zeros_mask_dict[conv_pname].mask is None or
            calculate_sparsity(self.zeros_mask_dict[conv_pname].mask) == 0):
            msglogger.info("remove_structures: aborting because there are no structures to prune")
            return 0

        # Use the mask to prune
        self.zeros_mask_dict[conv_pname].apply_mask(conv_p)
        
        remove_structures_fn(self.model, self.zeros_mask_dict, self.app_args.arch, self.app_args.dataset, optimizer=None)
        conv_p = distiller.model_find_param(self.model, conv_pname)
        return 1 - (self.get_layer_macs(layer) / macs_before)

    def validate(self):
        assert distiller.model_device(self.model) == 'cuda'
        top1, top5, vloss = self.services.validate_fn(model=self.model)
        return top1, top5, vloss

    def train(self, num_epochs, episode=0):
        # Train for zero or more epochs
        opt_cfg = self.app_args.optimizer_data
        optimizer = torch.optim.SGD(self.model.parameters(), lr=opt_cfg['lr'],
                                    momentum=opt_cfg['momentum'], weight_decay=opt_cfg['weight_decay'])
        compression_scheduler = self.create_scheduler()
        acc_list = []
        for _ in range(num_epochs):
            # Fine-tune the model
            accuracies = self.services.train_fn(model=self.model, compression_scheduler=compression_scheduler,
                                                optimizer=optimizer, epoch=episode)
            acc_list.extend(accuracies)
        del compression_scheduler
        return acc_list


class DistillerWrapperEnvironment(gym.Env):
    def __init__(self, model, app_args, amc_cfg, services):
        self.pylogger = distiller.data_loggers.PythonLogger(msglogger)
        self.tflogger = distiller.data_loggers.TensorBoardLogger(msglogger.logdir)
        self.orig_model = model
        self.app_args = app_args
        self.amc_cfg = amc_cfg
        self.services = services
        modules_list = amc_cfg.modules_dict[app_args.arch]
        self.net_wrapper = NetworkWrapper(model, app_args, services, modules_list)
        self.dense_model_macs, self.dense_model_size = self.net_wrapper.get_model_resources_requirements(model)

        self.reset(init_only=True)
        msglogger.info("Model %s has %d Convolution layers", self.app_args.arch, self.net_wrapper.num_layers())
        msglogger.info("\tTotal MACs: %s" % distiller.pretty_int(self.dense_model_macs))
        log_amc_config(amc_cfg)

        self.episode = 0
        self.best_reward = -1000
        self.action_low = amc_cfg.action_range[0]
        self.action_high = amc_cfg.action_range[1]
        # Gym spaces documentation: https://gym.openai.com/docs/
        if is_using_continuous_action_space(self.amc_cfg.agent_algo):
            if self.amc_cfg.agent_algo == "ClippedPPO-continuous":
                self.action_space = spaces.Box(PPO_MIN, PPO_MAX, shape=(1,))
            else:
                self.action_space = spaces.Box(self.action_low, self.action_high, shape=(1,)) 
            self.action_space.default_action = self.action_low
        else:
            self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(0, float("inf"), shape=(len(Observation._fields),))
        self.stats_file = AMCStatsFile(os.path.join(msglogger.logdir, 'amc.csv'))
        self.ft_stats_file = FineTuneStatsFile(os.path.join(msglogger.logdir, 'ft_top1.csv'))

        self.orig_model = copy.deepcopy(self.net_wrapper.model)
        if self.amc_cfg.pruning_method == "fm-reconstruction":
            if self.amc_cfg.pruning_pattern != "channels":
                raise ValueError("Feature-map reconstruction is only supported when pruning weights channels")
            # For feature-map reconstruction we need to collect a representative set of inter-layer feature-maps
            collect_intermediate_featuremap_samples(self.net_wrapper.model,
                                                    self.net_wrapper.validate, 
                                                    modules_list)
        
        #print(self.orig_model.intermediate_fms['output_fms']['module.layer1.1.conv1'])
        #assert False
        #print(self.orig_model.module.layer2[1].conv1.input_fm) #neta

    def reset(self, init_only=False):
        """Reset the environment.
        This is invoked by the Agent.
        """
        msglogger.info("Resetting the environment (init_only={})".format(init_only))
        self.current_layer_id = 0
        self.prev_action = 0
        self.model = copy.deepcopy(self.orig_model)
        if hasattr(self.net_wrapper.model, 'intermediate_fms'):
            self.model.intermediate_fms = self.net_wrapper.model.intermediate_fms
        self.net_wrapper.reset(self.model)
        self._removed_macs = 0
        self.action_history = []
        self.agent_action_history = []
        self.model_representation = self.get_model_representation()
        if init_only:
            return
        initial_observation = self.get_obs()
        return initial_observation

    def current_layer(self):
        return self.net_wrapper.get_layer(self.current_layer_id)

    def episode_is_done(self):
        return self.current_layer_id == self.net_wrapper.num_layers()

    def removed_macs(self):
        """Return the amount of MACs removed so far.
        This is normalized to the range 0..1
        """
        return self._removed_macs / self.dense_model_macs

    def render(self, mode='human'):
        """Provide some feedback to the user about what's going on.
        This is invoked by the Agent.
        """
        if self.current_layer_id == 0:
            msglogger.info("+" + "-" * 50 + "+")
            msglogger.info("Starting a new episode %d", self.episode)
            msglogger.info("+" + "-" * 50 + "+")

        msglogger.info("Render Environment: current_layer_id=%d" % self.current_layer_id)
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

    def step(self, pruning_action):
        """Take a step, given an action.

        The action represents the desired sparsity for the "current" layer.
        This function is invoked by the Agent.
        """
        msglogger.info("env.step - current_layer_id={}  episode={}".format(self.current_layer_id, self.episode))
        pruning_action = float(pruning_action[0])
        msglogger.info("\tAgent pruning_action={}".format(pruning_action))
        self.agent_action_history.append(pruning_action)

        if is_using_continuous_action_space(self.amc_cfg.agent_algo):
            if self.amc_cfg.agent_algo == "ClippedPPO-continuous":
                # We need to map PPO's infinite action-space (actions sampled from a Gaussian) to our action-space.
                pruning_action = adjust_ppo_output(pruning_action, self.action_high, self.action_low)
            else:
                pruning_action = np.clip(pruning_action, self.action_low, self.action_high)
        else:
            # Divide the action space into 10 discrete levels (0%, 10%, 20%,....90% sparsity)
            pruning_action = pruning_action / 10
        msglogger.info("\tAgent clipped pruning_action={}".format(pruning_action))

        if self.amc_cfg.action_constrain_fn is not None:
            pruning_action = self.amc_cfg.action_constrain_fn(self, pruning_action=pruning_action)
            msglogger.info("Constrained pruning_action={}".format(pruning_action))

        # Calculate the final compression rate
        total_macs_before, _ = self.net_wrapper.get_model_resources_requirements(self.model)
        layer_macs = self.net_wrapper.get_layer_macs(self.current_layer())
        msglogger.info("\tlayer_macs={:.2f}".format(layer_macs / self.dense_model_macs))
        msglogger.info("\tremoved_macs={:.2f}".format(self.removed_macs()))
        msglogger.info("\trest_macs={:.2f}".format(self.rest_macs()))
    
        # print(self.orig_model.module.layer2[1].conv1.input_fm)  # Neta
        #print(self.orig_model.intermediate_fms['output_fms']['module.layer1.1.conv1'])
        if pruning_action > 0:
            pruning_action = self.net_wrapper.remove_structures(self.current_layer_id,
                                                                fraction_to_prune=pruning_action,
                                                                prune_what=self.amc_cfg.pruning_pattern,
                                                                prune_how=self.amc_cfg.pruning_method)
        else:
            pruning_action = 0

        self.action_history.append(pruning_action)
        total_macs_after, _ = self.net_wrapper.get_model_resources_requirements(self.model)
        layer_macs_after_action = self.net_wrapper.get_layer_macs(self.current_layer())

        # Update the various counters after taking the step
        self.current_layer_id += 1
        self._removed_macs += (total_macs_before - total_macs_after)

        msglogger.info("actual_action={}".format(pruning_action))
        msglogger.info("layer_macs={} layer_macs_after_action={} removed now={}".format(layer_macs,
                                                                                        layer_macs_after_action,
                                                                                        (layer_macs - layer_macs_after_action)))
        msglogger.info("self._removed_macs={}".format(self._removed_macs))
        assert math.isclose(layer_macs_after_action / layer_macs, 1 - pruning_action)

        stats = ('Performance/Validation/',
                 OrderedDict([('requested_action', pruning_action)]))

        distiller.log_training_progress(stats, None,
                                        self.episode, steps_completed=self.current_layer_id,
                                        total_steps=self.net_wrapper.num_layers(), log_freq=1, loggers=[self.tflogger])

        if self.episode_is_done():
            msglogger.info("Episode is ending")
            observation = self.get_final_obs()
            reward, top1, total_macs, total_nnz = self.compute_reward()
            normalized_macs = total_macs / self.dense_model_macs * 100
            normalized_nnz = total_nnz / self.dense_model_size * 100
            self.finalize_episode(top1, reward, total_macs, normalized_macs,
                                  normalized_nnz, self.action_history, self.agent_action_history)
            self.episode += 1
        else:
            if self.amc_cfg.ft_frequency is not None and self.current_layer_id % self.amc_cfg.ft_frequency == 0:
                self.net_wrapper.train(1, self.episode)
            observation = self.get_obs()
            if self.amc_cfg.reward_frequency is not None and self.current_layer_id % self.amc_cfg.reward_frequency == 0:
                reward, top1, total_macs, total_nnz = self.compute_reward(False)
            else:
                reward = 0
        self.prev_action = pruning_action
        if self.episode_is_done():
            info = {"accuracy": top1, "compress_ratio": normalized_macs}
        else:
            info = {}
        return observation, reward, self.episode_is_done(), info

    def get_obs(self):
        """Produce a state embedding (i.e. an observation)"""

        current_layer_macs = self.net_wrapper.get_layer_macs(self.current_layer())
        current_layer_macs_pct = current_layer_macs/self.dense_model_macs
        current_layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, current_layer.name)

        obs = self.model_representation[self.current_layer_id, :]
        obs[-1] = self.prev_action
        obs[-2] = self.rest_macs()
        obs[-3] = self.removed_macs()
        msglogger.info("obs={}".format(Observation._make(obs)))
        assert (self.removed_macs() + current_layer_macs_pct + self.rest_macs()) <= 1
        return obs

    def get_final_obs(self):
        """Return the final state embedding (observation)
        The final state is reached after we traverse all of the Convolution layers.
        """
        obs = self.model_representation[-1, :]
        msglogger.info("obs={}".format(Observation._make(obs)))
        return obs

    def get_model_representation(self):
        """Produce a state embedding (i.e. an observation)"""
        num_layers = self.net_wrapper.num_layers()
        network_obs = np.empty(shape=(num_layers, LayerDescLen))
        for layer_id in range(num_layers):
            layer = self.net_wrapper.get_layer(layer_id)
            layer_macs = self.net_wrapper.get_layer_macs(layer)
            #layer_macs_pct = layer_macs/self.dense_model_macs
            conv_module = distiller.model_find_module(self.model, layer.name)
            obs = [layer.t,
                   conv_module.out_channels,
                   conv_module.in_channels,
                   #layer.ifm_h,
                   #layer.ifm_w,
                   layer.stride[0],
                   layer.k,
                   distiller.volume(conv_module.weight),
                   #layer_macs_pct,
                   layer_macs,
                   0,
                   0,
                   0]
            network_obs[layer_id: ] = np.array(obs)

        # ***********************************************************************
        # normalize the state
        #network_obs = np.array(network_obs, 'float')
        #msglogger.info("model representation=\n{}".format(network_obs))
        #print('=> shape of embedding (n_layer * n_dim): {}'.format(layer_embedding.shape))
        #assert len(network_obs.shape) == 2, network_obs.shape
        for feature in range(LayerDescLen):
            feature_vec = network_obs[:, feature]
            fmin = min(feature_vec)
            fmax = max(feature_vec)
            if fmax - fmin > 0:
                network_obs[:, feature] = (feature_vec - fmin) / (fmax - fmin)
        msglogger.debug("model representation=\n{}".format(network_obs))
        return network_obs

    def rest_macs_raw(self):
        """Return the number of remaining MACs in the layers following the current layer"""
        rest = 0
        for layer_id in range(self.current_layer_id, self.net_wrapper.num_layers()):
            rest += self.net_wrapper.get_layer_macs(self.net_wrapper.get_layer(layer_id + 1))
        return rest

    def rest_macs(self):
        return self.rest_macs_raw() / self.dense_model_macs

    def is_macs_constraint_achieved(self, compressed_model_total_macs):
        current_density = compressed_model_total_macs / self.dense_model_macs
        return self.amc_cfg.target_density >= current_density

    def compute_reward(self, log_stats=True):
        """Compute the reward"""
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])
        total_macs, total_nnz = self.net_wrapper.get_model_resources_requirements(self.model)
        if self.amc_cfg.perform_thinning:
            compression = distiller.model_numel(self.model, param_dims=[4]) / self.dense_model_size
        else:
            compression = 1 - distiller.model_sparsity(self.model)/100
            # What a hack!
            total_nnz *= compression

        accuracies = self.net_wrapper.train(self.amc_cfg.num_ft_epochs, self.episode)
        self.ft_stats_file.add_record([self.episode, accuracies])

        top1, top5, vloss = self.net_wrapper.validate()
        reward = self.amc_cfg.reward_fn(self, top1, top5, vloss, total_macs)

        if log_stats:
            macs_normalized = total_macs/self.dense_model_macs
            msglogger.info("Total parameters left: %.2f%%" % (compression*100))
            msglogger.info("Total compute left: %.2f%%" % (total_macs/self.dense_model_macs*100))

            stats = ('Performance/EpisodeEnd/',
                     OrderedDict([('Loss', vloss),
                                  ('Top1', top1),
                                  ('Top5', top5),
                                  ('reward', reward),
                                  ('total_macs', int(total_macs)),
                                  ('macs_normalized', macs_normalized*100),
                                  ('log(total_macs)', math.log(total_macs)),
                                  ('total_nnz', int(total_nnz))]))
            distiller.log_training_progress(stats, None, self.episode, steps_completed=0, total_steps=1,
                                            log_freq=1, loggers=[self.tflogger, self.pylogger])
        return reward, top1, total_macs, total_nnz

    def finalize_episode(self, top1, reward, total_macs, normalized_macs,
                         normalized_nnz, action_history, agent_action_history):
        """Write the details of one network to a CSV file and create a checkpoint file"""
        if reward > self.best_reward:
            self.best_reward = reward
            ckpt_name = self.save_checkpoint(is_best=True)
            msglogger.info("Best reward={}  episode={}  top1={}".format(reward, self.episode, top1))
        else:
            ckpt_name = self.save_checkpoint(is_best=False)

        fields = [self.episode, top1, reward, total_macs, normalized_macs,
                  normalized_nnz, ckpt_name, action_history, agent_action_history]
        self.stats_file.add_record(fields)

    def save_checkpoint(self, is_best=False):
        """Save the learned-model checkpoint"""
        scheduler = self.net_wrapper.create_scheduler()
        episode = str(self.episode).zfill(3)
        if is_best:
            fname = "BEST_adc_episode_{}".format(episode)
        else:
            fname = "adc_episode_{}".format(episode)

        self.services.save_checkpoint_fn(epoch=0, model=self.model,
                                         scheduler=scheduler, name=fname)
        del scheduler
        return fname


def collect_conv_details(model, dataset, perform_thinning, layers_to_prune=None):
    dummy_input = distiller.get_dummy_input(dataset)
    g = SummaryGraph(model, dummy_input)
    conv_layers = OrderedDict()
    total_macs = 0
    total_params = 0
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
            total_params += conv.weights_vol
            conv.macs = conv_op['attrs']['MACs']
            conv_pname = name + ".weight"
            conv_p = distiller.model_find_param(model, conv_pname)
            if not perform_thinning:
                #conv.macs *= distiller.density_ch(conv_p)  # Channel pruning
                conv.macs *= distiller.density_3D(conv_p)   # Filter pruning
            total_macs += conv.macs

            conv.ofm_h = g.param_shape(conv_op['outputs'][0])[2]
            conv.ofm_w = g.param_shape(conv_op['outputs'][0])[3]
            conv.ifm_h = g.param_shape(conv_op['inputs'][0])[2]
            conv.ifm_w = g.param_shape(conv_op['inputs'][0])[3]

            conv.name = name
            conv.id = id
            if layers_to_prune is None or name in layers_to_prune:
                conv_layers[len(conv_layers)] = conv
    return conv_layers, total_macs, total_params


import pandas as pd
def sample_networks(net_wrapper, services):
    """Sample networks from the posterior distribution.

    1. Sort the networks we discovered using AMC by their reward.
    2. Use the top 10% best-performing networks discovered by AMC to postulate a posterior distribution of the
       density/sparsity of each layer:
            p([layers-sparsity] | Top1, L1)
    3. Sample 100 networks from this distribution.
       For each such network: fine-tune, score using Top1, and save
    """
    #fname = "logs/resnet20___2019.01.29-102912/amc.csv"
    fname = "logs/resnet20___2019.02.03-210001/amc.csv"
    df = pd.read_csv(fname)

    #top1_sorted_df = df.sort_values(by=['top1'], ascending=False)
    top1_sorted_df = df.sort_values(by=['reward'], ascending=False)
    top10pct = top1_sorted_df[:int(len(df.index) * 0.1)]

    dense_macs, _ = net_wrapper.get_model_resources_requirements()
    layer_sparsities_list = []
    for index, row in top10pct.iterrows():
        layer_sparsities = row['action_history']
        layer_sparsities = layer_sparsities[1:-1].split(",")  # convert from string to list
        layer_sparsities = [float(sparsity) for sparsity in layer_sparsities]
        layer_sparsities_list.append(layer_sparsities)

    layer_sparsities = np.array(layer_sparsities_list)
    mean = layer_sparsities.mean(axis=0)
    cov = np.cov(layer_sparsities.T)
    num_networks = 100
    data = np.random.multivariate_normal(mean, cov, num_networks)

    orig_model = net_wrapper.model
    for i in range(num_networks):
        model = copy.deepcopy(orig_model)
        net_wrapper.reset(model)
        for layer_id, sparsity_level in enumerate(data[i]):
            sparsity_level = min(max(0, sparsity_level), ALMOST_ONE)
            net_wrapper.remove_structures(layer_id,
                                          fraction_to_prune=sparsity_level,
                                          prune_what="channels")

        net_wrapper.train(1)
        top1, top5, vloss = net_wrapper.validate()

        """Save the learned-model checkpoint"""
        scheduler = net_wrapper.create_scheduler()
        total_macs, _ = net_wrapper.get_model_resources_requirements(model)
        fname = "{}_top1_{:2f}__density_{:2f}_sampled".format(net_wrapper.arch, top1, total_macs/dense_macs)
        services.save_checkpoint_fn(epoch=0, model=net_wrapper.model,
                                    scheduler=scheduler, name=fname)
        del scheduler

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

"""
We thank Prof. Song Han and his team for their help with certain critical parts 
of this implementation.

AMC: AutoML for Model Compression and Acceleration on Mobile Devices.
     Yihui He, Ji Lin, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
     arXiv:1802.03494
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
    print("See instructions in the header of examples/automated_deep_compression/ADC.py")
    raise e
from gym import spaces
import distiller
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from distiller import normalize_module_name, SummaryGraph
from .utils.features_collector import collect_intermediate_featuremap_samples


msglogger = logging.getLogger("examples.automated_deep_compression")
Observation = namedtuple('Observation', ['t', 'n', 'c',  'h', 'w', 'stride', 'k', 'MACs',
                                         'Weights', 'reduced', 'rest', 'prev_a'])
ObservationLen = len(Observation._fields)
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
    return agent in ("DDPG", "ClippedPPO-continuous", "Random-policy", "TD3")


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

    shift = (action_high - action_low) / 2 + action_low
    scale = (action_high - action_low) / (PPO_MAX - PPO_MIN)
    pruning_action = ppo_pruning_action * scale + shift
    return float(pruning_action)


class NetworkMetadata(object):
    def __init__(self, model, dataset, dependency_type, modules_list):
        details = get_network_details(model, dataset, dependency_type, modules_list)
        self.all_layers, self.pruned_idxs, self.dependent_idxs, self._total_macs, self._total_nnz = details
        
    def get_layer_net_macs(self, layer):
        return layer.macs
    
    @property
    def total_macs(self):
        return self._total_macs

    @property
    def total_nnz(self):
        return self._total_nnz

    def layer_macs(self, layer):
        macs = layer.macs
        for dependent_mod in layer.dependencies:
            macs += self.name2layer(dependent_mod).macs
        return macs

    # def compress_layer(self, layer, compression_rate)
    def reduce_layer_macs(self, layer, reduction):
        total_macs_reduced = layer.macs * reduction
        total_nnz_reduced = layer.weights_vol * reduction
        layer.macs -= total_macs_reduced
        layer.weights_vol -= total_nnz_reduced
        for dependent_mod in layer.dependencies:
            macs_reduced = self.name2layer(dependent_mod).macs * reduction
            nnz_reduced = self.name2layer(dependent_mod).weights_vol * reduction
            total_macs_reduced += macs_reduced
            total_nnz_reduced += nnz_reduced
            self.name2layer(dependent_mod).macs -= macs_reduced
            self.name2layer(dependent_mod).weights_vol -= nnz_reduced
        self._total_macs -= total_macs_reduced
        self._total_nnz -= total_nnz_reduced

    def name2layer(self, name):
        layers = [layer for layer in self.all_layers.values() if layer.name == name]
        if len(layers) == 1:
            return layers[0]
        raise ValueError("illegal module name %s" % name)

    def get_model_budget(self): #, model, dataset):
        return self._total_macs, self._total_nnz

    def get_layer(self, layer_id):
        return self.all_layers[layer_id]

    def get_pruned_layer(self, layer_id):
        assert self.is_prunable(layer_id)
        return self.get_layer(layer_id)

    def is_prunable(self, layer_id):
        return layer_id in self.pruned_idxs

    def is_reducible(self, layer_id):
        return layer_id in self.pruned_idxs or layer_id in self.dependent_idxs

    def num_pruned_layers(self):
        return len(self.pruned_idxs)

    def num_layers(self):
        return len(self.all_layers)


class NetworkWrapper(object):
    def __init__(self, model, app_args, services, modules_list, pruning_pattern):
        self.app_args = app_args
        self.services = services
        self.cached_model_metadata = NetworkMetadata(model, app_args.dataset, 
                                                     pruning_pattern, modules_list)
        self.reset(model)

    def reset(self, model):
        self.model = model
        self.zeros_mask_dict = distiller.create_model_masks_dict(self.model)
        self.model_metadata = copy.deepcopy(self.cached_model_metadata)

    def get_resources_requirements(self):
        total_macs, total_nnz = self.model_metadata.get_model_budget()
        return total_macs, total_nnz

    @property
    def arch(self):
        return self.app_args.arch

    def num_pruned_layers(self):
        return self.model_metadata.num_pruned_layers()

    def get_pruned_layer(self, layer_id):
        return self.model_metadata.get_pruned_layer(layer_id)

    def get_layer(self, idx):
       return self.model_metadata.get_layer(idx)

    def layer_macs(self, layer):
        return self.model_metadata.layer_macs(layer)

    def get_layer_net_macs(self, layer):
        return self.model_metadata.get_layer_net_macs(layer)

    def name2layer(self, name):
        return self.model_metadata.name2layer(name)

    @property
    def total_macs(self):
        return self.model_metadata.total_macs

    @property
    def total_nnz(self):
        return self.model_metadata.total_nnz

    def create_scheduler(self):
        scheduler = distiller.CompressionScheduler(self.model)
        masks = {param_name: masker.mask for param_name, masker in self.zeros_mask_dict.items()}
        scheduler.load_state_dict(state={'masks_dict': masks})
        return scheduler

    def remove_structures(self, layer_id, fraction_to_prune, prune_what, prune_how, group_size, apply_thinning):
        """Physically remove channels and corresponding filters from the model

        Returns the compute-sparsity of the layer with index 'layer_id'
        """
        if layer_id not in self.model_metadata.pruned_idxs:
            raise ValueError("idx=%d is not in correct range " % layer_id)
        if fraction_to_prune < 0:
            raise ValueError("fraction_to_prune=%.3f is illegal" % fraction_to_prune)

        if fraction_to_prune == 0:
            return 0
        if fraction_to_prune == 1.0:
            # For now, prevent the removal of entire layers
            fraction_to_prune = ALMOST_ONE

        layer = self.model_metadata.get_pruned_layer(layer_id)
        macs_before = self.get_layer_net_macs(layer)
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)

        msglogger.debug("ADC: trying to remove %.1f%% %s from %s" % (fraction_to_prune*100, prune_what, conv_pname))

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
                                                                        fraction_to_prune, conv_pname,
                                                                        epsilon=epsilon, group_size=group_size)
            meta = None
        elif prune_how == "fm-reconstruction":
            pruner = distiller.pruning.FMReconstructionChannelPruner("adc_pruner", group_type,
                                                                     fraction_to_prune, conv_pname, group_size=group_size)
            meta = {'model': self.model}
        else:
            raise ValueError("Unknown pruning method")
        pruner.set_param_mask(conv_p, conv_pname, self.zeros_mask_dict, meta=meta)
        del pruner

        if (self.zeros_mask_dict[conv_pname].mask is None or 
                0 == calculate_sparsity(self.zeros_mask_dict[conv_pname].mask)):
            msglogger.debug("remove_structures: aborting because there are no structures to prune")
            return 0
        final_action = calculate_sparsity(self.zeros_mask_dict[conv_pname].mask)

        # Use the mask to prune
        self.zeros_mask_dict[conv_pname].apply_mask(conv_p)
        if apply_thinning:     
            remove_structures_fn(self.model, self.zeros_mask_dict, self.app_args.arch, self.app_args.dataset, optimizer=None)

        self.model_metadata.reduce_layer_macs(layer, final_action)
        macs_after = self.get_layer_net_macs(layer)
        assert 1. - (macs_after / macs_before) == final_action
        return final_action

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
        logdir = logging.getLogger().logdir
        self.tflogger = distiller.data_loggers.TensorBoardLogger(logdir)
        self.verbose = False
        if self.verbose:
            loglevel = logging.DEBUG
        else:
            loglevel = logging.INFO
            logging.getLogger().setLevel(logging.WARNING)
            #logging.getLogger("__main__").setLevel(logging.WARNING)
            #logging.getLogger("examples.classifer_compression.compress_classifier").setLevel(logging.WARNING)
        for module in ["examples.automated_deep_compression",
                       "distiller.data_loggers.logger",
                       "distiller.thinning", 
                       "distiller.pruning.ranked_structures_pruner"]:
            logging.getLogger(module).setLevel(loglevel)

        self.orig_model = copy.deepcopy(model)
        self.app_args = app_args
        self.amc_cfg = amc_cfg
        self.services = services
        try:
            modules_list = amc_cfg.modules_dict[app_args.arch]
        except KeyError:
            msglogger.warning("!!! The config file does not specify the modules to compress for %s" % app_args.arch)
            # Default to using all convolution layers
            distiller.assign_layer_fq_names(model)
            modules_list = [mod.distiller_name for mod in model.modules() if type(mod)==torch.nn.Conv2d]
            msglogger.warning("Using the following layers: %s" % ", ".join(modules_list))

        self.net_wrapper = NetworkWrapper(model, app_args, services, modules_list, amc_cfg.pruning_pattern)
        self.dense_model_macs, self.dense_model_size = self.net_wrapper.get_resources_requirements()
        self.reset(init_only=True)
        msglogger.debug("Model %s has %d modules (%d pruned)", self.app_args.arch, 
                                                               self.net_wrapper.model_metadata.num_layers(),
                                                               self.net_wrapper.model_metadata.num_pruned_layers())
        msglogger.debug("\tTotal MACs: %s" % distiller.pretty_int(self.dense_model_macs))
        self._max_episode_steps = self.net_wrapper.model_metadata.num_pruned_layers()  # Hack for Coach-TD3
        log_amc_config(amc_cfg)

        self.episode = 0
        self.best_reward = -1000
        self.action_low = amc_cfg.action_range[0]
        self.action_high = amc_cfg.action_range[1]

        if is_using_continuous_action_space(self.amc_cfg.agent_algo):
            if self.amc_cfg.agent_algo == "ClippedPPO-continuous":
                self.action_space = spaces.Box(PPO_MIN, PPO_MAX, shape=(1,))
            else:
                self.action_space = spaces.Box(self.action_low, self.action_high, shape=(1,)) 
            self.action_space.default_action = self.action_low
        else:
            self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(0, float("inf"), shape=(len(Observation._fields),))
        self.stats_file = AMCStatsFile(os.path.join(logdir, 'amc.csv'))
        self.ft_stats_file = FineTuneStatsFile(os.path.join(logdir, 'ft_top1.csv'))

        if self.amc_cfg.pruning_method == "fm-reconstruction":
            if self.amc_cfg.pruning_pattern != "channels":
                raise ValueError("Feature-map reconstruction is only supported when pruning weights channels")
            
            from functools import partial
            def acceptance_criterion(m, mod_names):
                # Collect feature-maps only for Conv2d layers, if they are in our modules list.
                return isinstance(m, torch.nn.Conv2d) and m.distiller_name in mod_names

            # For feature-map reconstruction we need to collect a representative set 
            # of inter-layer feature-maps
            from distiller.pruning import FMReconstructionChannelPruner
            collect_intermediate_featuremap_samples(
                self.net_wrapper.model,
                self.net_wrapper.validate, 
                partial(acceptance_criterion, mod_names=modules_list),
                partial(FMReconstructionChannelPruner.cache_featuremaps_fwd_hook, 
                        n_points_per_fm=self.amc_cfg.n_points_per_fm))
    @property
    def steps_per_episode(self):
        return self.net_wrapper.model_metadata.num_pruned_layers() 
        
    def reset(self, init_only=False):
        """Reset the environment.
        This is invoked by the Agent.
        """
        msglogger.info("Resetting the environment (init_only={})".format(init_only))
        self.current_state_id = 0
        self.current_layer_id = self.net_wrapper.model_metadata.pruned_idxs[self.current_state_id]
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
        return self.net_wrapper.get_pruned_layer(self.current_layer_id)

    def episode_is_done(self):
        return self.current_state_id == self.net_wrapper.model_metadata.num_pruned_layers()

    @property
    def removed_macs_pct(self):
        """Return the amount of MACs removed so far.
        This is normalized to the range 0..1
        """
        return self._removed_macs / self.dense_model_macs

    def render(self, mode='human'):
        """Provide some feedback to the user about what's going on.
        This is invoked by the Agent.
        """
        if self.current_state_id == 0:
            msglogger.info("+" + "-" * 50 + "+")
            msglogger.info("Starting a new episode %d", self.episode)
            msglogger.info("+" + "-" * 50 + "+")
        if not self.verbose:
            return
        msglogger.info("Render Environment: current_state_id=%d" % self.current_state_id)
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

    def step(self, pruning_action):
        """Take a step, given an action.

        The action represents the desired sparsity for the "current" layer.
        This function is invoked by the Agent.
        """
        msglogger.debug("env.step - current_state_id=%d (%s)  episode=%d" % 
                       (self.current_state_id, self.current_layer().name, self.episode))
        pruning_action = float(pruning_action[0])
        msglogger.debug("\tAgent pruning_action={}".format(pruning_action))
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
        msglogger.debug("\tAgent clipped pruning_action={}".format(pruning_action))

        if self.amc_cfg.action_constrain_fn is not None:
            pruning_action = self.amc_cfg.action_constrain_fn(self, pruning_action=pruning_action)
            msglogger.debug("Constrained pruning_action={}".format(pruning_action))

        # Calculate the final compression rate
        total_macs_before, _ = self.net_wrapper.get_resources_requirements()
        layer_macs = self.net_wrapper.layer_macs(self.current_layer())
        msglogger.debug("\tlayer_macs={:.2f}".format(layer_macs / self.dense_model_macs))
        msglogger.debug("\tremoved_macs={:.2f}".format(self.removed_macs_pct))
        msglogger.debug("\trest_macs={:.2f}".format(self.rest_macs()))
        msglogger.debug("\tcurrent_layer_id = %d" % self.current_layer_id)
        self.current_state_id += 1
        if pruning_action > 0:
            pruning_action = self.net_wrapper.remove_structures(self.current_layer_id,
                                                                fraction_to_prune=pruning_action,
                                                                prune_what=self.amc_cfg.pruning_pattern,
                                                                prune_how=self.amc_cfg.pruning_method,
                                                                group_size=self.amc_cfg.group_size,
                                                                apply_thinning=self.episode_is_done())
        else:
            pruning_action = 0

        self.action_history.append(pruning_action)
        total_macs_after_act, total_nnz_after_act = self.net_wrapper.get_resources_requirements()
        layer_macs_after_action = self.net_wrapper.layer_macs(self.current_layer())

        # Update the various counters after taking the step
        self._removed_macs += (total_macs_before - total_macs_after_act)

        msglogger.debug("\tactual_action={}".format(pruning_action))
        msglogger.debug("\tlayer_macs={} layer_macs_after_action={} removed now={}".format(layer_macs,
                                                                                        layer_macs_after_action,
                                                                                        (layer_macs - layer_macs_after_action)))
        msglogger.debug("\tself._removed_macs={}".format(self._removed_macs))
        assert math.isclose(layer_macs_after_action / layer_macs, 1 - pruning_action)

        stats = ('Performance/Validation/',
                 OrderedDict([('requested_action', pruning_action)]))

        distiller.log_training_progress(stats, None,
                                        self.episode, steps_completed=self.current_state_id,
                                        total_steps=self.net_wrapper.num_pruned_layers(), log_freq=1, loggers=[self.tflogger])

        if self.episode_is_done():
            msglogger.info("Episode is ending")
            observation = self.get_final_obs()
            reward, top1 = self.compute_reward(total_macs_after_act, total_nnz_after_act)
            normalized_macs = total_macs_after_act / self.dense_model_macs * 100
            normalized_nnz = total_macs_after_act / self.dense_model_size * 100
            self.finalize_episode(top1, reward, total_macs_after_act, normalized_macs,
                                  normalized_nnz, self.action_history, self.agent_action_history)
            self.episode += 1
        else:
            self.current_layer_id = self.net_wrapper.model_metadata.pruned_idxs[self.current_state_id]

            if self.amc_cfg.ft_frequency is not None and self.current_state_id % self.amc_cfg.ft_frequency == 0:
                self.net_wrapper.train(1, self.episode)
            observation = self.get_obs()
            if self.amc_cfg.reward_frequency is not None and self.current_state_id % self.amc_cfg.reward_frequency == 0:
                reward, top1 = self.compute_reward(total_macs_after_act, total_nnz_after_act, log_stats=False)
            else:
                reward = 0
        self.prev_action = pruning_action
        if self.episode_is_done():
            info = {"accuracy": top1, "compress_ratio": normalized_macs}
            msglogger.info(self.removed_macs_pct)
            if self.amc_cfg.protocol == "mac-constrained":
                # Sanity check (special case only for "mac-constrained")
                assert self.removed_macs_pct >= 1 - self.amc_cfg.target_density - 0.01
                pass
        else:
            info = {}
        return observation, reward, self.episode_is_done(), info

    def get_obs(self):
        """Produce a state embedding (i.e. an observation)"""
        current_layer_macs = self.net_wrapper.get_layer_net_macs(self.current_layer())
        current_layer_macs_pct = current_layer_macs/self.dense_model_macs
        current_layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, current_layer.name)

        obs = self.model_representation[self.current_state_id, :]
        obs[-1] = self.prev_action
        obs[-2] = self.rest_macs()
        obs[-3] = self.removed_macs_pct
        msglogger.debug("obs={}".format(Observation._make(obs)))
        # Sanity check
        assert (self.removed_macs_pct + current_layer_macs_pct + self.rest_macs()) <= 1
        return obs

    def get_final_obs(self):
        """Return the final state embedding (observation).

        The final state is reached after we traverse all of the Convolution layers.
        """
        obs = self.model_representation[-1, :]
        msglogger.debug("obs={}".format(Observation._make(obs)))
        return obs

    def get_model_representation(self):
        """Initialize an embedding representation of the entire model.

        At runtime, a specific row in the embedding matrix is chosen (depending on
        the current state) and the dynamic fields in the resulting state-embedding
        vector are updated. 
        """
        num_states = self.net_wrapper.num_pruned_layers()
        network_obs = np.empty(shape=(num_states, ObservationLen))
        for state_id, layer_id in enumerate(self.net_wrapper.model_metadata.pruned_idxs):
            layer = self.net_wrapper.get_layer(layer_id)
            layer_macs = self.net_wrapper.layer_macs(layer)
            conv_module = distiller.model_find_module(self.model, layer.name)
            obs = [state_id,
                   conv_module.out_channels,
                   conv_module.in_channels,
                   layer.ifm_h,
                   layer.ifm_w,
                   layer.stride[0],
                   layer.k,
                   distiller.volume(conv_module.weight),
                   layer_macs,
                   0, 0, 0]
            network_obs[state_id:] = np.array(obs)

        # Feature normalization
        for feature in range(ObservationLen):
            feature_vec = network_obs[:, feature]
            fmin = min(feature_vec)
            fmax = max(feature_vec)
            if fmax - fmin > 0:
                network_obs[:, feature] = (feature_vec - fmin) / (fmax - fmin)
        # msglogger.debug("model representation=\n{}".format(network_obs))
        return network_obs

    def rest_macs_raw(self):
        """Return the number of remaining MACs in the layers following the current layer"""
        rest, prunable_rest = 0, 0
        prunable_layers, rest_layers, layers_to_ignore = list(), list(), list()

        # Create a list of the IDs of the layers that are dependent on the current_layer.
        # We want to ignore these layers when we compute prunable_layers (and prunable_rest).
        for dependent_mod in self.current_layer().dependencies:
            layers_to_ignore.append(self.net_wrapper.name2layer(dependent_mod).id)

        for layer_id in range(self.current_layer_id+1, self.net_wrapper.model_metadata.num_layers()):
            layer_macs = self.net_wrapper.get_layer_net_macs(self.net_wrapper.get_layer(layer_id))
            if self.net_wrapper.model_metadata.is_reducible(layer_id):
                if layer_id not in layers_to_ignore:
                    prunable_layers.append((layer_id, self.net_wrapper.get_layer(layer_id).name, layer_macs))
                    prunable_rest += layer_macs
            else:
                rest_layers.append((layer_id, self.net_wrapper.get_layer(layer_id).name, layer_macs))
                rest += layer_macs

        msglogger.debug("prunable_layers={} rest_layers={}".format(prunable_layers, rest_layers))
        msglogger.debug("layer_id=%d, prunable_rest=%.3f rest=%.3f" % (self.current_layer_id, prunable_rest, rest))
        return prunable_rest, rest

    def rest_macs(self):
        return sum(self.rest_macs_raw()) / self.dense_model_macs

    def is_macs_constraint_achieved(self, compressed_model_total_macs):
        current_density = compressed_model_total_macs / self.dense_model_macs
        return self.amc_cfg.target_density >= current_density

    def compute_reward(self, total_macs, total_nnz, log_stats=True):
        """Compute the reward"""
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])
        compression = distiller.model_numel(self.model, param_dims=[4]) / self.dense_model_size

        # Fine-tune (this is a nop if self.amc_cfg.num_ft_epochs==0)
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
        return reward, top1

    def finalize_episode(self, top1, reward, total_macs, normalized_macs,
                         normalized_nnz, action_history, agent_action_history):
        """Write the details of one network to a CSV file and create a checkpoint file"""
        if reward > self.best_reward:
            self.best_reward = reward
            ckpt_name = self.save_checkpoint(is_best=True)
            msglogger.info("Best reward={}  episode={}  top1={}".format(reward, self.episode, top1))
        else:
            ckpt_name = ""
            if self.amc_cfg.save_chkpts:
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


def get_network_details(model, dataset, dependency_type, layers_to_prune=None):
    dummy_input = distiller.get_dummy_input(dataset)
    g = SummaryGraph(model, dummy_input)
    all_layers = OrderedDict()
    pruned_indices = list()
    dependent_layers = set()
    total_macs = 0
    total_params = 0
    layers = OrderedDict({mod_name: m for mod_name, m in model.named_modules() 
                          if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))})
    for id, (name, m) in enumerate(layers.items()):
        if isinstance(m, torch.nn.Conv2d):
            conv = SimpleNamespace()
            conv.t = len(pruned_indices)
            conv.k = m.kernel_size[0]
            conv.stride = m.stride

            # Use the SummaryGraph to obtain some other details of the models
            conv_op = g.find_op(normalize_module_name(name))
            assert conv_op is not None

            conv.weights_vol = conv_op['attrs']['weights_vol']
            total_params += conv.weights_vol
            conv.macs = conv_op['attrs']['MACs']
            conv.n_ofm = conv_op['attrs']['n_ofm']
            conv.n_ifm = conv_op['attrs']['n_ifm']
            conv_pname = name + ".weight"
            conv_p = distiller.model_find_param(model, conv_pname)
            total_macs += conv.macs

            conv.ofm_h = g.param_shape(conv_op['outputs'][0])[2]
            conv.ofm_w = g.param_shape(conv_op['outputs'][0])[3]
            conv.ifm_h = g.param_shape(conv_op['inputs'][0])[2]
            conv.ifm_w = g.param_shape(conv_op['inputs'][0])[3]

            conv.name = name
            conv.id = id
            all_layers[id] = conv
            if layers_to_prune is None or name in layers_to_prune:
                pruned_indices.append(id)
                # Find the data-dependent layers
                from .utils.data_dependencies import find_dependencies
                conv.dependencies = list()
                find_dependencies(dependency_type, g, all_layers, name, conv.dependencies)
                dependent_layers.add(tuple(conv.dependencies))
        elif isinstance(m, torch.nn.Linear):
            fc = SimpleNamespace()

            # Use the SummaryGraph to obtain some other details of the models
            fc_op = g.find_op(normalize_module_name(name))
            assert fc_op is not None

            fc.weights_vol = fc_op['attrs']['weights_vol']
            total_params += conv.weights_vol
            fc.macs = fc_op['attrs']['MACs']
            fc.n_ofm = fc_op['attrs']['n_ofm']
            fc.n_ifm = fc_op['attrs']['n_ifm']
            fc_pname = name + ".weight"
            fc_p = distiller.model_find_param(model, fc_pname)
            total_macs += fc.macs
            fc.name = name
            fc.id = id
            all_layers[id] = fc
 
    def convert_layer_names_to_indices(layer_names):
        """Args:
            layer_names - list of layer names
           Returns:
            list of layer indices
        """
        layer_indices = [index for name in layer_names for index, 
                            layer in all_layers.items() if layer.name == name[0]]
        return layer_indices

    dependent_indices = convert_layer_names_to_indices(dependent_layers)
    return all_layers, pruned_indices, dependent_indices, total_macs, total_params


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

    dense_macs, _ = net_wrapper.get_resources_requirements()
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
        total_macs, _ = net_wrapper.get_resources_requirements()
        fname = "{}_top1_{:2f}__density_{:2f}_sampled".format(net_wrapper.arch, top1, total_macs/dense_macs)
        services.save_checkpoint_fn(epoch=0, model=net_wrapper.model,
                                    scheduler=scheduler, name=fname)
        del scheduler

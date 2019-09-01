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
import gym
import distiller
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from distiller import normalize_module_name, SummaryGraph
from utils.features_collector import collect_intermediate_featuremap_samples
from utils.ac_loggers import AMCStatsLogger, FineTuneStatsLogger


msglogger = logging.getLogger("examples.auto_compression.amc")
Observation = namedtuple('Observation', ['t', 'type', 'n', 'c',  'h', 'w', 'stride', 'k', 'MACs',
                                         'Weights', 'reduced', 'rest', 'prev_a'])
ObservationLen = len(Observation._fields)
ALMOST_ONE = 0.9999


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
    """
    We need to map PPO's infinite action-space to our action-space (in PPO actions are sampled
    from a normalized Gaussian N(0,1)).
    We do this by a shift & scale operation:
    1. We crudely assume that most of the Gaussian maps to actions in a limited range {PPO_MIN..PPO_MAX}
        which allows us to clip the action, and scale it to the range of the actual action-space.
    2. We assume that the distribution the agent samples from is centered at zero, and so we shift
        actions to the center of the actual action-space.
    """
    shift = (action_high - action_low) / 2 + action_low
    scale = (action_high - action_low) / (PPO_MAX - PPO_MIN)
    pruning_action = ppo_pruning_action * scale + shift
    return float(pruning_action)


class NetworkMetadata(object):
    def __init__(self, model, dataset, dependency_type, modules_list):
        details = get_network_details(model, dataset, dependency_type, modules_list)
        self.all_layers, self.pruned_idxs, self.dependent_idxs, self._total_macs, self._total_nnz = details

    @property
    def total_macs(self):
        return self._total_macs

    @property
    def total_nnz(self):
        return self._total_nnz

    def layer_net_macs(self, layer):
        """Returns a MACs of a specific layer"""
        return layer.macs

    def layer_macs(self, layer):
        """Returns a MACs of a specific layer, with the impact on pruning-dependent layers"""
        macs = layer.macs
        for dependent_mod in layer.dependencies:
            macs += self.name2layer(dependent_mod).macs
        return macs

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

    def model_budget(self):
        return self._total_macs, self._total_nnz

    def get_layer(self, layer_id):
        return self.all_layers[layer_id]

    def get_pruned_layer(self, layer_id):
        assert self.is_prunable(layer_id)
        return self.get_layer(layer_id)

    def is_prunable(self, layer_id):
        return layer_id in self.pruned_idxs

    def is_compressible(self, layer_id):
        return layer_id in (self.pruned_idxs + self.dependent_idxs)

    def num_pruned_layers(self):
        return len(self.pruned_idxs)

    def num_layers(self):
        return len(self.all_layers)

    def performance_summary(self):
        # return OrderedDict({layer.name: (layer.macs, layer.weights_vol)
        #                    for layer in self.all_layers.values()})
        return OrderedDict({layer.name: layer.macs
                           for layer in self.all_layers.values()})


class NetworkWrapper(object):
    def __init__(self, model, app_args, services, modules_list, pruning_pattern):
        self.app_args = app_args
        self.services = services
        self.cached_model_metadata = NetworkMetadata(model, app_args.dataset, 
                                                     pruning_pattern, modules_list)
        self.cached_perf_summary = self.cached_model_metadata.performance_summary()
        self.reset(model)
        self.sparsification_masks = None

    def reset(self, model):
        self.model = model
        self.zeros_mask_dict = distiller.create_model_masks_dict(self.model)
        self.model_metadata = copy.deepcopy(self.cached_model_metadata)

    def get_resources_requirements(self):
        total_macs, total_nnz = self.model_metadata.model_budget()
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

    def layer_net_macs(self, layer):
        return self.model_metadata.layer_net_macs(layer)

    def name2layer(self, name):
        return self.model_metadata.name2layer(name)

    @property
    def total_macs(self):
        return self.model_metadata.total_macs

    @property
    def total_nnz(self):
        return self.model_metadata.total_nnz

    def performance_summary(self):
        """Return a dictionary representing the performance the model.

        We calculate the performance of each layer relative to the original (uncompressed) model.
        """
        current_perf = self.model_metadata.performance_summary()
        ret = OrderedDict()
        #return OrderedDict({k: v/v_baseline for ((k, v), (v_baseline)) in zip(current_perf.items(), self.cached_perf_summary.values())})
        for k, v in current_perf.items():
            ret[k] = v / self.cached_perf_summary[k]
        return ret

    def create_scheduler(self):
        scheduler = distiller.CompressionScheduler(self.model, self.zeros_mask_dict)
        return scheduler

    def remove_structures(self, layer_id, fraction_to_prune, prune_what, prune_how, 
                          group_size, apply_thinning, ranking_noise):
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
        macs_before = self.layer_net_macs(layer)
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)

        msglogger.debug("ADC: trying to remove %.1f%% %s from %s" % (fraction_to_prune*100, prune_what, conv_pname))

        if prune_what == "channels":
            calculate_sparsity = distiller.sparsity_ch
            if layer.type == "Linear":
                calculate_sparsity = distiller.sparsity_rows
            remove_structures_fn = distiller.remove_channels
            group_type = "Channels"
        elif prune_what == "filters":
            calculate_sparsity = distiller.sparsity_3D
            group_type = "Filters"
            remove_structures_fn = distiller.remove_filters
        else:
            raise ValueError("unsupported structure {}".format(prune_what))

        if prune_how in ["l1-rank", "stochastic-l1-rank"]:
            # Create a channel/filter-ranking pruner
            pruner = distiller.pruning.L1RankedStructureParameterPruner(
                "auto_pruner", group_type, fraction_to_prune, conv_pname,
                noise=ranking_noise, group_size=group_size)
            meta = None
        elif prune_how == "fm-reconstruction":
            pruner = distiller.pruning.FMReconstructionChannelPruner(
                "auto_pruner", group_type, fraction_to_prune, conv_pname, 
                group_size, math.ceil, ranking_noise=ranking_noise)
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
            self.cache_spasification_masks()
            remove_structures_fn(self.model, self.zeros_mask_dict, self.app_args.arch, self.app_args.dataset, optimizer=None)

        self.model_metadata.reduce_layer_macs(layer, final_action)
        macs_after = self.layer_net_macs(layer)
        assert 1. - (macs_after / macs_before) == final_action
        return final_action

    def validate(self):
        top1, top5, vloss = self.services.validate_fn(model=self.model)
        return top1, top5, vloss

    def train(self, num_epochs, episode=0):
        """Train for zero or more epochs"""
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

    def cache_spasification_masks(self):
        masks = {param_name: masker.mask for param_name, masker in self.zeros_mask_dict.items()}
        self.sparsification_masks = copy.deepcopy(masks)


class DistillerWrapperEnvironment(gym.Env):
    def __init__(self, model, app_args, amc_cfg, services):
        self.pylogger = distiller.data_loggers.PythonLogger(
            logging.getLogger("examples.auto_compression.amc.summaries"))
        logdir = logging.getLogger().logdir
        self.tflogger = distiller.data_loggers.TensorBoardLogger(logdir)
        self._render = False
        self.orig_model = copy.deepcopy(model)
        self.app_args = app_args
        self.amc_cfg = amc_cfg
        self.services = services

        try:
            modules_list = amc_cfg.modules_dict[app_args.arch]
        except KeyError:
            raise ValueError("The config file does not specify the modules to compress for %s" % app_args.arch)
        self.net_wrapper = NetworkWrapper(model, app_args, services, modules_list, amc_cfg.pruning_pattern)
        self.original_model_macs, self.original_model_size = self.net_wrapper.get_resources_requirements()
        self.reset(init_only=True)
        self._max_episode_steps = self.net_wrapper.model_metadata.num_pruned_layers()  # Hack for Coach-TD3
        self.episode = 0
        self.best_reward = float("-inf")
        self.action_low, self.action_high = amc_cfg.action_range
        #self.action_high = amc_cfg.action_range[1]
        self._log_model_info()
        log_amc_config(amc_cfg)
        self._configure_action_space()
        self.observation_space = gym.spaces.Box(0, float("inf"), shape=(len(Observation._fields),))
        self.stats_logger = AMCStatsLogger(os.path.join(logdir, 'amc.csv'))
        self.ft_stats_logger = FineTuneStatsLogger(os.path.join(logdir, 'ft_top1.csv'))

        if self.amc_cfg.pruning_method == "fm-reconstruction":
            self._collect_fm_reconstruction_samples(modules_list)

    def _collect_fm_reconstruction_samples(self, modules_list):
        """Run the forward-pass on the selected dataset and collect feature-map samples.

        These data will be used when we optimize the compressed-net's weights by trying
        to reconstruct these samples.
        """
        from functools import partial
        if self.amc_cfg.pruning_pattern != "channels":
            raise ValueError("Feature-map reconstruction is only supported when pruning weights channels")

        def acceptance_criterion(m, mod_names):
            # Collect feature-maps only for Conv2d layers, if they are in our modules list.
            return isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)) and m.distiller_name in mod_names

        # For feature-map reconstruction we need to collect a representative set
        # of inter-layer feature-maps
        from distiller.pruning import FMReconstructionChannelPruner
        collect_intermediate_featuremap_samples(
            self.net_wrapper.model,
            self.net_wrapper.validate,
            partial(acceptance_criterion, mod_names=modules_list),
            partial(FMReconstructionChannelPruner.cache_featuremaps_fwd_hook,
                    n_points_per_fm=self.amc_cfg.n_points_per_fm))

    def _log_model_info(self):
        msglogger.debug("Model %s has %d modules (%d pruned)", self.app_args.arch,
                                                               self.net_wrapper.model_metadata.num_layers(),
                                                               self.net_wrapper.model_metadata.num_pruned_layers())
        msglogger.debug("\tTotal MACs: %s" % distiller.pretty_int(self.original_model_macs))
        msglogger.debug("\tTotal weights: %s" % distiller.pretty_int(self.original_model_size))

    def _configure_action_space(self):
        if is_using_continuous_action_space(self.amc_cfg.agent_algo):
            if self.amc_cfg.agent_algo == "ClippedPPO-continuous":
                self.action_space = gym.spaces.Box(PPO_MIN, PPO_MAX, shape=(1,))
            else:
                self.action_space = gym.spaces.Box(self.action_low, self.action_high, shape=(1,))
            self.action_space.default_action = self.action_low
        else:
            self.action_space = gym.spaces.Discrete(10)


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
        self.removed_macs = 0
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
        return self.removed_macs / self.original_model_macs

    def render(self, mode='human'):
        """Provide some feedback to the user about what's going on.
        This is invoked by the Agent.
        """
        if not self._render:
            return
        msglogger.info("Render Environment: current_state_id=%d" % self.current_state_id)
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

    def step(self, pruning_action):
        """Take a step, given an action.

        The action represents the desired sparsity for the "current" layer (i.e. the percentage of weights to remove).
        This function is invoked by the Agent.
        """
        if self.current_state_id == 0:
            msglogger.info("+" + "-" * 50 + "+")
            msglogger.info("Episode %d is starting" % self.episode)

        pruning_action = float(pruning_action[0])
        msglogger.debug("env.step - current_state_id=%d (%s) episode=%d action=%.2f" %
                        (self.current_state_id, self.current_layer().name, self.episode, pruning_action))
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
        msglogger.debug("\tlayer_macs={:.2f}".format(layer_macs / self.original_model_macs))
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
                                                                apply_thinning=self.episode_is_done(),
                                                                ranking_noise=self.amc_cfg.ranking_noise)
                                                                #random_state=self.random_state)
        else:
            pruning_action = 0

        self.action_history.append(pruning_action)
        total_macs_after_act, total_nnz_after_act = self.net_wrapper.get_resources_requirements()
        layer_macs_after_action = self.net_wrapper.layer_macs(self.current_layer())

        # Update the various counters after taking the step
        self.removed_macs += (total_macs_before - total_macs_after_act)

        msglogger.debug("\tactual_action={}".format(pruning_action))
        msglogger.debug("\tlayer_macs={} layer_macs_after_action={} removed now={}".format(layer_macs,
                                                                                        layer_macs_after_action,
                                                                                        (layer_macs - layer_macs_after_action)))
        msglogger.debug("\tself._removed_macs={}".format(self.removed_macs))
        assert math.isclose(layer_macs_after_action / layer_macs, 1 - pruning_action)

        stats = ('Performance/Validation/',
                 OrderedDict([('requested_action', pruning_action)]))

        distiller.log_training_progress(stats, None,
                                        self.episode, steps_completed=self.current_state_id,
                                        total_steps=self.net_wrapper.num_pruned_layers(), log_freq=1, loggers=[self.tflogger])

        if self.episode_is_done():
            msglogger.info("Episode %d is ending" % self.episode)
            observation = self.get_final_obs()
            reward, top1, top5, vloss = self.compute_reward(total_macs_after_act, total_nnz_after_act)
            self.finalize_episode(reward, (top1, top5, vloss), total_macs_after_act, total_nnz_after_act,
                                  self.action_history, self.agent_action_history)
            self.episode += 1
        else:
            self.current_layer_id = self.net_wrapper.model_metadata.pruned_idxs[self.current_state_id]

            if self.amc_cfg.ft_frequency is not None and self.current_state_id % self.amc_cfg.ft_frequency == 0:
                self.net_wrapper.train(1, self.episode)
            observation = self.get_obs()
            if self.amc_cfg.reward_frequency is not None and self.current_state_id % self.amc_cfg.reward_frequency == 0:
                reward, top1, top5, vloss = self.compute_reward(total_macs_after_act, total_nnz_after_act)
            else:
                reward = 0
        self.prev_action = pruning_action
        if self.episode_is_done():
            normalized_macs = total_macs_after_act / self.original_model_macs * 100
            info = {"accuracy": top1, "compress_ratio": normalized_macs}
            if self.amc_cfg.protocol == "mac-constrained":
                # Sanity check (special case only for "mac-constrained")
                assert self.removed_macs_pct >= 1 - self.amc_cfg.target_density - 0.002 # 0.01
                pass
        else:
            info = {}
        return observation, reward, self.episode_is_done(), info

    def get_obs(self):
        """Produce a state embedding (i.e. an observation)"""
        current_layer_macs = self.net_wrapper.layer_net_macs(self.current_layer())
        current_layer_macs_pct = current_layer_macs/self.original_model_macs

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
            mod = distiller.model_find_module(self.model, layer.name)
            if isinstance(mod, torch.nn.Conv2d):
                obs = [state_id,
                       0,
                       mod.out_channels,
                       mod.in_channels,
                       layer.ifm_h,
                       layer.ifm_w,
                       layer.stride[0],
                       layer.k,
                       distiller.volume(mod.weight),
                       layer_macs,
                       0, 0, 0]
            elif isinstance(mod, torch.nn.Linear):
                obs = [state_id,
                       1,
                       mod.out_features,
                       mod.in_features,
                       layer.ifm_h,
                       layer.ifm_w,
                       0,
                       1,
                       distiller.volume(mod.weight),
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
        nonprunable_rest, prunable_rest = 0, 0
        prunable_layers, nonprunable_layers, layers_to_ignore = list(), list(), list()

        # Create a list of the IDs of the layers that are dependent on the current_layer.
        # We want to ignore these layers when we compute prunable_layers (and prunable_rest).
        for dependent_mod in self.current_layer().dependencies:
            layers_to_ignore.append(self.net_wrapper.name2layer(dependent_mod).id)

        for layer_id in range(self.current_layer_id+1, self.net_wrapper.model_metadata.num_layers()):
            layer_macs = self.net_wrapper.layer_net_macs(self.net_wrapper.get_layer(layer_id))
            if self.net_wrapper.model_metadata.is_compressible(layer_id):
                if layer_id not in layers_to_ignore:
                    prunable_layers.append((layer_id, self.net_wrapper.get_layer(layer_id).name, layer_macs))
                    prunable_rest += layer_macs

        for layer_id in list(range(0, self.net_wrapper.model_metadata.num_layers())):
            if not self.net_wrapper.model_metadata.is_compressible(layer_id): #and
                layer_macs = self.net_wrapper.layer_net_macs(self.net_wrapper.get_layer(layer_id))
                nonprunable_layers.append((layer_id, self.net_wrapper.get_layer(layer_id).name, layer_macs))
                nonprunable_rest += layer_macs

        msglogger.debug("prunable_layers={} nonprunable_layers={}".format(prunable_layers, nonprunable_layers))
        msglogger.debug("layer_id=%d (%s), prunable_rest=%.3f nonprunable_rest=%.3f" %
                        (self.current_layer_id, self.current_layer().name, prunable_rest, nonprunable_rest))
        return prunable_rest, nonprunable_rest

    def rest_macs(self):
        return sum(self.rest_macs_raw()) / self.original_model_macs

    def is_macs_constraint_achieved(self, compressed_model_total_macs):
        current_density = compressed_model_total_macs / self.original_model_macs
        return self.amc_cfg.target_density >= current_density

    def compute_reward(self, total_macs, total_nnz):
        """Compute the reward.

        We use the validation dataset (the size of the validation dataset is
        configured when the data-loader is instantiated)"""
        num_elements = distiller.model_params_size(self.model, param_dims=[2, 4], param_types=['weight'])

        # Fine-tune (this is a nop if self.amc_cfg.num_ft_epochs==0)
        accuracies = self.net_wrapper.train(self.amc_cfg.num_ft_epochs, self.episode)
        self.ft_stats_logger.add_record([self.episode, accuracies])

        top1, top5, vloss = self.net_wrapper.validate()
        reward = self.amc_cfg.reward_fn(self, top1, top5, vloss, total_macs)
        return reward, top1, top5, vloss

    def finalize_episode(self, reward, val_results, total_macs, total_nnz,
                         action_history, agent_action_history, log_stats=True):
        """Write the details of one network to the logger and create a checkpoint file"""
        top1, top5, vloss = val_results
        normalized_macs = total_macs / self.original_model_macs * 100
        normalized_nnz = total_nnz / self.original_model_size * 100

        if reward > self.best_reward:
            self.best_reward = reward
            ckpt_name = self.save_checkpoint(is_best=True)
            msglogger.info("Best reward={}  episode={}  top1={}".format(reward, self.episode, top1))
        else:
            ckpt_name = self.save_checkpoint(is_best=False)

        import json
        performance = self.net_wrapper.performance_summary()
        fields = [self.episode, top1, reward, total_macs, normalized_macs, normalized_nnz,
                  ckpt_name, json.dumps(action_history), json.dumps(agent_action_history),
                  json.dumps(performance)]
        self.stats_logger.add_record(fields)
        msglogger.info("Top1: %.2f - compute: %.2f%% - params:%.2f%% - actions: %s",
                       top1, normalized_macs, normalized_nnz, self.action_history)
        if log_stats:
            stats = ('Performance/EpisodeEnd/',
                     OrderedDict([('Loss', vloss),
                                  ('Top1', top1),
                                  ('Top5', top5),
                                  ('reward', reward),
                                  ('total_macs', int(total_macs)),
                                  ('macs_normalized', normalized_macs),
                                  ('log(total_macs)', math.log(total_macs)),
                                  ('total_nnz', int(total_nnz))]))
            distiller.log_training_progress(stats, None, self.episode, steps_completed=0, total_steps=1,
                                            log_freq=1, loggers=[self.tflogger, self.pylogger])

    def save_checkpoint(self, is_best=False):
        """Save the learned-model checkpoint"""
        episode = str(self.episode).zfill(3)
        if is_best:
            fname = "BEST_adc_episode_{}".format(episode)
        else:
            fname = "adc_episode_{}".format(episode)
        if is_best or self.amc_cfg.save_chkpts:
            # Always save the best episodes, and depending on amc_cfg.save_chkpts save all other episodes
            scheduler = self.net_wrapper.create_scheduler()
            extras = {"creation_masks": self.net_wrapper.sparsification_masks}
            self.services.save_checkpoint_fn(epoch=0, model=self.model,
                                             scheduler=scheduler, name=fname, extras=extras)
            del scheduler
        return fname


def get_network_details(model, dataset, dependency_type, layers_to_prune=None):
    def make_conv(model, conv_module, g, name, seq_id, layer_id):
        conv = SimpleNamespace()
        conv.type = "Conv2D"
        conv.name = name
        conv.id = layer_id
        conv.t = seq_id
        conv.k = conv_module.kernel_size[0]
        conv.stride = conv_module.stride

        # Use the SummaryGraph to obtain some other details of the models
        conv_op = g.find_op(normalize_module_name(name))
        assert conv_op is not None

        conv.weights_vol = conv_op['attrs']['weights_vol']
        conv.macs = conv_op['attrs']['MACs']
        conv.n_ofm = conv_op['attrs']['n_ofm']
        conv.n_ifm = conv_op['attrs']['n_ifm']
        conv_pname = name + ".weight"
        conv_p = distiller.model_find_param(model, conv_pname)
        conv.ofm_h = g.param_shape(conv_op['outputs'][0])[2]
        conv.ofm_w = g.param_shape(conv_op['outputs'][0])[3]
        conv.ifm_h = g.param_shape(conv_op['inputs'][0])[2]
        conv.ifm_w = g.param_shape(conv_op['inputs'][0])[3]
        return conv

    def make_fc(model, fc_module, g, name, seq_id, layer_id):
        fc = SimpleNamespace()
        fc.type = "Linear"
        fc.name = name
        fc.id = layer_id
        fc.t = seq_id

        # Use the SummaryGraph to obtain some other details of the models
        fc_op = g.find_op(normalize_module_name(name))
        assert fc_op is not None

        fc.weights_vol = fc_op['attrs']['weights_vol']
        fc.macs = fc_op['attrs']['MACs']
        fc.n_ofm = fc_op['attrs']['n_ofm']
        fc.n_ifm = fc_op['attrs']['n_ifm']
        fc_pname = name + ".weight"
        fc_p = distiller.model_find_param(model, fc_pname)
        fc.ofm_h = g.param_shape(fc_op['outputs'][0])[0]
        fc.ofm_w = g.param_shape(fc_op['outputs'][0])[1]
        fc.ifm_h = g.param_shape(fc_op['inputs'][0])[0]
        fc.ifm_w = g.param_shape(fc_op['inputs'][0])[1]

        return fc

    dummy_input = distiller.get_dummy_input(dataset)
    g = SummaryGraph(model, dummy_input)
    all_layers = OrderedDict()
    pruned_indices = list()
    dependent_layers = set()
    total_macs = 0
    total_params = 0

    unfiltered_layers = layers_topological_order(model, dummy_input)
    mods = dict(model.named_modules())
    layers = OrderedDict({mod_name: mods[mod_name] for mod_name in unfiltered_layers
                          if mod_name in mods and
                          isinstance(mods[mod_name], (torch.nn.Conv2d, torch.nn.Linear))})

    # layers = OrderedDict({mod_name: m for mod_name, m in model.named_modules()
    #                       if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))})
    for layer_id, (name, m) in enumerate(layers.items()):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if isinstance(m, torch.nn.Conv2d):
                new_layer = make_conv(model, m, g, name, seq_id=len(pruned_indices), layer_id=layer_id)
                all_layers[layer_id] = new_layer
                total_params += new_layer.weights_vol
                total_macs += new_layer.macs
            elif isinstance(m, torch.nn.Linear):
                new_layer = make_fc(model, m, g, name, seq_id=len(pruned_indices), layer_id=layer_id)
                all_layers[layer_id] = new_layer
                total_params += new_layer.weights_vol
                total_macs += new_layer.macs

            if layers_to_prune is None or name in layers_to_prune:
                pruned_indices.append(layer_id)
                # Find the data-dependent layers of this convolution
                from utils.data_dependencies import find_dependencies
                new_layer.dependencies = list()
                find_dependencies(dependency_type, g, all_layers, name, new_layer.dependencies)
                dependent_layers.add(tuple(new_layer.dependencies))

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


def layers_topological_order(model, dummy_input, recurrent=False):
    """
    Prepares an ordered list of layers to quantize sequentially. This list has all the layers ordered by their
    topological order in the graph.
    Args:
        model (nn.Module): the model to quantize.
        dummy_input (torch.Tensor): an input to be passed through the model.
        recurrent (bool): indication on whether the model might have recurrent connections.
    """

    class _OpRank:
        def __init__(self, adj_entry, rank=None):
            self.adj_entry = adj_entry
            self._rank = rank or 0

        @property
        def rank(self):
            return self._rank

        @rank.setter
        def rank(self, val):
            self._rank = max(val, self._rank)

        def __repr__(self):
            return '_OpRank(\'%s\' | %d)' % (self.adj_entry.op_meta.name, self.rank)

    adj_map = SummaryGraph(model, dummy_input).adjacency_map()
    ranked_ops = {k: _OpRank(v, 0) for k, v in adj_map.items()}

    def _recurrent_ancestor(ranked_ops_dict, dest_op_name, src_op_name):
        def _is_descendant(parent_op_name, dest_op_name):
            successors_names = [op.name for op in adj_map[parent_op_name].successors]
            if dest_op_name in successors_names:
                return True
            for succ_name in successors_names:
                if _is_descendant(succ_name, dest_op_name):
                    return True
            return False

        return _is_descendant(dest_op_name, src_op_name) and \
            (0 < ranked_ops_dict[dest_op_name].rank < ranked_ops_dict[src_op_name].rank)

    def rank_op(ranked_ops_dict, op_name, rank):
        ranked_ops_dict[op_name].rank = rank
        for child_op in adj_map[op_name].successors:
            # In recurrent models: if a successor is also an ancestor - we don't increment its rank.
            if not recurrent or not _recurrent_ancestor(ranked_ops_dict, child_op.name, op_name):
                rank_op(ranked_ops_dict, child_op.name, ranked_ops_dict[op_name].rank + 1)

    roots = [k for k, v in adj_map.items() if len(v.predecessors) == 0]
    for root_op_name in roots:
        rank_op(ranked_ops, root_op_name, 0)

     # Take only the modules from the original model
    # module_dict = dict(model.named_modules())
    # Neta
    ret = sorted([k for k in ranked_ops.keys()],
                 key=lambda k: ranked_ops[k].rank)

    # Check that only the actual roots have a rank of 0
    assert {k for k in ret if ranked_ops[k].rank == 0} <= set(roots)
    return ret


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

    original_macs, _ = net_wrapper.get_resources_requirements()
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
        fname = "{}_top1_{:2f}__density_{:2f}_sampled".format(net_wrapper.arch, top1, total_macs/original_macs)
        services.save_checkpoint_fn(epoch=0, model=net_wrapper.model,
                                    scheduler=scheduler, name=fname)
        del scheduler

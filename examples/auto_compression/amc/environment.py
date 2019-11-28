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
from utils.features_collector import collect_intermediate_featuremap_samples
from utils.ac_loggers import AMCStatsLogger, FineTuneStatsLogger


msglogger = logging.getLogger("examples.auto_compression.amc")
Observation = namedtuple('Observation', ['t', 'type', 'n', 'c',  'h', 'w', 'stride', 'k', 'MACs',
                                         'Weights', 'reduced', 'rest', 'prev_a'])
ObservationLen = len(Observation._fields)


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


from utils.net_wrapper import NetworkWrapper


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
                #assert self.removed_macs_pct >= 1 - self.amc_cfg.target_density - 0.002 # 0.01
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

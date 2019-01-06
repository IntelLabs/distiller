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

$ time python3  compress_classifier.py --arch=plain20_cifar ../../../data.cifar --adc --resume=checkpoint.plain20_cifar.pth.tar --name="AMC-plain20" --lr=0.1

Coach installation:
===================
After creating the virtual environment and installing Distiller's Python package dependencies, go ahead and
setup Coach per: https://github.com/NervanaSystems/coach#installation.

Make sure that you install Coach's package dependencies into the same virtual environment that already contains
Distiller's dependency packages.  You do this by ensuring that Distiller's virtual environment is the active environment
when you install Coach.
*NOTE: you may need to update TensorFlow to the expected version:
    $ pip3 install tensorflow==1.9.0

Finally, if you are running Coach in a development environment, you need to tell the Python runtime where to find
the Coach code:
$ export PYTHONPATH=<path-to-coach-code>


Spinningup installation:
========================

Spinup require that we use exactly Python 3.6 so if you are not using this Python version see the instructions here:
    http://ubuntuhandbook.org/index.php/2017/07/install-python-3-6-1-in-ubuntu-16-04-lts/
    $ sudo update-alternatives --config python3

For Python 3.6 you may also need to install a new virtual-env:
    $ sudo apt-get install python3.6-venv

Then create and activate your venv, and populate it with the Distiller packages:
    $ python3 -m venv  distiller_env_python3.6
    $ source distiller_env_python3.6/bin/activate
    $ pip3 install -r requirements.txt

You want to install Spinup into this venv.  First clone Spinup and then install it into your venv:
    $ cd <spinningup-repo>
    $ sudo apt-get install python3.6-dev
    $ pip3 install -e .


https://spinningup.openai.com/en/latest/user/installation.html?highlight=license

"""
import math
import os
import copy
import logging
import numpy as np
import torch
import csv
import gym
from gym import spaces
import distiller
from apputils import SummaryGraph
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from distiller import normalize_module_name


# Choose which RL library to use: Coach from Intel AI Lab, or Spinup from OpenAI
#RLLIB = "spinup"
RLLIB = "coach"


msglogger = logging.getLogger()
Observation = namedtuple('Observation', ['n', 'c', 'h', 'w', 'stride', 'k', 'MACs', 'reduced', 'rest', 'prev_a'])
ALMOST_ONE = 0.9999
#RL_AGENT = "DDPG"
#RL_AGENT = "PPO"
RL_AGENT = "ClippedPPO"


def is_using_continuous_action_space(agent):
    return agent in ("DDPG", "ClippedPPO")


def add_automl_args(argparser, arch_choices=None, enable_pretrained=False):
    """
    Helper function to make it easier to add command-line arguments for AMC to any application.

    Arguments:
        argparser (argparse.ArgumentParser): Existing parser to which to add the arguments
    """
    group = argparser.add_argument_group('AutoML Compression Arguments')
    group.add_argument('--amc-protocol', choices=["mac-constrained",
                                                  "param-constrained",
                                                  "accuracy-guaranteed",
                                                  "mac-constrained-experimental"],
                       default="mac-constrained", help='Compression-policy search protocol')
    group.add_argument('--amc-ft-epochs', type=int, default=1,
                       help='The number of epochs to fine-tune each discovered network')
    group.add_argument('--amc-save-chkpts', action='store_true', default=False,
                       help='Save checkpoints of all discovered networks')
    group.add_argument('--amc-action-range',  type=float, nargs=2, default=[0.0, 0.80],
                       help='Density action range (a_min, a_max)')
    group.add_argument('--amc-heatup-epochs', type=int, default=100,
                       help='The number of epochs for heatup/exploration')
    group.add_argument('--amc-training-epochs', type=int, default=300,
                       help='The number of epochs for training/exploitation')
    group.add_argument('--amc-reward-every-step', action='store_true', default=False,
                       help='Compute the reward at every step')


    # group.add_argument('--amc-thinning', action='store_true', default=False,
    #                    help='Perform netowrk thinning after altering each layer')


if RLLIB == "spinup":
    import tensorflow as tf
    from spinup.algos.ddpg import core
    from .ddpg import ddpg

    def ddpg_spinup(env1, env2):
        from spinup.utils.run_utils import setup_logger_kwargs
        exp_name = "Test"
        seed = 0
        # The number and size of the Actor-Critic MLP hidden layers
        layers, hid = 2, 300
        logger_kwargs = setup_logger_kwargs(exp_name)  # ,  seed)

        ddpg.ddpg(env=env1, test_env=env2, actor_critic=core.mlp_actor_critic,
                  ac_kwargs=dict(hidden_sizes=[hid]*layers, output_activation=tf.sigmoid),
                  gamma=1,  # discount rate
                  seed=seed,
                  epochs=400,
                  replay_size=2000,
                  batch_size=64,
                  start_steps=env1.amc_cfg.num_heatup_epochs,
                  steps_per_epoch=800 * env1.num_layers(),  # every 50 episodes perform 10 episodes of testing
                  act_noise=0.5,
                  pi_lr=1e-4,
                  q_lr=1e-3,
                  logger_kwargs=logger_kwargs)


if RLLIB == "coach":
    from rl_coach.base_parameters import TaskParameters
    from rl_coach.core_types import EnvironmentSteps
    from rl_coach.schedules import ConstantSchedule, PieceWiseSchedule, ExponentialSchedule

    # When we import the graph_manager from the ADC_DDPG preset, we implicitly instruct
    # Coach to create and use our DistillerWrapperEnvironment environment.
    # So Distiller calls Coach, which creates the environment, trains the agent, and ends.
    if RL_AGENT == "DDPG":
        from examples.automated_deep_compression.presets.ADC_DDPG import graph_manager, agent_params
    elif RL_AGENT == "PPO":
        from examples.automated_deep_compression.presets.ADC_PPO import graph_manager, agent_params
    elif RL_AGENT == "ClippedPPO":
        from examples.automated_deep_compression.presets.ADC_ClippedPPO import graph_manager, agent_params


def log_amc_config(amc_cfg):
    try:
        msglogger.info('AMC configuration:')
        for k, v in amc_cfg.items():
            msglogger.info("\t{} : {}".format(k, v))
    except TypeError as e:
        pass


def count_conv_layer(model):
    """Count the number of Convolution layers exist in this model"""
    conv_cnt = 0
    for module in model.modules():
        if type(module) == torch.nn.Conv2d:
            conv_cnt += 1
    return conv_cnt


def mac_constrained_experimental_reward_fn(env, top1, top5, vloss, total_macs):
    """A more intuitive reward for constraining the compute and optimizing the
    accuracy under this constraint.
    """
    macs_normalized = total_macs/env.dense_model_macs
    reward = top1/100
    if macs_normalized > (env.amc_cfg.target_density+0.002):
        reward = -3 - macs_normalized
    else:
        reward += 1
    return reward


def harmonic_mean_reward_fn(env, top1, top5, vloss, total_macs):
    """This reward is based on the idea of weighted harmonic mean

    Balance compute and accuracy provided a beta value that weighs the two components.
    See: https://en.wikipedia.org/wiki/F1_score
    """
    beta = 1
    #beta = 0.75  # How much to favor accuracy
    macs_normalized = total_macs/env.dense_model_macs
    reward = (1 + beta**2) * top1/100 * macs_normalized / (beta**2 * macs_normalized + top1/100)
    return reward


experimental_reward_fn = harmonic_mean_reward_fn


def do_adc(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn):
    dataset = args.dataset
    arch = args.arch
    perform_thinning = True  # args.amc_thinning
    num_ft_epochs = args.amc_ft_epochs
    action_range = args.amc_action_range
    np.random.seed()
    conv_cnt = count_conv_layer(model)

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
            'protocol': args.amc_protocol,
            'compute_reward_every_step': args.amc_reward_every_step,
            'perform_thinning': perform_thinning,
            'num_ft_epochs': num_ft_epochs,
            'action_range': action_range,
            'conv_cnt': conv_cnt})
    if args.amc_protocol == "accuracy-guaranteed":
        amc_cfg.target_density = None
        amc_cfg.reward_fn = lambda env, top1, top5, vloss, total_macs: -(1-top1/100) * math.log(total_macs)
        amc_cfg.action_constrain_fn = None
    elif args.amc_protocol == "mac-constrained":
        amc_cfg.target_density = 0.5
        amc_cfg.reward_fn = lambda env, top1, top5, vloss, total_macs: top1/100
        amc_cfg.action_constrain_fn = DistillerWrapperEnvironment.get_action
    elif args.amc_protocol == "mac-constrained-experimental":
        amc_cfg.target_density = 0.5
        amc_cfg.reward_fn = experimental_reward_fn
        amc_cfg.action_constrain_fn = None
    else:
        raise ValueError("{} is not supported currently".format(args.amc_protocol))

    steps_per_episode = conv_cnt
    amc_cfg.heatup_noise = 0.5
    amc_cfg.initial_training_noise = 0.5
    amc_cfg.training_noise_decay = 0.996  # 0.998
    amc_cfg.num_heatup_epochs = args.amc_heatup_epochs
    amc_cfg.num_training_epochs = args.amc_training_epochs
    training_noise_duration = amc_cfg.num_training_epochs * steps_per_episode
    heatup_duration = amc_cfg.num_heatup_epochs * steps_per_episode

    if RLLIB == "spinup":
        msglogger.info("AMC: Using spinup")
        env1 = DistillerWrapperEnvironment(model, app_args, amc_cfg, services)
        env2 = DistillerWrapperEnvironment(model, app_args, amc_cfg, services)
        ddpg_spinup(env1, env2)
    else:
        msglogger.info("AMC: Using coach")

        # These parameters are passed to the Distiller environment
        graph_manager.env_params.additional_simulator_parameters = {'model': model,
                                                                    'app_args': app_args,
                                                                    'amc_cfg': amc_cfg,
                                                                    'services': services}

        if RL_AGENT == "DDPG":
            agent_params.exploration.noise_percentage_schedule = PieceWiseSchedule([
                (ConstantSchedule(amc_cfg.heatup_noise), EnvironmentSteps(heatup_duration)),
                (ExponentialSchedule(amc_cfg.initial_training_noise, 0, amc_cfg.training_noise_decay),
                 EnvironmentSteps(training_noise_duration))])

        # agent_params.exploration.noise_percentage_schedule = ConstantSchedule(0)

        coach_logs_dir = os.path.join(msglogger.logdir, 'coach')
        os.mkdir(coach_logs_dir)
        task_parameters = TaskParameters(experiment_path=coach_logs_dir)
        graph_manager.create_graph(task_parameters)
        graph_manager.improve()


class DistillerWrapperEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model, app_args, amc_cfg, services):
        self.pylogger = distiller.data_loggers.PythonLogger(msglogger)
        self.tflogger = distiller.data_loggers.TensorBoardLogger(msglogger.logdir)
        USING_SINGLE_GPU = False
        if USING_SINGLE_GPU:
            model = distiller.make_non_parallel_copy(model)
        self.orig_model = model

        self.app_args = app_args
        self.amc_cfg = amc_cfg
        self.services = services
        self.conv_layers, self.dense_model_macs, self.dense_model_size = collect_conv_details(
            model, self.app_args.dataset,
            self.amc_cfg.perform_thinning)

        self.reset(init_only=True)
        msglogger.info("Model %s has %d Convolution layers", self.app_args.arch, len(self.conv_layers))
        msglogger.info("\tTotal MACs: %s" % distiller.pretty_int(self.dense_model_macs))
        log_amc_config(amc_cfg)

        self.episode = 0
        self.best_reward = -1000
        self.action_low = amc_cfg.action_range[0]
        self.action_high = amc_cfg.action_range[1]
        # Gym spaces documentation: https://gym.openai.com/docs/
        if is_using_continuous_action_space(RL_AGENT):
            self.action_space = spaces.Box(self.action_low, self.action_high, shape=(1,))
            self.action_space.default_action = self.action_low
        else:
            self.action_space = spaces.Discrete(10)
        self.STATE_EMBEDDING_LEN = len(Observation._fields)
        #self.observation_space = spaces.Box(0, float("inf"), shape=(self.STATE_EMBEDDING_LEN+self.num_layers(),))
        self.observation_space = spaces.Box(0, float("inf"), shape=(self.STATE_EMBEDDING_LEN+1,))
        self.create_network_record_file()

    def reset(self, init_only=False):
        """Reset the environment.
        This is invoked by the Agent.
        """
        msglogger.info("Resetting the environment (init_only={})".format(init_only))
        self.current_layer_id = 0
        self.prev_action = 0
        self.model = copy.deepcopy(self.orig_model)
        self.zeros_mask_dict = distiller.create_model_masks_dict(self.model)
        # self._remaining_macs = self.dense_model_macs
        self._removed_macs = 0
        self.action_history = []
        self.agent_action_history = []
        if init_only:
            return
        initial_observation = self.get_obs()
        return initial_observation

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
            msglogger.info("Starting a new episode")
            msglogger.info("+" + "-" * 50 + "+")

        msglogger.info("Render Environment: current_layer_id=%d" % self.current_layer_id)
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

    def get_action(self, pruning_action):
        """Compute a resource-constrained action"""
        reduced = self._removed_macs
        rest = self.rest_macs_raw() * self.action_high
        target_reduction = (1 - self.amc_cfg.target_density) * self.dense_model_macs

        duty = target_reduction - (reduced + rest)
        flops = self.get_layer_macs(self.current_layer())
        assert flops > 0
        pruning_action_final = min(self.action_high, max(pruning_action, duty/flops))
        if pruning_action_final != pruning_action:
            msglogger.info("action ********** pruning_action={}==>pruning_action_final={:.2f}: reduced={:.2f} rest={:.2f} target={:.2f} duty={:.2f} flops={:.2f}".
                           format(pruning_action, pruning_action_final, reduced/self.dense_model_macs,
                                  rest/self.dense_model_macs, 1-self.amc_cfg.target_density,
                                  duty/self.dense_model_macs,
                                  flops/self.dense_model_macs))
        return pruning_action_final

    def step(self, pruning_action):
        """Take a step, given an action.

        The action represents the desired sparsity.
        This function is invoked by the Agent.
        """
        msglogger.info("env.step - current_layer_id={}  episode={}".format(self.current_layer_id, self.episode))
        msglogger.info("\tAgent pruning_action={}".format(pruning_action))
        if is_using_continuous_action_space(RL_AGENT):
            pruning_action = np.clip(pruning_action[0], self.action_low, self.action_high)
        else:
            # Divide the action space into 10 discrete levels (0%, 10%, 20%,....90% sparsity)
            pruning_action = pruning_action / 10
        msglogger.info("\tAgent clipped pruning_action={}".format(pruning_action))
        self.agent_action_history.append(pruning_action)
        if self.amc_cfg.protocol == "mac-constrained":
            pruning_action = self.get_action(pruning_action)
            msglogger.info("Constrained pruning_action={}".format(pruning_action))

        _, total_macs_before, _ = collect_conv_details(self.model, self.app_args.dataset, self.amc_cfg.perform_thinning)
        layer_macs = self.get_layer_macs(self.current_layer())
        msglogger.info("\tlayer_macs={:.2f}".format(layer_macs / self.dense_model_macs))
        msglogger.info("\tremoved_macs={:.2f}".format(self.removed_macs()))
        msglogger.info("\trest_macs={:.2f}".format(self.rest_macs()))

        if pruning_action > 0:
            pruning_action = self.__remove_structures(self.current_layer_id,
                                                      fraction_to_prune=pruning_action,
                                                      prune_what="filters")
        else:
            pruning_action = 0

        self.action_history.append(pruning_action)
        _, total_macs_after, _ = collect_conv_details(self.model, self.app_args.dataset, self.amc_cfg.perform_thinning)
        layer_macs_after_action = self.get_layer_macs(self.current_layer())

        # Update the various counters after taking the step
        self.current_layer_id += 1
        self._removed_macs += (total_macs_before - total_macs_after)

        msglogger.info("actual_action={}".format(pruning_action))
        msglogger.info("layer_macs={} layer_macs_after_action={} removed now={}".format(layer_macs,
                                                                                        layer_macs_after_action,
                                                                                        (layer_macs - layer_macs_after_action)))
        msglogger.info("self._removed_macs={}".format(self._removed_macs))
        assert (layer_macs_after_action / layer_macs) == (1 - pruning_action)

        stats = ('Peformance/Validation/',
                 OrderedDict([('requested_action', pruning_action)]))

        distiller.log_training_progress(stats, None,
                                        self.episode, steps_completed=self.current_layer_id,
                                        total_steps=self.amc_cfg.conv_cnt, log_freq=1, loggers=[self.tflogger])

        if self.episode_is_done():
            msglogger.info("Episode is ending")
            observation = self.get_final_obs()
            reward, top1, total_macs, total_nnz = self.compute_reward()
            self.episode += 1
            normalized_macs = total_macs / self.dense_model_macs * 100
            normalized_nnz = total_nnz / self.dense_model_size * 100
            self.record_network_details(top1, reward, total_macs, normalized_macs,
                                        normalized_nnz, self.action_history, self.agent_action_history)
        else:
            observation = self.get_obs()
            reward = 0
            if self.amc_cfg.compute_reward_every_step:
                reward, top1, total_macs, total_nnz = self.compute_reward(False)

        self.prev_action = pruning_action
        info = {}
        return observation, reward, self.episode_is_done(), info

    def one_hot(self, n, r):
        """Produce a one-hot representation of the layer id"""
        #return [1 if i == n else 0 for i in range(r)]
        return [n]

    def get_obs(self):
        """Produce a state embedding (i.e. an observation)"""

        current_layer_macs = self.get_layer_macs(self.current_layer())
        current_layer_macs_pct = current_layer_macs/self.dense_model_macs
        current_layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, current_layer.name)
        obs = [#current_layer.t,
                conv_module.out_channels,
                conv_module.in_channels,
                current_layer.ifm_h,
                current_layer.ifm_w,
                current_layer.stride[0],
                current_layer.k,
                current_layer_macs_pct*100,
                self.removed_macs()*100,
                self.rest_macs()*100,
                self.prev_action*100]
        onehot_id = self.one_hot(self.current_layer_id, self.num_layers())
        msglogger.info("obs={} {}".format(onehot_id, Observation._make(obs)))
        obs = np.array(onehot_id + obs)
        assert (self.removed_macs() + current_layer_macs_pct + self.rest_macs()) <= 1
        return obs

    def get_final_obs(self):
        """Return the final state embedding (observation)
        The final state is reached after we traverse all of the Convolution layers.
        """
        obs = [#-1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                self.removed_macs()*100,
                self.rest_macs()*100,
                self.prev_action*100]
        onehot_id = self.one_hot(self.num_layers(), self.num_layers())
        msglogger.info("obs={} {}".format(onehot_id, Observation._make(obs)))
        obs = np.array(onehot_id + obs)
        return obs

    def rest_macs_raw(self):
        """Return the number of remaining MACs in the layers following the current layer"""
        rest = 0
        for layer_id in range(self.current_layer_id, self.num_layers()):
            rest += self.get_layer_macs(self.get_layer(layer_id + 1))
        return rest

    def rest_macs(self):
        return self.rest_macs_raw() / self.dense_model_macs

    def get_layer_macs(self, layer):
        """Return the number of MACs required to compute <layer>'s Convolution"""
        if layer is None:
            return 0

        conv_module = distiller.model_find_module(self.model, layer.name)
        # MACs = volume(OFM) * (#IFM * K^2)
        dense_macs = (conv_module.out_channels * layer.ofm_h * layer.ofm_w) * (conv_module.in_channels * layer.k**2)
        if self.amc_cfg.perform_thinning:
            return dense_macs

        # If we didn't physically remove structures, we need to use the structural sparsity to compute MACs
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)
        # return dense_macs * distiller.density_ch(conv_p)  # Channel pruning
        return dense_macs * distiller.density_3D(conv_p)  # Filter pruning

    def __remove_structures(self, idx, fraction_to_prune, prune_what="channels"):
        """Physically remove channels and corresponding filters from the model

        Returns the compute-sparsity of the layer with index 'idx'
        """
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
        macs_before = self.get_layer_macs(layer)
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)

        msglogger.info("ADC: removing %.1f%% %s from %s" % (fraction_to_prune*100, prune_what, conv_pname))

        if prune_what == "channels":
            calculate_sparsity = distiller.sparsity_ch
            remove_structures = distiller.remove_channels
            group_type = "Channels"
        elif prune_what == "filters":
            calculate_sparsity = distiller.sparsity_3D
            group_type = "Filters"
            remove_structures = distiller.remove_filters
        else:
            raise ValueError("unsupported structure {}".format(prune_what))
        # Create a channel-ranking pruner
        pruner = distiller.pruning.L1RankedStructureParameterPruner("adc_pruner", group_type,
                                                                    fraction_to_prune, conv_pname)
        pruner.set_param_mask(conv_p, conv_pname, self.zeros_mask_dict, meta=None)

        if (self.zeros_mask_dict[conv_pname].mask is None or
            calculate_sparsity(self.zeros_mask_dict[conv_pname].mask) == 0):
            msglogger.info("__remove_structures: aborting because there are no channels to prune")
            return 0

        # Use the mask to prune
        self.zeros_mask_dict[conv_pname].apply_mask(conv_p)

        if self.amc_cfg.perform_thinning:
            remove_structures(self.model, self.zeros_mask_dict, self.app_args.arch, self.app_args.dataset, optimizer=None)
            conv_p = distiller.model_find_param(self.model, conv_pname)
            #return distiller.volume(conv_p) / layer.weights_vol
            return 1 - (self.get_layer_macs(layer) / macs_before)
        #actual_sparsity = calculate_sparsity(conv_p)
        #return actual_sparsity
        # This is a hack
        assert False, "We should not get to this point"

    def compute_reward(self, log_stats=True):
        """Compute the reward"""
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

        if self.amc_cfg.perform_thinning:
            _, total_macs, total_nnz = collect_conv_details(self.model, self.app_args.dataset, self.amc_cfg.perform_thinning)
            compression = distiller.model_numel(self.model, param_dims=[4]) / self.dense_model_size
        else:
            _, total_macs, total_nnz = collect_conv_details(self.model, self.app_args.dataset, self.amc_cfg.perform_thinning)
            compression = 1 - distiller.model_sparsity(self.model)/100
            # What a hack!
            total_nnz *= compression

        # Train for zero or more epochs
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.app_args.optimizer_data['lr'],
                                    momentum=self.app_args.optimizer_data['momentum'],
                                    weight_decay=self.app_args.optimizer_data['weight_decay'])
        for _ in range(self.amc_cfg.num_ft_epochs):
            # Fine-tune the model
            self.services.train_fn(model=self.model, compression_scheduler=self.create_scheduler(),
                                   optimizer=optimizer,
                                   epoch=self.episode)
        # Validate
        top1, top5, vloss = self.services.validate_fn(model=self.model, epoch=self.episode)
        reward = self.amc_cfg.reward_fn(self, top1, top5, vloss, total_macs)

        if log_stats:
            macs_normalized = total_macs/self.dense_model_macs
            msglogger.info("Total parameters left: %.2f%%" % (compression*100))
            msglogger.info("Total compute left: %.2f%%" % (total_macs/self.dense_model_macs*100))

            stats = ('Peformance/Validation/',
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

    def create_network_record_file(self):
        """Create the CSV file and write the column names"""
        fields = ['episode', 'top1', 'reward', 'total_macs', 'normalized_macs',
                  'normalized_nnz', 'ckpt_name', 'action_history', 'agent_action_history']
        with open(os.path.join(msglogger.logdir, 'amc.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def record_network_details(self, top1, reward, total_macs, normalized_macs,
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
        with open(os.path.join(msglogger.logdir, 'amc.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def create_scheduler(self):
        scheduler = distiller.CompressionScheduler(self.model)
        masks = {param_name: masker.mask for param_name, masker in self.zeros_mask_dict.items()}
        scheduler.load_state_dict(state={'masks_dict': masks})
        return scheduler

    def save_checkpoint(self, is_best=False):
        """Save the learned-model checkpoint"""
        scheduler = self.create_scheduler()
        episode = str(self.episode).zfill(3)
        if is_best:
            fname = "BEST_adc_episode_{}".format(episode)
        else:
            fname = "adc_episode_{}".format(episode)

        self.services.save_checkpoint_fn(epoch=self.episode, model=self.model,
                                         scheduler=scheduler, name=fname)
        return fname


def get_dummy_input(dataset):
    if dataset == 'imagenet':
        dummy_input = torch.randn(1, 3, 224, 224)
    elif dataset == 'cifar10':
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        raise ValueError("dataset %s is not supported" % dataset)
    return dummy_input


def collect_conv_details(model, dataset, perform_thinning):
    dummy_input = get_dummy_input(dataset)
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
            conv_layers[len(conv_layers)] = conv
    return conv_layers, total_macs, total_params

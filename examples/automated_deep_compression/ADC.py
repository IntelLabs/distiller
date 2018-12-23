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

After creating the virtual environment and installing Distiller's Python package dependencies, go ahead and
setup Coach per: https://github.com/NervanaSystems/coach#installation.

Make sure that you install Coach's package dependencies into the same virtual environment that already contains
Distiller's dependency packages.  You do this by ensuring that Distiller's virtual environment is the active environment
when you install Coach.
*NOTE: you may need to update TensorFlow to the expected version:
    $ pip3 install tensorflow==1.9.0

Finally, if you are running Coach in a development environment, you need to tell the Python runtime where to find the Coach code:
$ export PYTHONPATH=<path-to-coach-code>

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
from rl_coach import logger
from rl_coach.base_parameters import TaskParameters

# When we import the graph_manager from the ADC_DDPG preset, we implicitly instruct
# Coach to create and use our DistillerWrapperEnvironment environment.
# So Distiller calls Coach, which creates the environment, trains the agent, and ends.
from examples.automated_deep_compression.presets.ADC_DDPG import graph_manager, agent_params
# Coach imports
from rl_coach.schedules import ConstantSchedule, PieceWiseSchedule, ExponentialSchedule
from rl_coach.core_types import EnvironmentSteps


msglogger = logging.getLogger()
Observation = namedtuple('Observation', ['t', 'n', 'c', 'h', 'w', 'stride', 'k', 'MACs', 'reduced', 'rest', 'prev_a'])
ALMOST_ONE = 0.9999


def add_automl_args(argparser, arch_choices=None, enable_pretrained=False):
    """
    Helper function to make it easier to add command-line arguments for AMC to any application.

    Arguments:
        argparser (argparse.ArgumentParser): Existing parser to which to add the arguments
    """
    group = argparser.add_argument_group('AutoML Compression Arguments')
    group.add_argument('--amc-ft-epochs', type=int, default=1,
                       help='The number of epochs to fine-tune each discovered network')
    group.add_argument('--amc-save-chkpts', action='store_true', default=False,
                       help='Save checkpoints of all discovered networks')
    group.add_argument('--amc-action-range',  type=float, nargs=2, default=[0.0, 0.80],
                       help='Density action range (a_min, a_max)')
    group.add_argument('--amc-thinning', action='store_true', default=False,
                       help='Perform netowrk thinning after altering each layer')


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


def do_adc(model, args, optimizer_data, validate_fn, save_checkpoint_fn, train_fn):
    dataset = args.dataset
    arch = args.arch
    perform_thinning = args.amc_thinning
    num_training_epochs = args.amc_ft_epochs
    action_range = args.amc_action_range
    np.random.seed()
    #task_parameters = TaskParameters(experiment_path=logger.get_experiment_path('adc'))
    coach_logs_dir = os.path.join(msglogger.logdir, 'coach')
    os.mkdir(coach_logs_dir)
    task_parameters = TaskParameters(experiment_path=coach_logs_dir)
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
    if True:
        amc_cfg = distiller.utils.MutableNamedTuple({
                'perform_thinning': perform_thinning,
                'num_training_epochs': num_training_epochs,
                'action_range': action_range,
                'normalize_obs': False,
                'desired_reduction': None,
                'reward_fn': lambda top1, top5, vloss, total_macs: -1 * (1-top1/100) * math.log(total_macs),
                'conv_cnt': conv_cnt,
                'max_reward': -1000})
    else:
        amc_cfg = distiller.utils.MutableNamedTuple({
                'perform_thinning': perform_thinning,
                'num_training_epochs': num_training_epochs,
                'action_range': action_range,
                'normalize_obs': False,
                'desired_reduction': 0.8,
                'reward_fn': lambda top1, top5, vloss, total_macs: top1/100,
                'conv_cnt': conv_cnt,
                'max_reward': -1000})

    # These parameters are passed to the Distiller environment
    graph_manager.env_params.additional_simulator_parameters = {'model': model,
                                                                'app_args': app_args,
                                                                'amc_cfg': amc_cfg,
                                                                'services': services}
    steps_per_episode = conv_cnt
    amc_cfg.exploration_noise = 0.5
    amc_cfg.exploration_duration = 100 * steps_per_episode
    amc_cfg.exploitation_decay = 0.996
    amc_cfg.exploitation_duration = 300 * steps_per_episode

    agent_params.exploration.noise_percentage_schedule = PieceWiseSchedule([
        (ConstantSchedule(amc_cfg.exploration_noise), EnvironmentSteps(amc_cfg.exploration_duration)),
        (ExponentialSchedule(amc_cfg.exploration_noise, 0, amc_cfg.exploitation_decay),
         EnvironmentSteps(amc_cfg.exploitation_duration))])

    graph_manager.create_graph(task_parameters)
    graph_manager.improve()


class DistillerWrapperEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, model, app_args, amc_cfg, services):
        self.pylogger = distiller.data_loggers.PythonLogger(msglogger)
        self.tflogger = distiller.data_loggers.TensorBoardLogger(msglogger.logdir)
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

        self.debug_stats = {'episode': 0}
        self.action_low = amc_cfg.action_range[0]
        self.action_high = amc_cfg.action_range[1]
        # Gym spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Box(self.action_low, self.action_high, shape=(1,))
        self.action_space.default_action = self.action_low

        self.STATE_EMBEDDING_LEN = len(Observation._fields)
        self.observation_space = spaces.Box(0, float("inf"), shape=(self.STATE_EMBEDDING_LEN,))
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
        self._remaining_macs = self.dense_model_macs
        self._removed_macs = 0
        self.action_history = []
        if init_only:
            return

        layer_macs = self.get_layer_macs(self.current_layer())
        initial_observation = self._get_obs(layer_macs)
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

    def remaining_macs(self):
        """Return the amount of MACs remaining in the model's unprocessed Convolution layers.
        This is normalized to the range 0..1
        """
        return self._remaining_macs / self.dense_model_macs

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

        msglogger.info("Environment: current_layer_id=%d" % self.current_layer_id)
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

    def get_action(self, a):
        reduced = self._removed_macs
        rest = self._remaining_macs * self.action_high
        target = self.amc_cfg.desired_reduction * self.dense_model_macs

        duty = target - (reduced + rest)
        flops = self.get_layer_macs(self.current_layer())
        a_final = max(a, duty/flops)
        a_final = min(a_final, self.action_high)

        if a_final != a:
            msglogger.info("action ********** a={}==>a_final={:.2f}: reduced={:.2f} remaining={:.2f} rest={:.2f} target={:.2f} duty={:.2f} flops={:.2f}".
                           format(a, a_final, reduced/self.dense_model_macs,
                                  self.remaining_macs(),
                                  rest/self.dense_model_macs, self.amc_cfg.desired_reduction,
                                  duty/self.dense_model_macs, flops/self.dense_model_macs))
        return a_final

    def create_network_record_file(self):
        """Create the CSV file and write the column names"""
        fields = ['top1', 'reward', 'total_macs', 'normalized_macs', 'total_nnz', 'ckpt_name', 'action_history']
        with open(os.path.join(msglogger.logdir, 'amc.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def record_network_details(self, top1, reward, total_macs, normalized_macs, total_nnz, action_history):
        """Write the details of one network to a CSV file and create a checkpoint file"""
        if reward > self.amc_cfg.max_reward:
            self.amc_cfg.max_reward = reward
            ckpt_name = self.save_checkpoint(is_best=True)
            msglogger.info("Best reward={}  episode={}  top1={}".format(reward, self.debug_stats['episode'], top1))
        else:
            ckpt_name = self.save_checkpoint(is_best=False)

        fields = [top1, reward, total_macs, normalized_macs, total_nnz, ckpt_name, action_history]
        with open(os.path.join(msglogger.logdir, 'amc.csv'), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def create_scheduler(self):
        # if self.amc_cfg.perform_thinning:
        #     # We don't want to apply the masks if we performed thinning
        #     return None
        scheduler = distiller.CompressionScheduler(self.model)
        masks = {param_name: masker.mask for param_name, masker in self.zeros_mask_dict.items()}
        scheduler.load_state_dict(state={'masks_dict': masks})
        return scheduler

    def save_checkpoint(self, is_best=False):
        # Save the learned-model checkpoint
        scheduler = self.create_scheduler()
        episode = self.debug_stats['episode']
        episode = str(episode).zfill(3)
        if is_best:
            fname = "BEST_adc_episode_{}".format(episode)
        else:
            fname = "adc_episode_{}".format(episode)

        self.services.save_checkpoint_fn(epoch=self.debug_stats['episode'], model=self.model,
                                         scheduler=scheduler, name=fname)
        return fname

    def step(self, action):
        """Take a step, given an action.

        The action represents the desired density.
        This function is invoked by the Agent.
        """
        msglogger.info("env.step - current_layer_id={}\n\tAgent action={}".format(self.current_layer_id, action))
        assert action == 0 or (action >= self.action_low-0.001 and action <= self.action_high+0.001)
        if self.amc_cfg.desired_reduction is not None:
            action = self.get_action(action)
            msglogger.info("Constrained action={} (leave)".format(action))
        action = 1 - action
        layer_macs = self.get_layer_macs(self.current_layer())
        msglogger.info("\tlayer_macs={} removed_macs={:.2f} remaining_macs={:.2f}".format(layer_macs,
                                                                                          self.removed_macs(),
                                                                                          self.remaining_macs()))
        if action > 0 and self.current_layer_id >= 0:
            actual_action = self.__remove_structures(self.current_layer_id,
                                                     fraction_to_prune=action,
                                                     prune_what="filters")
        else:
            actual_action = 0
        self.action_history.append(actual_action)
        layer_macs_after_action = self.get_layer_macs(self.current_layer())

        # Update the various counters after taking the step
        self.current_layer_id += 1
        self._removed_macs += (layer_macs - layer_macs_after_action)
        self._remaining_macs -= layer_macs # next_layer_macs

        stats = ('Peformance/Validation/',
                 OrderedDict([('requested_action', action),
                              ('actual_action', 1-actual_action)]))
        distiller.log_training_progress(stats, None, self.debug_stats['episode'], steps_completed=self.current_layer_id,
                                        total_steps=self.amc_cfg.conv_cnt, log_freq=1, loggers=[self.tflogger])

        if self.episode_is_done():
            observation = self.get_final_obs()
            reward, top1, total_macs, total_nnz = self.compute_reward()
            self.debug_stats['episode'] += 1
            normalized_macs = total_macs/self.dense_model_macs * 100
            self.record_network_details(top1, reward, total_macs, normalized_macs, total_nnz, self.action_history)
        else:
            observation = self._get_obs(layer_macs)
            reward = 0

        self.prev_action = 1 - action
        msglogger.info("###################### self.prev_action={}".format(self.prev_action))
        info = {}
        return observation, reward, self.episode_is_done(), info

    def _get_obs4(self, current_layer_macs, current_layer, conv_module):
        """Produce a state embedding (i.e. an observation)"""

        if self.amc_cfg.normalize_obs:
            # TODO: this should be normalized per the real values of the dense layer!!!!
            obs = np.array([current_layer.t,
                            conv_module.out_channels / 512,
                            conv_module.in_channels / 512,
                            current_layer.ifm_h / 32,
                            current_layer.ifm_w / 32,
                            current_layer.stride[0] / 2,
                            current_layer.k / 3,
                            current_layer_macs / self.dense_model_macs,
                            self.removed_macs(), self.remaining_macs(), self.prev_action])
        else:
            obs = np.array([current_layer.t,
                            conv_module.out_channels, conv_module.in_channels,
                            current_layer.ifm_h, current_layer.ifm_w, current_layer.stride[0], current_layer.k,
                            current_layer_macs/self.dense_model_macs,
                            self.removed_macs(), self.remaining_macs(), self.prev_action])

        msglogger.info("obs={}".format(Observation._make(obs)))

        assert len(obs) == self.STATE_EMBEDDING_LEN
        print("macs(%)={}\nremoved macs={}\nremaining macs={}".format(current_layer_macs / self.dense_model_macs,
                                                                      self.removed_macs(),
                                                                      self.remaining_macs()))
        #assert (current_layer_macs / self.dense_model_macs + self.removed_macs() + self.remaining_macs()) <= 1
        assert (self.removed_macs() + self.remaining_macs()) <= 1
        return obs

    def _get_obs(self, current_layer_macs):
        current_layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, current_layer.name)
        return self._get_obs4(current_layer_macs, current_layer, conv_module)

    def get_final_obs(self):
        """Return the final state embedding (observation)
        The final state is reached after we traverse all of the Convolution layers.
        """
        obs = np.array([-1, 0, 0,
                         0, 0, 0, 0,
                         0, self.removed_macs(), 0, 1 - self.prev_action])

        assert len(obs) == self.STATE_EMBEDDING_LEN
        return obs

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
            return distiller.volume(conv_p) / layer.weights_vol
        actual_sparsity = calculate_sparsity(conv_p)
        return actual_sparsity

    def compute_reward(self):
        """The ADC paper defines reward = -Error"""
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

        if self.amc_cfg.perform_thinning:
            _, total_macs, total_nnz = collect_conv_details(self.model, self.app_args.dataset, self.amc_cfg.perform_thinning)
            compression = distiller.model_numel(self.model, param_dims=[4]) / self.dense_model_size
        else:
            _, total_macs, total_nnz = collect_conv_details(self.model, self.app_args.dataset, self.amc_cfg.perform_thinning)
            compression = 1 - distiller.model_sparsity(self.model)/100
            # What a hack!
            total_nnz *= compression

        msglogger.info("Total parameters left: %.2f%%" % (compression*100))
        msglogger.info("Total compute left: %.2f%%" % (total_macs/self.dense_model_macs*100))
        # Train for zero or more epochs
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.app_args.optimizer_data['lr'],
                                    momentum=self.app_args.optimizer_data['momentum'],
                                    weight_decay=self.app_args.optimizer_data['weight_decay'])
        for _ in range(self.amc_cfg.num_training_epochs):
            self.services.train_fn(model=self.model, compression_scheduler=self.create_scheduler(),
                                   optimizer=optimizer,
                                   epoch=self.debug_stats['episode'])
        # Validate
        top1, top5, vloss = self.services.validate_fn(model=self.model, epoch=self.debug_stats['episode'])
        reward = self.amc_cfg.reward_fn(top1, top5, vloss, total_macs)

        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5),
                              ('reward', reward),
                              ('total_macs', int(total_macs)),
                              ('macs_normalized', total_macs/self.dense_model_macs*100),
                              ('log(total_macs)', math.log(total_macs)),
                              ('total_nnz', int(total_nnz))]))
        distiller.log_training_progress(stats, None, self.debug_stats['episode'], steps_completed=0, total_steps=1,
                                        log_freq=1, loggers=[self.tflogger, self.pylogger])
        return reward, top1, total_macs, total_nnz


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
    #model = distiller.make_non_parallel_copy(model)
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
            #assert distiller.density_ch(conv_p) == 1
            total_macs += conv.macs

            conv.ofm_h = g.param_shape(conv_op['outputs'][0])[2]
            conv.ofm_w = g.param_shape(conv_op['outputs'][0])[3]
            conv.ifm_h = g.param_shape(conv_op['inputs'][0])[2]
            conv.ifm_w = g.param_shape(conv_op['inputs'][0])[3]

            conv.name = name
            conv.id = id
            conv_layers[len(conv_layers)] = conv
    return conv_layers, total_macs, total_params


# import os
# import pandas as pd
# from tabulate import tabulate
# import apputils
# from models import create_model
#
#
# def summarize_experiment(experiment_dir, dataset, arch, validate_fn):
#     df = pd.DataFrame(columns=['File', 'NNZ', 'MACs', 'Top1'])
#     for file in os.listdir(experiment_dir):
#         if file.endswith(".pth.tar"):
#             cnt_macs, cnt_params, top1 = get_experiment_performance_summary(os.path.join(experiment_dir, file), dataset, arch, validate_fn)
#             df.loc[len(df.index)] = [file, cnt_params, cnt_macs, top1]
#     t = tabulate(df, headers='keys', tablefmt='psql', floatfmt=".2f")
#     print(t)
#     csv_fname = os.path.join(experiment_dir, "arch_space" + ".csv")
#     print("Saving results to: {}".format(csv_fname))
#     df.to_csv(csv_fname, sep=',')
#
#
# def get_experiment_performance_summary(chkpt_fname, dataset, arch, validate_fn):
#     model = create_model(False, dataset, arch)
#     model, compression_scheduler, start_epoch = apputils.load_checkpoint(model, chkpt_fname)
#
#     dummy_input = get_dummy_input(dataset)
#     perf_df = distiller.model_performance_summary(model, dummy_input, 1)
#     total_macs = perf_df['MACs'].sum()
#     top1, top5, vloss = validate_fn(model=model, epoch=-1)
#     return total_macs, distiller.model_numel(model), top1

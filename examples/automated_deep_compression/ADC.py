import random
import math
import copy
import logging
import numpy as np
import torch
import gym
from gym import spaces
import distiller
from apputils import SummaryGraph
from collections import OrderedDict, namedtuple
from types import SimpleNamespace
from distiller import normalize_module_name

from base_parameters import TaskParameters
from examples.automated_deep_compression.presets.ADC_DDPG import graph_manager

msglogger = logging.getLogger()
Observation = namedtuple('Observation', ['t', 'n', 'c', 'h', 'w', 'stride', 'k', 'MACs', 'reduced', 'rest', 'prev_a'])
ALMOST_ONE = 0.9999

# TODO: this is also defined in test_pruning.py
def create_model_masks(model):
    # Create the masks
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = distiller.ParameterMasker(name)
        zeros_mask_dict[name] = masker
    return zeros_mask_dict


USE_COACH = True
PERFORM_THINNING = True

def coach_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn):
    task_parameters = TaskParameters(framework_type="tensorflow",
                                     experiment_path="./experiments/test")
    extra_params = {'save_checkpoint_secs': None,
                    'render': True}
    task_parameters.__dict__.update(extra_params)

    graph_manager.env_params.additional_simulator_parameters = {
        'model': model,
        'dataset': dataset,
        'arch': arch,
        'data_loader': data_loader,
        'validate_fn': validate_fn,
        'save_checkpoint_fn': save_checkpoint_fn,
        'action_range': (0.15, 0.97)
    }
    graph_manager.create_graph(task_parameters)
    graph_manager.improve()


def random_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn):
    """Random ADC agent"""
    action_range = (0.0, 1.0)
    env = CNNEnvironment(model, dataset, arch, data_loader,
                         validate_fn, save_checkpoint_fn, action_range)

    best = [-1000, None]
    env.action_space = RandomADCActionSpace(action_range[0], action_range[1])
    for ep in range(100):
        observation = env.reset()
        action_config = []
        for t in range(100):
            #env.render(0, 0)
            msglogger.info("[episode={}:{}] observation = {}".format(ep, t, observation))
            # take a random action
            action = env.action_space.sample()
            action_config.append(action)
            observation, reward, done, info = env.step(action)
            if reward > best[0]:
                best[0] = reward
                best[1] = action_config
                msglogger.info("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                msglogger.info("New solution found: episode={} reward={} config={}".format(ep, reward, action_config))
            if done:
                msglogger.info("Episode finished after {} timesteps".format(t+1))
                break


def do_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn):
    np.random.seed()

    if USE_COACH:
        return coach_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn)
    return random_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn)


class RandomADCActionSpace(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self):
        return random.uniform(self.low, self.high)


class PredictableADCActionSpace(object):
    def __init__(self, low, high):
        #self.actions = (0.51, 0.26, 0.23, 0.09, 0.24, 0.36, 0.90, 0.97, 0.98, 0.98, 0.98, 0.98, 0)
        #self.actions = (0.51, 0.26, 0.23, 0.09, 0.24, 0.36, 0.0, 0.0, 0.50, 0.50, 0.50, 0.50, 0)
        #self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50, 0.65, 0.60, 0.00, 0.00, 0)
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.8, 0.00, 0)   # Top1 90.100000    Top5 99.420000    reward -0.113175
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.7, 0.00, 0)  # Top1 90.540000    Top5 99.360000    reward -0.124923
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.6, 0.00, 0)  # Top1 90.600000    Top5 99.340000    reward -0.128869

        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.8, 0.8, 0)   # Top1 87.600000    Top5 98.980000    reward -0.198718
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.0, 0.8, 0.8, 0.65)  # Top1 74.720000    Top5 97.700000    reward -0.447991
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0.8, 0.8, 0.8, 0.65) # Top1 39.540000    Top5 95.420000    reward -0.886748

        #self.actions = [0] * 13                                                           # Top1 90.480000    Top5 99.400000    reward -0.117374
        self.step = 0
        self.episode = 0
        self.l1 = 0
        self.l2 = 0
        self.update_action_vector()

    def update_action_vector(self):
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0, 0.8, 0.05, 0)  # Top1 89.640000    Top5 99.520000    reward -0.093653
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0, 0.8, 0.05, self.episode * 0.05)
        self.actions = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00, 0.0, 0, self.l1 * 0.05, self.l2 * 0.05, 0)

    def sample(self):
        action = self.actions[self.step]
        self.step = (self.step + 1) % len(self.actions)
        if self.step == 0:
            self.l1 = (self.l1 + 1) % 20
            if self.l1 == 0:
                self.l2 = (self.l2 + 1) % 20
            if self.l2 == 19:
                print("Done - exiting")
                exit()
            self.update_action_vector()
        return action


def collect_conv_details(model, dataset):
    if dataset == 'imagenet':
        dummy_input = torch.randn(1, 3, 224, 224)
    elif dataset == 'cifar10':
        dummy_input = torch.randn(1, 3, 32, 32)
    else:
        raise ValueError("dataset %s is not supported" % dataset)

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

            total_nnz += conv_op['attrs']['weights_vol']
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


class CNNEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
     #STATE_EMBEDDING_LEN = len(Observation._fields) + 12
    STATE_EMBEDDING_LEN = len(Observation._fields)

    def __init__(self, model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn, action_range):
        self.pylogger = distiller.data_loggers.PythonLogger(msglogger)
        self.tflogger = distiller.data_loggers.TensorBoardLogger(msglogger.logdir)

        self.dataset = dataset
        self.arch = arch
        self.data_loader = data_loader
        self.validate_fn = validate_fn
        self.save_checkpoint_fn = save_checkpoint_fn
        self.orig_model = model

        self.max_reward = -1000

        self.conv_layers, self.dense_model_macs, self.dense_model_size = collect_conv_details(model, dataset)
        self.reset(init_only=True)
        msglogger.info("Model %s has %d Convolution layers", arch, len(self.conv_layers))
        msglogger.info("\tTotal MACs: %s" % distiller.pretty_int(self.dense_model_macs))

        self.debug_stats = {'episode': 0}
        self.action_low = action_range[0]
        self.action_high = action_range[1]
        # Gym
        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Box(self.action_low, self.action_high, shape=(1,))
        self.action_space.default_action = self.action_low
        self.observation_space = spaces.Box(0, float("inf"), shape=(self.STATE_EMBEDDING_LEN,))

    def reset(self, init_only=False):
        """Reset the environment.
        This is invoked by the Agent.
        """
        msglogger.info("Resetting the environment (init_only={})".format(init_only))
        self.current_layer_id = -1
        self.prev_action = 0
        self.model = copy.deepcopy(self.orig_model)
        self.zeros_mask_dict = create_model_masks(self.model)
        self._remaining_macs = self.dense_model_macs
        self._removed_macs = 0
        if init_only:
            return

        #layer_macs = self.get_macs(self.current_layer())
        #return self._get_obs(layer_macs)
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
        #desired_reduction = 0.5e8
        desired_reduction = 2.3e8
        #desired_reduction = 1.5e8
        #if self.current_layer_id == 0:
        #    reduced = 0
        reduced = self._removed_macs
        rest = self._remaining_macs

        duty = desired_reduction - (reduced + rest)
        flops = self.get_macs(self.current_layer())
        msglogger.info("action ********** a={}  duty={} desired_reduction={} reduced={}  rest={}  flops={}".format(a, duty, desired_reduction, reduced, rest, flops))

        if duty > 0:
            #duty = 0.9*desired_reduction - (reduced + rest)
            duty = desired_reduction - (reduced + rest)
            msglogger.info("action ********** duty/flops={}".format(duty / flops))
            msglogger.info("action ********** 1 - duty/flops={}".format(1 - duty / flops))
            #a = max(1-self.action_low, min(a, 1 - duty/flops))

            ##
            ##  Consider using max=0 for R = error * macs
            ##           using max= self.action_low for FLOP-limited?  Add noise so it doesn't get stuck in one place?
            ##
            #a = max(self.action_low, min(a, 1 - duty/flops))
            a = max(0, min(a, 1 - duty/flops))
        return a

    # def get_action(self, a):
    #     desired_reduction = 1.5e8
    #     #if self.current_layer_id == 0:
    #     #    reduced = 0
    #     reduced = self._removed_macs
    #     rest = self._remaining_macs
    #
    #     duty = desired_reduction - reduced - rest
    #     flops = self.get_macs(self.current_layer())
    #     msglogger.info("action ********** a={}  duty={} desired_reduction={} reduced={}  rest={}  flops={}".format(a, duty, desired_reduction, reduced, rest, flops))
    #
    #     if duty > 0:
    #         msglogger.info("action ********** duty/flops={}".format(duty / flops))
    #         msglogger.info("action ********** 1 - duty/flops={}".format(1 - duty / flops))
    #         #a = max(1-self.action_low, min(a, 1 - duty/flops))
    #         a = max(self.action_low, min(a, 1 - duty/flops))
    #     return a

    def save_checkpoint(self, is_best=False):
        # Save the learned-model checkpoint
        scheduler = distiller.CompressionScheduler(self.model)
        masks = {param_name: masker.mask for param_name, masker in self.zeros_mask_dict.items()}
        scheduler.load_state_dict(state={'masks_dict': masks})
        if is_best:
            name = "BEST_adc_episode_{}".format(self.debug_stats['episode'])
        else:
            name = "adc_episode_{}".format(self.debug_stats['episode'])
        self.save_checkpoint_fn(epoch=self.debug_stats['episode'], model=self.model, scheduler=scheduler, name=name)


    def step(self, action):
        """Take a step, given an action.

        The action represents the desired sparsity.
        This function is invoked by the Agent.
        """
        msglogger.info("env.step - current_layer_id={} action={}".format(self.current_layer_id, action))
        assert action == 0 or (action >= self.action_low-0.001 and action <= self.action_high+0.001)
        #action = self.get_action(action)
        msglogger.info("action ********** (leave) {}".format(action))
        action = 1 - action
        layer_macs = self.get_macs(self.current_layer())
        if action > 0 and self.current_layer_id>-1:
            actual_action = self.__remove_channels(self.current_layer_id, action, prune_what="filters")
        else:
            actual_action = 0
        layer_macs_after_action = self.get_macs(self.current_layer())

        # Update the various counters after taking the step
        self.current_layer_id += 1
        next_layer_macs = self.get_macs(self.current_layer())
        self._removed_macs += (layer_macs - layer_macs_after_action)
        self._remaining_macs -= next_layer_macs
        self.prev_action = actual_action

        stats = ('Peformance/Validation/',
                 {'action': action} )
        distiller.log_training_progress(stats, None, self.debug_stats['episode'], steps_completed=self.current_layer_id,
                                        total_steps=13,
                                        log_freq=1, loggers=[self.tflogger])

    # def step(self, action):
    #     """Take a step, given an action.
    #
    #     The action represents the desired sparsity.
    #     This function is invoked by the Agent.
    #     """
    #     msglogger.info("env.step - current_layer_id={} action={}".format(self.current_layer_id, action))
    #     assert action == 0 or (action >= self.action_low and action <= self.action_high)
    #     action = 1 - action
    #     layer_macs = self.get_macs(self.current_layer())
    #     if action > 0 and self.current_layer_id>-1:
    #         actual_action = self.__remove_channels(self.current_layer_id, action, prune_what="filters")
    #     else:
    #         actual_action = 0
    #     layer_macs_after_action = self.get_macs(self.current_layer())
    #
    #     # Update the various counters after taking the step
    #     self.current_layer_id += 1
    #     next_layer_macs = self.get_macs(self.current_layer())
    #     self._removed_macs += (layer_macs - layer_macs_after_action)
    #     self._remaining_macs -= next_layer_macs
    #     self.prev_action = actual_action
    #
    #     stats = ('Peformance/Validation/',
    #              {'action': action} )
    #     distiller.log_training_progress(stats, None, self.debug_stats['episode'], steps_completed=self.current_layer_id,
    #                                     total_steps=13,
    #                                     log_freq=1, loggers=[self.tflogger])

        if self.episode_is_done():
            observation = self.get_final_obs()
            reward, top1 = self.compute_reward()
            # Save the learned-model checkpoint
            #self.save_checkpoint()
            # scheduler = distiller.CompressionScheduler(self.model)
            # scheduler.load_state_dict(state={'masks_dict': self.zeros_mask_dict})
            # name = "adc_episode_{}".format(self.debug_stats['episode'])
            # self.save_checkpoint_fn(epoch=self.debug_stats['episode'], model=self.model, scheduler=scheduler, name=name)
            self.debug_stats['episode'] += 1

            if reward > self.max_reward:
                self.max_reward = reward
                self.save_checkpoint(is_best=True)
                msglogger.info("Best reward={}  episode={}  top1={}".format(reward, self.debug_stats['episode'], top1))

        else:
            observation = self._get_obs(next_layer_macs)
            if True:
                reward = 0
            else:
                reward,_ = self.compute_reward()

        #self.prev_action = actual_action
        info = {}
        return observation, reward, self.episode_is_done(), info

    def _get_obs1(self, macs):
        """Produce a state embedding (i.e. an observation)"""

        layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, layer.name)

        obs = np.array([layer.t, conv_module.out_channels, conv_module.in_channels,
                        layer.ifm_h, layer.ifm_w, layer.stride[0], layer.k,
                        macs/self.dense_model_macs,
                        self.removed_macs(), self.remaining_macs(), 1-self.prev_action])

        assert len(obs) == self.STATE_EMBEDDING_LEN
        assert (macs/self.dense_model_macs + self.removed_macs() + self.remaining_macs()) <= 1
        #msglogger.info("obs={}".format(Observation._make(obs)))
        msglogger.info("obs={}".format(obs))
        return obs

    def _get_obs2(self, macs):
        """Produce a state embedding (i.e. an observation)"""

        layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, layer.name)

        obs = np.array([layer.t, conv_module.out_channels, conv_module.in_channels,
                        layer.ifm_h, layer.ifm_w, layer.stride[0], layer.k,
                        macs/self.dense_model_macs,
                        self.removed_macs(), self.remaining_macs(), 1-self.prev_action])

        id = np.zeros(13)
        id[layer.t] = 1
        obs = np.array([conv_module.out_channels, conv_module.in_channels,
                        layer.ifm_h, layer.ifm_w, layer.stride[0], layer.k,
                        macs/self.dense_model_macs,
                        self.removed_macs(), self.remaining_macs(), 1-self.prev_action])

        obs = np.concatenate([id, obs])
        assert len(obs) == self.STATE_EMBEDDING_LEN
        assert (macs/self.dense_model_macs + self.removed_macs() + self.remaining_macs()) <= 1
        #msglogger.info("obs={}".format(Observation._make(obs)))
        msglogger.info("obs={}".format(obs))
        return obs

    def _get_obs3(self, macs):
        """Produce a state embedding (i.e. an observation)"""

        layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, layer.name)

        obs = np.array([layer.t, conv_module.out_channels, conv_module.in_channels,
                        layer.ifm_h, layer.ifm_w, layer.stride[0], layer.k,
                        macs/self.dense_model_macs,
                        self.removed_macs(), self.remaining_macs(), 1-self.prev_action])

        id = np.zeros(13)
        id[layer.t] = 1
        # NORMALIZE THE FEATURES!!
        obs = np.array([conv_module.out_channels/512, conv_module.in_channels/512,
                        layer.ifm_h/32, layer.ifm_w/32, layer.stride[0]/2, layer.k/3,
                        macs/self.dense_model_macs,
                        self.removed_macs(), self.remaining_macs(), 1-self.prev_action])

        obs = np.concatenate([id, obs])
        assert len(obs) == self.STATE_EMBEDDING_LEN
        assert (macs/self.dense_model_macs + self.removed_macs() + self.remaining_macs()) <= 1
        #msglogger.info("obs={}".format(Observation._make(obs)))
        msglogger.info("obs={}".format(obs))
        return obs

    def _get_obs4(self, macs):
        """Produce a state embedding (i.e. an observation)"""

        layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, layer.name)

        # NORMALIZE THE FEATURES!!
        obs = np.array([layer.t, conv_module.out_channels / 512, conv_module.in_channels / 512,
                        layer.ifm_h / 32, layer.ifm_w / 32, layer.stride[0] / 2, layer.k / 3,
                        macs / self.dense_model_macs,
                        self.removed_macs(), self.remaining_macs(), 1 - self.prev_action])

        assert len(obs) == self.STATE_EMBEDDING_LEN
        assert (macs / self.dense_model_macs + self.removed_macs() + self.remaining_macs()) <= 1
        msglogger.info("obs={}".format(Observation._make(obs)))
        return obs

    def _get_obs(self, macs):
        #return self._get_obs3(macs)
        return self._get_obs4(macs)


    def get_final_obs(self):
        """Return the final stae embedding (observation)
        The final state is reached after we traverse all of the Convolution layers.
        """
        if True:
            obs = np.array([-1, 0, 0,
                             0, 0, 0, 0,
                             0, self.removed_macs(), 0, 1 - self.prev_action])
        else:
            id = np.zeros(13)
            obs = np.array([ 0, 0,
                             0, 0, 0, 0,
                             0, self.removed_macs(), 0, 1 - self.prev_action])
            obs = np.concatenate([id, obs])

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
        #reward = -1 * (1 - top1/100)
        #reward = -1 * (1-top1/100) * math.log(total_macs/self.dense_model_macs)
        reward = -1 * (1-top1/100) * math.log(total_macs)
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
        stats = ('Peformance/Validation/',
                 OrderedDict([('Loss', vloss),
                              ('Top1', top1),
                              ('Top5', top5),
                              ('reward', reward),
                              ('total_macs', int(total_macs)),
                              ('log(total_macs)', math.log(total_macs)),
                              ('log(total_macs/self.dense_model_macs)', math.log(total_macs/self.dense_model_macs)),
                              ('total_nnz', int(total_nnz))]))
        distiller.log_training_progress(stats, None, self.debug_stats['episode'], steps_completed=0, total_steps=1,
                                        log_freq=1, loggers=[self.tflogger, self.pylogger])
        return reward, top1

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


def do_adc(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn):
    np.random.seed()

    if USE_COACH:
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
            'save_checkpoint_fn': save_checkpoint_fn
        }
        graph_manager.create_graph(task_parameters)
        graph_manager.improve()
        return

    """Random ADC agent"""
    env = CNNEnvironment(model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn)

    for ep in range(10):
        observation = env.reset()
        for t in range(100):
            env.render(0, 0)
            msglogger.info("[episode={}:{}] observation = {}".format(ep, t, observation))
            # take a random action
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                msglogger.info("Episode finished after {} timesteps".format(t+1))
                break


class RandomADCActionSpace(object):
    def sample(self):
        return random.uniform(0, 1)


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
    for id, (name, m) in enumerate(model.named_modules()):
        if isinstance(m, torch.nn.Conv2d):
            conv = SimpleNamespace()
            conv.t = len(conv_layers)
            conv.k = m.kernel_size[0]
            conv.stride = m.stride

            # Use the SummaryGraph to obtain some other details of the models
            conv_op = g.find_op(normalize_module_name(name))
            assert conv_op is not None

            conv.macs = conv_op['attrs']['MACs']
            total_macs += conv.macs
            conv.ofm_h = g.param_shape(conv_op['outputs'][0])[2]
            conv.ofm_w = g.param_shape(conv_op['outputs'][0])[3]
            conv.ifm_h = g.param_shape(conv_op['inputs'][0])[2]
            conv.ifm_w = g.param_shape(conv_op['inputs'][0])[3]

            conv.name = name
            conv.id = id
            conv_layers[len(conv_layers)] = conv

    return conv_layers, total_macs


class CNNEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    STATE_EMBEDDING_LEN = len(Observation._fields)

    def __init__(self, model, dataset, arch, data_loader, validate_fn, save_checkpoint_fn):
        self.pylogger = distiller.data_loggers.PythonLogger(msglogger)
        self.tflogger = distiller.data_loggers.TensorBoardLogger(msglogger.logdir)

        self.action_space = RandomADCActionSpace()
        self.dataset = dataset
        self.arch = arch
        self.data_loader = data_loader
        self.validate_fn = validate_fn
        self.save_checkpoint_fn = save_checkpoint_fn
        self.orig_model = model

        self.conv_layers, self.dense_model_macs = collect_conv_details(model, dataset)
        self.reset(init_only=True)
        msglogger.info("Model %s has %d Convolution layers", arch, len(self.conv_layers))
        msglogger.info("\tTotal MACs: %s" % distiller.pretty_int(self.dense_model_macs))

        self.debug_stats = {'episode': 0}

        # Gym
        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Box(0, 1, shape=(1,))
        self.observation_space = spaces.Box(0, float("inf"), shape=(self.STATE_EMBEDDING_LEN,))

    def reset(self, init_only=False):
        """Reset the environment.
        This is invoked by the Agent.
        """
        msglogger.info("Resetting the environment")
        self.current_layer_id = 0
        self.prev_action = 0
        self.model = copy.deepcopy(self.orig_model)
        self.zeros_mask_dict = create_model_masks(self.model)
        self._remaining_macs = self.dense_model_macs
        self._removed_macs = 0

        # self.unprocessed_layers = []
        # for conv in self.conv_layers:
        #     self.unprocessed_layers.append(conv)
        # self.processed_layers = []
        if init_only:
            return

        #layer_macs = self.get_macs(self.current_layer())
        #return self._get_obs(layer_macs)
        obs, _, _, _, = self.step(0)
        return obs


    def num_layers(self):
        return len(self.conv_layers)

    def current_layer(self):
        try:
            return self.conv_layers[self.current_layer_id]
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

    def step(self, action):
        """Take a step, given an action.
        This is invoked by the Agent.
        """
        layer_macs = self.get_macs(self.current_layer())
        if action > 0:
            actual_action = self.__remove_channels(self.current_layer_id, action)
        else:
            actual_action = 0
        layer_macs_after_action = self.get_macs(self.current_layer())

        # Update the various counters after taking the step
        self.current_layer_id += 1
        next_layer_macs = self.get_macs(self.current_layer())
        self._removed_macs += (layer_macs - layer_macs_after_action)
        self._remaining_macs -= next_layer_macs

        #self.prev_action = actual_action
        if self.episode_is_done():
            observation = self.get_final_obs()
            reward = self.compute_reward()
            # Save the learned-model checkpoint
            scheduler = distiller.CompressionScheduler(self.model)
            scheduler.load_state_dict(state={'masks_dict': self.zeros_mask_dict})
            self.save_checkpoint_fn(epoch=self.debug_stats['episode'], model=self.model, scheduler=scheduler)
            self.debug_stats['episode'] += 1
        else:
            observation = self._get_obs(next_layer_macs)
            if True:
                reward = 0
            else:
                reward = self.compute_reward()

        self.prev_action = actual_action
        info = {}
        return observation, reward, self.episode_is_done(), info

    def _get_obs(self, macs):
        """Produce a state embedding (i.e. an observation)"""

        layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, layer.name)

        obs = np.array([layer.t, conv_module.out_channels, conv_module.in_channels,
                        layer.ifm_h, layer.ifm_w, layer.stride[0], layer.k,
                        macs/self.dense_model_macs, self.removed_macs(), self.remaining_macs(), self.prev_action])

        assert len(obs) == self.STATE_EMBEDDING_LEN
        assert (macs/self.dense_model_macs + self.removed_macs() + self.remaining_macs()) <= 1
        msglogger.info("obs={}".format(Observation._make(obs)))
        return obs

    def get_final_obs(self):
        """Return the final stae embedding (observation)
        The final state is reached after we traverse all of the Convolution layers.
        """
        obs = np.array([-1, 0, 0,
                         0, 0, 0, 0,
                         0, self.removed_macs(), 0, self.prev_action])
        assert len(obs) == self.STATE_EMBEDDING_LEN
        return obs

    def get_macs(self, layer):
        """Return the number of MACs required to compute <layer>'s Convolution"""
        if layer is None:
            return 0

        conv_module = distiller.model_find_module(self.model, layer.name)
        # MACs = volume(OFM) * (#IFM * K^2)
        return (conv_module.out_channels * layer.ofm_h * layer.ofm_w) * (conv_module.in_channels * layer.k**2)

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

        msglogger.info("ADC: removing %.1f%% channels from %s" % (fraction_to_prune*100, conv_pname))

        if prune_what == "channels":
            calculate_sparsity = distiller.sparsity_ch
            reg_regims = {conv_pname: [fraction_to_prune, "Channels"]}
            remove_structures = distiller.remove_channels
        else:
            calculate_sparsity = distiller.sparsity_3D
            reg_regims = {conv_pname: [fraction_to_prune, "3D"]}
            remove_structures = distiller.remove_filters

        # Create a channel-ranking pruner
        pruner = distiller.pruning.L1RankedStructureParameterPruner("adc_pruner", reg_regims)
        pruner.set_param_mask(conv_p, conv_pname, self.zeros_mask_dict, meta=None)

        if (self.zeros_mask_dict[conv_pname].mask is None or
            calculate_sparsity(self.zeros_mask_dict[conv_pname].mask) == 0):
            msglogger.info("__remove_channels: aborting because there are no channels to prune")
            return 0

        # Use the mask to prune
        self.zeros_mask_dict[conv_pname].apply_mask(conv_p)
        actual_sparsity = calculate_sparsity(conv_p)
        remove_structures(self.model, self.zeros_mask_dict, self.arch, self.dataset, optimizer=None)
        return actual_sparsity

    def compute_reward(self):
        """The ADC paper defines reward = -Error"""
        distiller.log_weights_sparsity(self.model, -1, loggers=[self.pylogger])

        top1, top5, vloss = self.validate_fn(model=self.model, epoch=self.debug_stats['episode'])
        _, total_macs = collect_conv_details(self.model, self.dataset)
        reward = -1 * vloss * math.log(total_macs)
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
                              ('log(total_macs)', math.log(total_macs))]))
        distiller.log_training_progress(stats, None, self.debug_stats['episode'], steps_completed=0, total_steps=1,
                                        log_freq=1, loggers=[self.tflogger, self.pylogger])

        return reward

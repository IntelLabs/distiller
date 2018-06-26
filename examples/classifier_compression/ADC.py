import random
import math
import copy
import logging
import torch
import gym
from gym import spaces
import distiller
from apputils import SummaryGraph
from collections import OrderedDict
from types import SimpleNamespace
from distiller import normalize_module_name

from base_parameters import TaskParameters
from examples.automated_deep_compression.presets.ADC_DDPG import graph_manager

msglogger = logging.getLogger()


# TODO: move this to utils
def pretty_int(i):
    return "{:,}".format(i)


# TODO: this is also defined in test_pruning.py
def create_model_masks(model):
    # Create the masks
    zeros_mask_dict = {}
    for name, param in model.named_parameters():
        masker = distiller.ParameterMasker(name)
        zeros_mask_dict[name] = masker
    return zeros_mask_dict


USE_COACH = True
def do_adc(model, dataset, arch, data_loader, validate_fn):
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
            'validate_fn': validate_fn
        }
        graph_manager.create_graph(task_parameters)
        graph_manager.improve()
        return

    """Random ADC agent"""
    env = CNNEnvironment(model, dataset, arch, data_loader, validate_fn)

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
    STATE_EMBEDDING_LEN = 11

    def __init__(self, model, dataset, arch, data_loader, validate_fn):
        self.action_space = RandomADCActionSpace()
        self.dataset = dataset
        self.arch = arch
        self.data_loader = data_loader
        self.validate_fn = validate_fn
        self.orig_model = model

        self.conv_layers, self.total_macs = collect_conv_details(model, dataset)
        self.reset(init_only=True)
        msglogger.info("Model %s has %d Convolution layers", arch, len(self.conv_layers))
        msglogger.info("\tTotal MACs: %s" % pretty_int(self.total_macs))

        # Gym
        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Box(0, 1, shape=(1,))
        self.observation_space = spaces.Box(0, float("inf"), shape=(self.STATE_EMBEDDING_LEN,))
        # Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0]))

    def reset(self, init_only=False):
        """Reset the environment.
        This is invoked by the Agent.
        """
        msglogger.info("Resetting the environment")
        self.current_layer_id = 1
        self.prev_action = 0
        self.model = copy.deepcopy(self.orig_model)
        #self.model = self.model.cuda()
        self.zeros_mask_dict = create_model_masks(self.model)
        self._remaining_macs = self.total_macs
        self._removed_macs = 0

        # self.unprocessed_layers = []
        # for conv in self.conv_layers:
        #     self.unprocessed_layers.append(conv)
        # self.processed_layers = []
        if init_only:
            return

        layer_macs = self.get_macs(self.current_layer())
        return self._get_obs(layer_macs)

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
        #return 1 - self.sum_list_macs(self.unprocessed_layers) / self.total_macs
        return self._remaining_macs / self.total_macs

    def removed_macs(self):
        """Return the amount of MACs removed so far.
        This is normalized to the range 0..1
        """
        #return self.sum_list_macs(self.processed_layers) / self.total_macs
        return self._removed_macs / self.total_macs

    def sum_list_macs(self, conv_list):
        """Sum the MACs in the provided list of Convolution layers"""
        total_macs = 0
        for conv in conv_list:
            total_macs += conv.macs
        return total_macs

    def render(self, mode, close):
        """Provide some feedback to the user about what's going on
        This is invoked by the Agent.
        """
        if self.current_layer_id == 0:
            msglogger.info("+" + "-" * 50 + "+")
            msglogger.info("Starting a new episode")
            msglogger.info("+" + "-" * 50 + "+")

        msglogger.info("Environment: current_layer_id=%d" % self.current_layer_id)
        pylogger = distiller.data_loggers.PythonLogger(msglogger)
        distiller.log_weights_sparsity(self.model, -1, loggers=[pylogger])

    def step(self, action):
        """Take a step, given an action.
        This is invoked by the Agent.
        """
        layer_macs = self.get_macs(self.current_layer())
        self.__remove_channels(self.current_layer_id, action)
        layer_macs_after_action = self.get_macs(self.current_layer())

        # Update the various counters after taking the step
        self.current_layer_id += 1
        next_layer_macs = self.get_macs(self.current_layer())
        self._removed_macs += (layer_macs - layer_macs_after_action)
        self._remaining_macs -= layer_macs

        if self.episode_is_done():
            observation = self.get_final_obs()
            reward = self.compute_reward()
        else:
            observation = self._get_obs(next_layer_macs)
            reward = 0

        self.prev_action = action.item()
        info = {}
        return observation, reward, self.episode_is_done(), info

    def _get_obs(self, macs):
        """Produce a state embedding (i.e. an observation)"""

        layer = self.current_layer()
        conv_module = distiller.model_find_module(self.model, layer.name)

        obs = (layer.t, conv_module.out_channels, conv_module.in_channels,
               layer.ifm_h, layer.ifm_w, layer.stride, layer.k,
               macs, self.removed_macs(), self.remaining_macs(), self.prev_action)
        assert len(obs) == self.STATE_EMBEDDING_LEN
        return obs

    def get_final_obs(self):
        """Return the final stae embedding (observation)
        The final state is reached after we traverse all of the Convolution layers.
        """
        obs = (-1, 0, 0,
               0, 0, 0, 0,
               0, self.removed_macs(), 0, self.prev_action)
        assert len(obs) == self.STATE_EMBEDDING_LEN
        return obs

    def get_macs(self, layer):
        """Return the number of MACs required to compute <layer>'s Convolution"""
        if layer is None:
            return 0

        conv_module = distiller.model_find_module(self.model, layer.name)
        # MACs = volume(OFM) * (#IFM * K^2)
        return (conv_module.out_channels * layer.ofm_h * layer.ofm_w) * (conv_module.in_channels * layer.k**2)

    def __remove_channels(self, idx, fraction_to_prune):
        """Physically remove channels and corresponding filters from the model"""
        if idx not in range(self.num_layers()):
            raise ValueError("idx=%d is not in correct range (0-%d)" % (idx, self.num_layers()))

        if fraction_to_prune == 0:
            return

        if fraction_to_prune < 0:
            raise ValueError("fraction_to_prune=%f is illegal" % (fraction_to_prune))

        layer = self.conv_layers[idx]
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)

        msglogger.info("ADC: removing %.1f%% channels from %s" % (fraction_to_prune*100, conv_pname))

        # Create a channel-ranking pruner
        reg_regims = {conv_pname: [fraction_to_prune, "Channels"]}
        pruner = distiller.pruning.L1RankedStructureParameterPruner("channel_pruner", reg_regims)
        pruner.set_param_mask(conv_p, conv_pname, self.zeros_mask_dict, meta=None)

        if self.zeros_mask_dict[conv_pname].mask is None or \
           distiller.sparsity_ch(self.zeros_mask_dict[conv_pname].mask) == 0:
            msglogger.info("remove_channels: aborting because there are no channels to prune")
            return

        # Use the mask to prune
        self.zeros_mask_dict[conv_pname].apply_mask(conv_p)
        distiller.remove_channels(self.model, self.zeros_mask_dict, self.arch, self.dataset)

    def compute_reward(self):
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=")
        """The ADC paper defines reward = -Error"""

        pylogger = distiller.data_loggers.PythonLogger(msglogger)
        distiller.log_weights_sparsity(self.model, -1, loggers=[pylogger])

        top1, top5, vloss = self.validate_fn(model=self.model.cuda(), epoch=0)

        _, total_macs = collect_conv_details(self.model, self.dataset)
        reward = -1 * vloss * math.log(total_macs)
        msglogger.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ {}".format(reward))
        return reward

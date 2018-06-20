import random
import logging
import torch
import gym
from gym import spaces
import distiller
from apputils import SummaryGraph
from collections import OrderedDict
from types import SimpleNamespace
msglogger = logging.getLogger()


# TODO: this is from thinning.py
def normalize_layer_name(layer_name):
    start = layer_name.find('module.')
    normalized_layer_name = layer_name
    if start != -1:
        normalized_layer_name = layer_name[:start] + layer_name[start + len('module.'):]
    return normalized_layer_name

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


def do_adc(model, dataset, arch):
    """Random ADC agent"""
    env = CNNEnvironment(model, dataset, arch)

    for i_episode in range(10):
        observation = env.reset()
        for t in range(19):
            msglogger.info("{}".format(observation))
            env._render(0,0)
            # take a random action
            action = env.action_space.sample()
            observation, reward, done, info = env._step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


class RandomADCActionSpace(object):
    def sample(self):
        return random.uniform(0, 1)


class CNNEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}
    STATE_EMBEDDING_LEN = 10

    def __init__(self, model, dataset, arch):
        self.idx_cntr = 0
        self.action_space = RandomADCActionSpace()
        self.dataset = dataset
        self.arch = arch
        self.model = model
        self.conv_layers, self.total_macs = self.__collect_conv_details()
        msglogger.info("Model %s has %d Convolution layers", arch, len(self.conv_layers))
        msglogger.info("\tTotal MACs: %s" % pretty_int(self.total_macs))

        self.zeros_mask_dict = create_model_masks(self.model)

        # Gym
        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Box(0, 1, shape=(1,))
        self.observation_space = spaces.Box(0, 1, shape=(self.STATE_EMBEDDING_LEN,))
        # Box(low=np.array([-1.0,-2.0]), high=np.array([2.0,4.0]))


    def __collect_conv_details(self):
        if self.dataset == 'imagenet':
            dummy_input = torch.randn(1, 3, 224, 224)
        elif self.dataset == 'cifar10':
            dummy_input = torch.randn(1, 3, 32, 32)
        else:
            raise ValueError("dataset %s is not supported" % dataset)

        g = SummaryGraph(self.model, dummy_input)
        conv_layers = OrderedDict()
        total_macs = 0
        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                conv = SimpleNamespace()
                conv.t = len(conv_layers)
                conv.k = m.kernel_size[0]
                conv.n = m.out_channels
                conv.c = m.in_channels
                conv.stride = m.stride

                # Use the SummaryGraph to obtain some other details of the models
                conv_op = g.find_op(normalize_layer_name(name))
                assert conv_op is not None

                conv.macs = conv_op['attrs']['MACs']
                total_macs += conv.macs
                conv.h = g.param_shape(conv_op['inputs'][0])[2]
                conv.w = g.param_shape(conv_op['inputs'][0])[3]
                conv.name = name
                conv_layers[len(conv_layers)] = conv

        return conv_layers, total_macs


    def _reset(self):
        self.idx_cntr = 0
        return self._get_obs()


    def num_layers(self):
        return len(self.conv_layers)


    def _render(self, mode, close):
        if self.idx_cntr == 0:
            msglogger.info("+" + "-" * 20 + "+")
            msglogger.info("Starting a new episode")
            msglogger.info("+" + "-" * 20 + "+")

        pylogger = distiller.data_loggers.PythonLogger(msglogger)
        distiller.log_weights_sparsity(self.model, -1, loggers=[pylogger])


    def _step(self, action):
        # Take a step
        self.remove_channels(self.idx_cntr, action)
        self.idx_cntr = (self.idx_cntr + 1) % self.num_layers()
        observation = self._get_obs()
        reward = 0
        done = False
        info = None
        return observation, reward, done, info


    def _get_obs(self):
        """Produce a state embedding"""
        layer = self.conv_layers[self.idx_cntr]
        conv_module = distiller.model_find_module(self.model, layer.name)
        layer.macs = 0
        layer.reduced = 0
        layer.rest = 0
        obs = (layer.t, conv_module.out_channels, conv_module.in_channels, layer.h, layer.w, layer.stride, layer.k,
               layer.macs, layer.reduced, layer.rest)
        assert len(obs) == self.STATE_EMBEDDING_LEN
        return obs


    def remove_channels(self, idx, fraction_to_prune):
        if idx not in range(self.num_layers()):
            raise ValueError("idx=%d is not in correct range (0-%d)" %(idx, self.num_layers()))

        if fraction_to_prune <= 0:
            raise ValueError("fraction_to_prune=%f is illegal" %(fraction_to_prune))

        layer = self.conv_layers[idx]
        conv_pname = layer.name + ".weight"
        conv_p = distiller.model_find_param(self.model, conv_pname)

        msglogger.info("ADC: removing %.1f%% channels from %s" % (fraction_to_prune*100, conv_pname))

        # Create a channel-ranking pruner
        reg_regims = {conv_pname : [fraction_to_prune, "Channels"]}
        pruner = distiller.pruning.L1RankedStructureParameterPruner("channel_pruner", reg_regims)
        pruner.set_param_mask(conv_p, conv_pname, self.zeros_mask_dict, meta=None)

        if self.zeros_mask_dict[conv_pname].mask is None or distiller.sparsity_ch(self.zeros_mask_dict[conv_pname].mask) == 0:
            msglogger.info("remove_channels: aborting because there are no channels to prune")
            return

        # Use the mask to prune
        self.zeros_mask_dict[conv_pname].apply_mask(conv_p)
        distiller.remove_channels(self.model, self.zeros_mask_dict, self.arch, self.dataset)

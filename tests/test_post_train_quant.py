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
import pytest
import torch
import torch.testing
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import random
from copy import deepcopy

from distiller.quantization import RangeLinearQuantParamLayerWrapper, LinearQuantMode, ClipMode, \
    RangeLinearQuantConcatWrapper, RangeLinearQuantEltwiseMultWrapper, RangeLinearQuantEltwiseAddWrapper, \
    PostTrainLinearQuantizer
from distiller.data_loggers import QuantCalibrationStatsCollector, collector_context
import distiller.modules
from common import WrappedSequential


###############################################################################
# Test Convolution
###############################################################################

@pytest.fixture()
def conv_input():
    return torch.cat((torch.tensor([[[[-7, 5], [2, -3]]]], dtype=torch.float32),
                      torch.tensor([[[[-15, 10], [-1, 5]]]], dtype=torch.float32)), 0)


@pytest.fixture()
def conv_weights():
    return torch.tensor([[[[-1, -0.5, 0], [0.5, 1, 1.5], [2, 2.5, 3]]],
                         [[[-0.3, -0.25, -0.2], [-0.15, -0.1, -0.05], [0, 0.05, 0.1]]]], dtype=torch.float32)


conv_stats_dict = OrderedDict()
conv_stats_dict['inputs'] = OrderedDict()
conv_stats_dict['inputs'][0] = {'min': -15, 'max': 10, 'avg_min': -11, 'avg_max': 7.5}
conv_stats_dict['output'] = {'min': -3.5, 'max': 14.5, 'avg_min': -1.025, 'avg_max': 8.5}


@pytest.mark.parametrize(
    "mode, clip_acts, per_channel_wts, conv_stats, expected_output",
    [
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE, False, None,
         torch.cat((torch.tensor([[[[-3.648135333, -2.14596196], [0.858384784, 2.432090222]],
                                   [[0.214596196, 0.500724457], [0.715320653, 0.786852719]]]], dtype=torch.float32),
                    torch.tensor([[[[12.51811144, 13.01883589], [14.0918168, 14.59254133]],
                                   [[1.359109242, 1.645237503], [1.573705438, 1.645237503]]]], dtype=torch.float32)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.AVG, False, None,
         torch.cat((torch.tensor([[[[-1.089218234, -1.089218234], [1.055180164, 2.518817167]],
                                   [[0.238266489, 0.476532978], [0.680761396, 0.782875606]]]], dtype=torch.float32),
                    torch.tensor([[[[7.59048957, 7.59048957], [7.59048957, 7.59048957]],
                                   [[1.123256304, 1.259408583], [1.089218234, 1.089218234]]]], dtype=torch.float32)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE, True, None,
         torch.cat((torch.tensor([[[[-3.648135333, -2.14596196], [0.858384784, 2.432090222]],
                                   [[0.214596196, 0.429192392], [0.715320653, 0.858384784]]]], dtype=torch.float32),
                    torch.tensor([[[[12.51811144, 13.01883589], [14.09181687, 14.59254133]],
                                   [[1.430641307, 1.502173372], [1.573705438, 1.645237503]]]], dtype=torch.float32)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.AVG, True, None,
         torch.cat((torch.tensor([[[[-1.089768056, -1.089768056], [1.055712804, 2.52008863]],
                                   [[0.238386762, 0.408663021], [0.681105035, 0.817326042]]]], dtype=torch.float32),
                    torch.tensor([[[[7.59432114, 7.59432114], [7.59432114, 7.59432114]],
                                   [[1.191933811, 1.15787856], [1.123823308, 1.089768056]]]], dtype=torch.float32)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE, False, conv_stats_dict,
         torch.cat((torch.tensor([[[[-3.529411765, -2.117647059], [0.917647059, 2.4]],
                                   [[0.211764706, 0.494117647], [0.705882353, 0.776470588]]]], dtype=torch.float32),
                    torch.tensor([[[[12.56470588, 12.98823529], [14.04705882, 14.47058824]],
                                   [[1.341176471, 1.623529412], [1.552941176, 1.623529412]]]], dtype=torch.float32)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.AVG, False, conv_stats_dict,
         torch.cat((torch.tensor([[[[-1.008529412, -1.008529412], [1.045882353, 2.502647059]],
                                   [[0.224117647, 0.485588235], [0.672352941, 0.784411765]]]], dtype=torch.float32),
                    torch.tensor([[[[8.516470588, 8.516470588], [8.516470588, 8.516470588]],
                                   [[1.120588235, 1.27], [1.120588235, 1.083235294]]]], dtype=torch.float32)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE, True, conv_stats_dict,
         torch.cat((torch.tensor([[[[-3.529411765, -2.117647059], [0.917647059, 2.4]],
                                   [[0.211764706, 0.423529412], [0.705882353, 0.847058824]]]], dtype=torch.float32),
                    torch.tensor([[[[12.56470588, 12.98823529], [14.04705882, 14.47058824]],
                                   [[1.482352941, 1.482352941], [1.623529412, 1.623529412]]]], dtype=torch.float32)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.AVG, True, conv_stats_dict,
         torch.cat((torch.tensor([[[[-1.008529412, -1.008529412], [1.045882353, 2.502647059]],
                                   [[0.261470588, 0.410882353], [0.672352941, 0.821764706]]]], dtype=torch.float32),
                    torch.tensor([[[[8.516470588, 8.516470588], [8.516470588, 8.516470588]],
                                   [[1.195294118, 1.157941176], [1.120588235, 1.083235294]]]], dtype=torch.float32)),
                   dim=0)
         )
    ]
)
def test_conv_layer_wrapper(conv_input, conv_weights, mode, clip_acts, per_channel_wts, conv_stats, expected_output):
    layer = torch.nn.Conv2d(conv_input.shape[1], expected_output.shape[1], conv_weights.shape[-1],
                            padding=1, bias=False)
    layer.weight.data = conv_weights

    model = RangeLinearQuantParamLayerWrapper(layer, 8, 8, mode=mode, clip_acts=clip_acts,
                                              per_channel_wts=per_channel_wts, activation_stats=conv_stats)

    with pytest.raises(RuntimeError):
        model(conv_input)

    model.eval()

    output = model(conv_input)

    torch.testing.assert_allclose(output, expected_output)


###############################################################################
# Test Linear
###############################################################################

@pytest.fixture()
def linear_input():
    return torch.tensor([[-7, 5, 2, -3]], dtype=torch.float32)


@pytest.fixture()
def linear_weights():
    return torch.tensor([[-1, 0.5, 0, 0.5],
                         [-0.05, 0, 0.05, 0.1],
                         [0.3, 0.6, -0.1, -0.2]], dtype=torch.float32)


@pytest.fixture()
def linear_bias():
    return torch.tensor([-0.3, 0.1, -0.5], dtype=torch.float32)


@pytest.mark.parametrize(
    "mode, clip_acts, per_channel_wts, expected_output",
    [
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE, False,
         torch.tensor([[7.686200692, 0.241135708, 0.783691051]], dtype=torch.float32)),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE, True,
         torch.tensor([[7.698823529, 0.241531719, 0.784978085]], dtype=torch.float32)),
        (LinearQuantMode.SYMMETRIC, ClipMode.NONE, False,
         torch.tensor([[7.728687457, 0.243423227, 0.791125488]], dtype=torch.float32)),
        (LinearQuantMode.SYMMETRIC, ClipMode.NONE, True,
         torch.tensor([[7.728687457, 0.243423227, 0.791125488]], dtype=torch.float32))
    ]
)
def test_linear_layer_wrapper(linear_input, linear_weights, linear_bias,
                              mode, clip_acts, per_channel_wts, expected_output):
    layer = torch.nn.Linear(linear_input.shape[1], expected_output.shape[1], bias=True)
    layer.weight.data = linear_weights
    layer.bias.data = linear_bias

    model = RangeLinearQuantParamLayerWrapper(layer, 8, 8, mode=mode, clip_acts=clip_acts,
                                              per_channel_wts=per_channel_wts)

    with pytest.raises(RuntimeError):
        model(linear_input)

    model.eval()

    output = model(linear_input)

    torch.testing.assert_allclose(output, expected_output)


###############################################################################
# Test Concat
###############################################################################

@pytest.fixture()
def inputs():
    in_0_b_0 = torch.tensor([[[[-10, 31], [5, 10]], [[1, 8], [-3, 7]]]], dtype=torch.float32)
    in_0_b_1 = torch.tensor([[[[-8, 16], [-15, -12]], [[-20, 13], [8, 0]]]], dtype=torch.float32)
    in_0 = torch.cat((in_0_b_0, in_0_b_1), 0)
    in_1_b_0 = torch.tensor([[[[-3, 6], [0, 8]], [[4, 10], [-7, 1]]]], dtype=torch.float32)
    in_1_b_1 = torch.tensor([[[[-100, 50], [6, 12]], [[80, -30], [-16, 3]]]], dtype=torch.float32)
    in_1 = torch.cat((in_1_b_0, in_1_b_1), 0)
    return in_0, in_1


input_stats = OrderedDict()
input_stats[0] = {'min': -20, 'max': 31, 'avg_min': -15, 'avg_max': 23.5}
input_stats[1] = {'min': -100, 'max': 80, 'avg_min': -53.5, 'avg_max': 45}


@pytest.fixture()
def concat_stats():
    stats = OrderedDict()
    stats['inputs'] = input_stats
    stats['output'] = {'min': -100, 'max': 80, 'avg_min': -55, 'avg_max': 55.5}
    return stats


@pytest.mark.parametrize(
    "mode, clip_acts, expected_output",
    [
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE,
         torch.tensor([[[[-9.882352941, 31.05882353], [4.941176471, 9.882352941]],
                        [[0.705882353, 7.764705882], [-2.823529412, 7.058823529]],
                        [[-2.823529412, 5.647058824], [0, 7.764705882]],
                        [[4.235294118, 9.882352941], [-7.058823529, 0.705882353]]],
                       [[[-7.764705882, 16.23529412], [-14.82352941, -12]],
                        [[-19.76470588, 12.70588235], [7.764705882, 0]],
                        [[-100.2352941, 50.11764706], [5.647058824, 12]],
                        [[79.76470588, -29.64705882], [-16.23529412, 2.823529412]]]])),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.AVG,
         torch.tensor([[[[-9.966666667, 23.4], [4.766666667, 9.966666667]],
                        [[0.866666667, 7.8], [-3.033333333, 6.933333333]],
                        [[-3.033333333, 6.066666667], [0, 8.233333333]],
                        [[3.9, 9.966666667], [-6.933333333, 1.3]]],
                       [[[-7.8, 16.03333333], [-14.73333333, -12.13333333]],
                        [[-14.73333333, 13], [7.8, 0]],
                        [[-53.73333333, 44.63333333], [6.066666667, 12.13333333]],
                        [[44.63333333, -30.33333333], [-16.03333333, 3.033333333]]]])),
        (LinearQuantMode.SYMMETRIC, ClipMode.NONE,
         torch.tensor([[[[-10.23622047, 30.70866142], [4.724409449, 10.23622047]],
                        [[0.787401575, 7.874015748], [-3.149606299, 7.086614173]],
                        [[-3.149606299, 6.299212598], [0, 7.874015748]],
                        [[3.937007874, 10.23622047], [-7.086614173, 0.787401575]]],
                       [[[-7.874015748, 15.7480315], [-14.96062992, -11.81102362]],
                        [[-19.68503937, 12.5984252], [7.874015748, 0]],
                        [[-100, 50.39370079], [6.299212598, 11.81102362]],
                        [[80.31496063, -29.92125984], [-15.7480315, 3.149606299]]]])),
        (LinearQuantMode.SYMMETRIC, ClipMode.AVG,
         torch.tensor([[[[-10.0511811, 23.5984252], [4.807086614, 10.0511811]],
                        [[0.874015748, 7.866141732], [-3.059055118, 6.992125984]],
                        [[-3.059055118, 5.681102362], [0, 7.866141732]],
                        [[3.933070866, 10.0511811], [-6.992125984, 0.874015748]]],
                       [[[-7.866141732, 15.73228346], [-14.85826772, -12.23622047]],
                        [[-20.1023622, 13.11023622], [7.866141732, 0]],
                        [[-53.7519685, 50.25590551], [5.681102362, 11.7992126]],
                        [[53.31496063, -29.71653543], [-16.16929134, 3.059055118]]]]))
    ]
)
def test_concat_layer_wrapper(inputs, concat_stats, mode, clip_acts, expected_output):
    with pytest.raises(ValueError):
        # Check exception on wrong layer type
        RangeLinearQuantConcatWrapper(torch.nn.Module(), 8, mode, clip_acts, concat_stats)

    layer = distiller.modules.Concat(dim=1)

    with pytest.raises(NotImplementedError):
        # Check exception on no stats
        RangeLinearQuantConcatWrapper(layer, 8, mode, clip_acts, activation_stats=None)

    model = RangeLinearQuantConcatWrapper(layer, 8, mode, clip_acts, concat_stats)
    model.eval()
    output = model(*inputs)

    torch.testing.assert_allclose(output, expected_output)


###############################################################################
# Test Element-Wise Multiplication
###############################################################################

@pytest.fixture()
def eltwise_mult_stats():
    stats = OrderedDict()
    stats['inputs'] = input_stats
    stats['output'] = {'min': -1600, 'max': 800, 'avg_min': -800, 'avg_max': 493}
    return stats


@pytest.mark.parametrize(
    "mode, clip_acts, expected_output",
    [
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE,
         torch.tensor([[[[28.23529412, 178.8235294], [0, 75.29411765]],
                        [[0, 75.29411765], [18.82352941, 9.411764706]]],
                       [[[800, 800], [-84.70588235, -141.1764706]],
                        [[-1590.588235, -385.8823529], [-131.7647059, 0]]]])),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.AVG,
         torch.tensor([[[[30.42352941, 147.0470588], [0, 81.12941176]],
                        [[5.070588235, 81.12941176], [20.28235294, 10.14117647]]],
                       [[[431, 491.8470588], [-91.27058824, -141.9764706]],
                        [[-669.3176471, -390.4352941], [-126.7647059, 0]]]])),
        (LinearQuantMode.SYMMETRIC, ClipMode.NONE,
         torch.tensor([[[[37.79527559, 201.5748031], [0, 75.59055118]],
                        [[0, 88.18897638], [25.19685039, 0]]],
                       [[[806.2992126, 806.2992126], [-88.18897638, -138.5826772]],
                        [[-1612.598425, -390.5511811], [-125.984252, 0]]]])),
        (LinearQuantMode.SYMMETRIC, ClipMode.AVG,
         torch.tensor([[[[31.49606299, 138.5826772], [0, 81.88976378]],
                        [[6.299212598, 81.88976378], [18.8976378, 6.299212598]]],
                       [[[428.3464567, 800], [-88.18897638, -144.8818898]],
                        [[-806.2992126, -390.5511811], [-125.984252, 0]]]]))
    ]
)
def test_eltwise_mult_layer_wrapper(inputs, eltwise_mult_stats, mode, clip_acts, expected_output):
    with pytest.raises(ValueError):
        # Check exception on wrong layer type
        RangeLinearQuantEltwiseMultWrapper(torch.nn.Module(), 8, mode, clip_acts, eltwise_mult_stats)

    layer = distiller.modules.EltwiseMult()

    with pytest.raises(NotImplementedError):
        # Check exception on no stats
        RangeLinearQuantEltwiseMultWrapper(layer, 8, mode, clip_acts, activation_stats=None)

    model = RangeLinearQuantEltwiseMultWrapper(layer, 8, mode, clip_acts, eltwise_mult_stats)
    model.eval()
    output = model(*inputs)

    torch.testing.assert_allclose(output, expected_output)


###############################################################################
# Test Element-Wise Addition
###############################################################################

@pytest.fixture()
def eltwise_add_stats():
    stats = OrderedDict()
    stats['inputs'] = input_stats
    stats['output'] = {'min': -108, 'max': 66, 'avg_min': -60.5, 'avg_max': 51.5}
    return stats


@pytest.mark.parametrize(
    "mode, clip_acts, expected_output",
    [
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.NONE,
         torch.tensor([[[[-12.96470588, 36.16470588], [4.776470588, 17.74117647]],
                        [[4.776470588, 17.74117647], [-9.552941176, 7.505882353]]],
                       [[[-107.8117647, 65.50588235], [-9.552941176, 0]],
                        [[60.04705882, -16.37647059], [-8.188235294, 2.729411765]]]])),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, ClipMode.AVG,
         torch.tensor([[[[-13.17647059, 29.86666667], [4.831372549, 18.00784314]],
                        [[4.831372549, 18.00784314], [-10.10196078, 8.345098039]]],
                       [[[-60.61176471, 51.38823529], [-8.784313725, 0]],
                        [[29.86666667, -17.12941176], [-7.905882353, 3.074509804]]]])),
        (LinearQuantMode.SYMMETRIC, ClipMode.NONE,
         torch.tensor([[[[-13.60629921, 36.56692913], [5.102362205, 17.85826772]],
                        [[5.102362205, 17.85826772], [-9.354330709, 7.653543307]]],
                       [[[-108, 66.33070866], [-9.354330709, 0]],
                        [[59.52755906, -17.00787402], [-8.503937008, 3.401574803]]]])),
        (LinearQuantMode.SYMMETRIC, ClipMode.AVG,
         torch.tensor([[[[-12.86220472, 29.05905512], [4.763779528, 18.1023622]],
                        [[4.763779528, 18.1023622], [-10.00393701, 8.098425197]]],
                       [[[-60.97637795, 60.5], [-9.051181102, 0]],
                        [[33.34645669, -17.1496063], [-8.098425197, 2.858267717]]]]))
    ]
)
def test_eltwise_add_layer_wrapper(inputs, eltwise_add_stats, mode, clip_acts, expected_output):
    with pytest.raises(ValueError):
        # Check exception on wrong layer type
        RangeLinearQuantEltwiseAddWrapper(torch.nn.Module(), 8, mode, clip_acts, test_eltwise_add_layer_wrapper)

    layer = distiller.modules.EltwiseAdd()

    with pytest.raises(NotImplementedError):
        # Check exception on no stats
        RangeLinearQuantEltwiseAddWrapper(layer, 8, mode, clip_acts, activation_stats=None)

    model = RangeLinearQuantEltwiseAddWrapper(layer, 8, mode, clip_acts, eltwise_add_stats)
    model.eval()
    output = model(*inputs)

    torch.testing.assert_allclose(output, expected_output)


###############################################################################
# Test Clipping Overrides
###############################################################################

class DummyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DummyRNN, self).__init__()
        self.rnn = distiller.modules.DistillerLSTM(input_size, hidden_size, num_layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, h=None):
        y, h = self.rnn(x, h)
        y = self.softmax(y)
        return y, h


@pytest.fixture()
def rnn_model():
    return DummyRNN(20, 128, 1)


@pytest.fixture()
def rnn_model_stats(rnn_model):
    collector = QuantCalibrationStatsCollector(rnn_model)
    dummy_input = torch.randn(35, 50, 20)
    with collector_context(collector):
        y, h = rnn_model(dummy_input)
    return collector.value()

# This warning is a PyTorch bug, to be fixed in a future release (https://github.com/pytorch/pytorch/pull/20026)
@pytest.mark.filterwarnings('ignore:new_zeros is a legacy constructor and is not supported in the JIT')
# The next 2 warning are the result of the  LSTM implementation iterating over tensors, which the PyTorch tracing
# mechanism doesn't like. Since the tracing done within PostTrainLinearQuantizer always uses the same input, there
# is no actual problem and we can ignore the warnings.
@pytest.mark.filterwarnings('ignore:Iterating over a tensor might cause the trace to be incorrect')
@pytest.mark.filterwarnings('ignore:Converting a tensor to a Python index might cause the trace to be incorrect')
@pytest.mark.parametrize(
    "overrides, e_clip_acts, e_n_stds",
    [
        (None, ClipMode.AVG, 0),

        (distiller.utils.yaml_ordered_load("""
        rnn.cells.0.eltwisemult_hidden:
            clip_acts: NONE
        """), ClipMode.NONE, 0),

        (distiller.utils.yaml_ordered_load("""
        rnn.cells.0.eltwisemult_hidden:
            clip_acts: N_STD
            clip_n_stds: 2
        """), ClipMode.N_STD, 2)
    ]
)
def test_override_no_clip(overrides, e_clip_acts, e_n_stds, rnn_model, rnn_model_stats):
    quantizer = PostTrainLinearQuantizer(rnn_model, clip_acts="AVG", clip_n_stds=0, overrides=overrides,
                                         model_activation_stats=rnn_model_stats)
    quantizer.prepare_model(torch.randn(1, 1, 20))
    assert isinstance(quantizer.model.rnn.cells[0].eltwisemult_hidden, RangeLinearQuantEltwiseMultWrapper)
    assert quantizer.model.rnn.cells[0].eltwisemult_hidden.clip_acts == e_clip_acts
    assert quantizer.model.rnn.cells[0].eltwisemult_hidden.clip_n_stds == e_n_stds


###############################################################################
# Stats Fusion Testing Utilities
###############################################################################

def stats_entry(min, max, min_avg, max_avg, mean, std):
    return OrderedDict([('min', min), ('max', max),
                        ('avg_min', min_avg), ('avg_max', max_avg),
                        ('mean', mean), ('std', std)])


def gen_stats_for_model(model):
    def gen_entry():
        entry = OrderedDict()
        a, b = random.uniform(-10, 10), random.uniform(-10, 10)
        entry['min'] = min(a, b)
        entry['max'] = max(a, b)
        c, d = random.uniform(a, b), random.uniform(a, b)
        entry['avg_min'] = min(c, d)
        entry['avg_max'] = max(c, d)
        entry['mean'] = (c + d) / 2.
        entry['std'] = random.random()
        return entry

    stats = OrderedDict()
    last = None
    for n, m in model.named_modules():
        if distiller.has_children(m):
            continue
        curr_stats = OrderedDict()
        curr_stats['inputs'] = OrderedDict()
        curr_stats['inputs'][0] = deepcopy(last['output']) if last else gen_entry()
        curr_stats['output'] = gen_entry()
        stats[n] = curr_stats
        last = curr_stats
    return stats


###############################################################################
# Test Stats Fusion - No Fusion
###############################################################################

# This warning seems to be a bug in batch_norm implementation, which compares a tensor to the value 1
@pytest.mark.filterwarnings('ignore:Converting a tensor to a Python boolean might cause the trace to be incorrect')
@pytest.mark.parametrize(
    'model, input_shape',
    [
        (WrappedSequential(nn.ReLU(), nn.BatchNorm1d(5)), (10, 5)),
        (WrappedSequential(nn.Conv2d(10, 20, 3), nn.BatchNorm2d(20, track_running_stats=False)), (10, 10, 50, 50)),
        (WrappedSequential(nn.Linear(10, 20), nn.BatchNorm1d(20, track_running_stats=False)), (10, 10)),
        (WrappedSequential(nn.Conv2d(10, 20, 3), nn.MaxPool2d(2)), (10, 10, 50, 50)),
    ],
    ids=['relu->bn', 'conv->bn_no_stats', 'linear->bn_no_stats', 'conv->pool']
)
def test_stats_fusion_no_fuse(model, input_shape):
    stats = gen_stats_for_model(model)
    quantizer = PostTrainLinearQuantizer(model, model_activation_stats=deepcopy(stats))
    quantizer.prepare_model(torch.randn(input_shape))
    assert quantizer.model_activation_stats == stats


###############################################################################
# Test Stats Fusion - No Activation
###############################################################################

class ConvBnActPool(nn.Module):
    def __init__(self, act_type, act_as_module):
        super(ConvBnActPool, self).__init__()
        self.conv = nn.Conv2d(10, 20, 3)
        self.bn = nn.BatchNorm2d(20)
        self.act_type = act_type
        self.act_as_module = act_as_module
        if act_type is not None:
            if act_as_module:
                self.act = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid()}[act_type]
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act_type is not None:
            if self.act_as_module:
                x = self.act(x)
            else:
                f = {'relu': F.relu, 'tanh': torch.tanh, 'sigmoid': torch.sigmoid}[self.act_type]
                x = f(x)
        x = self.pool(x)
        return x


def test_stats_fusion_just_bn():
    model = ConvBnActPool(None, False)
    stats = gen_stats_for_model(model)
    quantizer = PostTrainLinearQuantizer(model, model_activation_stats=deepcopy(stats))
    quantizer.prepare_model(torch.randn((10, 10, 20, 20)))

    expected = deepcopy(stats)
    expected.pop('bn')  # After BN folding BN stats are removed
    expected['conv']['output'] = deepcopy(stats['bn']['output'])
    assert quantizer.model_activation_stats == expected


###############################################################################
# Test Stats Fusion - Sequential, single activation
###############################################################################

@pytest.mark.parametrize(
    'act_type, act_as_module, bn_out_stats, conv_out_expected_stats',
    [
        ('relu', True, stats_entry(-5., 5., -3., 3., 0., 0.5), None),
        ('relu', False, stats_entry(-5., 5., -3., 3., 0., 0.5), stats_entry(0., 5., 0, 3., 0., 0.5)),
        ('relu', False, stats_entry(1., 5., 2., 3., 2.5, 0.5), stats_entry(1., 5., 2., 3., 2.5, 0.5)),
        ('relu', False, stats_entry(-5., -1., -4., -2., -2.5, 0.5), stats_entry(0., 0, 0, 0., -2.5, 0.5)),
        ('tanh', True, stats_entry(-5., 5., -3., 3., 0., 0.5), stats_entry(-4., 4., -3., 3., 0., 0.5)),
        ('tanh', False, stats_entry(-6., 3., -5., 1., 0., 0.5), stats_entry(-4., 3., -4., 1., 0., 0.5)),
        ('tanh', False, stats_entry(1., 6., 2., 3., 2.5, 0.5), stats_entry(1., 4., 2., 3., 2.5, 0.5)),
        ('tanh', False, stats_entry(-2., 3., -1., 2., 0, 0.5), stats_entry(-2., 3., -1., 2., 0, 0.5)),
        ('sigmoid', True, stats_entry(-8., 8., -7., 7., 0., 0.5), stats_entry(-6., 6., -6., 6., 0., 0.5)),
        ('sigmoid', False, stats_entry(-8., 3., -7., 1., 0., 0.5), stats_entry(-6., 3., -6., 1., 0., 0.5)),
        ('sigmoid', False, stats_entry(1., 8., 2., 3., 2.5, 0.5), stats_entry(1., 6., 2., 3., 2.5, 0.5)),
        ('sigmoid', False, stats_entry(-2., 3., -1., 2., 0, 0.5), stats_entry(-2., 3., -1., 2., 0, 0.5)),
    ],
    ids=['relu_as_module', 'relu_pos_neg', 'relu_all_pos', 'relu_all_neg',
         'tanh_as_module_all_out', 'tanh_min_out', 'tanh_max_out', 'tanh_all_in',
         'sigmoid_as_module_all_out', 'sigmoid_min_out', 'sigmoid_max_out', 'sigmoid_all_in']
)
def test_stats_fusion_sequential(act_type, act_as_module, bn_out_stats, conv_out_expected_stats):
    model = ConvBnActPool(act_type, act_as_module)
    stats = gen_stats_for_model(model)
    stats['bn']['output'] = bn_out_stats
    quantizer = PostTrainLinearQuantizer(model, model_activation_stats=deepcopy(stats))
    quantizer.prepare_model(torch.randn((10, 10, 20, 20)))

    expected = deepcopy(stats)
    expected.pop('bn')  # After BN folding BN stats are removed
    if act_type == 'relu':
        if act_as_module:
            expected['conv']['output'] = deepcopy(stats['act']['output'])
            expected['act']['inputs'][0] = deepcopy(stats['act']['output'])
        else:
            expected['conv']['output'] = conv_out_expected_stats
    else:
        expected['conv']['output'] = conv_out_expected_stats
        if act_as_module:
            expected['act']['inputs'][0] = conv_out_expected_stats

    assert quantizer.model_activation_stats == expected


###############################################################################
# Test Stats Fusion - Split before activation
###############################################################################

class LinearBNSplitAct(nn.Module):
    def __init__(self, act1_type, act2_type):
        super(LinearBNSplitAct, self).__init__()
        self.linear = nn.Linear(10, 40)
        self.bn = nn.BatchNorm1d(40)
        acts_map = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}
        self.act1 = acts_map[act1_type]()
        self.act2 = acts_map[act2_type]()

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        t1, t2 = x.chunk(2, dim=1)
        a1 = self.act1(t1)
        a2 = self.act2(t2)
        return a1 + a2


@pytest.mark.parametrize(
    'act1_type, act2_type, bn_out_stats, linear_out_expected_stats',
    [
        ('relu', 'relu', stats_entry(-5., 5., -3., 3., 0., 0.5), stats_entry(0., 5., 0, 3., 0., 0.5)),
        ('relu', 'sigmoid', stats_entry(-5., 5., -3., 3., 0., 0.5), stats_entry(-5., 5., -3., 3., 0., 0.5)),
        ('relu', 'tanh', stats_entry(-5., 5., -3., 3., 0., 0.5), stats_entry(-5., 5., -3., 3., 0., 0.5)),
        ('tanh', 'tanh', stats_entry(-5., 5., -3., 3., 0., 0.5), stats_entry(-4., 4., -3., 3., 0., 0.5)),
        ('tanh', 'sigmoid', stats_entry(-8., 8., -7., 7., 0., 0.5), stats_entry(-6., 6., -6., 6., 0., 0.5)),
        ('sigmoid', 'sigmoid', stats_entry(-8., 8., -7., 7., 0., 0.5), stats_entry(-6., 6., -6., 6., 0., 0.5))
    ],
    ids=['relu-relu', 'relu-sigmoid', 'relu-tanh', 'tanh-tanh', 'tanh-sigmoid', 'sigmoid-sigmoid']
)
def test_stats_fusion_split_act(act1_type, act2_type, bn_out_stats, linear_out_expected_stats):
    model = LinearBNSplitAct(act1_type, act2_type)
    stats = gen_stats_for_model(model)
    stats['bn']['output'] = bn_out_stats
    quantizer = PostTrainLinearQuantizer(model, model_activation_stats=deepcopy(stats))
    quantizer.prepare_model(torch.randn(10, 10))

    expected = deepcopy(stats)
    expected.pop('bn')  # After BN folding BN stats are removed
    expected['linear']['output'] = linear_out_expected_stats
    assert quantizer.model_activation_stats == expected

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

import os
import sys
import pytest
import torch
import torch.testing

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from distiller.quantization import RangeLinearQuantParamLayerWrapper, LinearQuantMode


@pytest.fixture()
def conv_input():
    return torch.cat((torch.tensor([[[[-7, 5], [2, -3]]]], dtype=torch.float64),
                      torch.tensor([[[[-15, 10], [-1, 5]]]], dtype=torch.float64)), 0)


@pytest.fixture()
def conv_weights():
    return torch.tensor([[[[-1, -0.5, 0], [0.5, 1, 1.5], [2, 2.5, 3]]],
                         [[[-0.3, -0.25, -0.2], [-0.15, -0.1, -0.05], [0, 0.05, 0.1]]]], dtype=torch.float64)


@pytest.mark.parametrize(
    "mode, clip_acts, per_channel_wts, expected_output",
    [
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, False, False,
         torch.cat((torch.tensor([[[[-3.648135333, -2.14596196], [0.858384784, 2.432090222]],
                                   [[0.214596196, 0.500724457], [0.715320653, 0.786852719]]]], dtype=torch.float64),
                    torch.tensor([[[[12.51811144, 13.01883589], [14.0918168, 14.59254133]],
                                   [[1.359109242, 1.645237503], [1.573705438, 1.645237503]]]], dtype=torch.float64)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, True, False,
         torch.cat((torch.tensor([[[[-1.089218234, -1.089218234], [1.055180164, 2.518817167]],
                                   [[0.238266489, 0.476532978], [0.680761396, 0.782875606]]]], dtype=torch.float64),
                    torch.tensor([[[[7.59048957, 7.59048957], [7.59048957, 7.59048957]],
                                   [[1.123256304, 1.259408583], [1.089218234, 1.089218234]]]], dtype=torch.float64)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, False, True,
         torch.cat((torch.tensor([[[[-3.648135333, -2.14596196], [0.858384784, 2.432090222]],
                                   [[0.214596196, 0.429192392], [0.715320653, 0.858384784]]]], dtype=torch.float64),
                    torch.tensor([[[[12.51811144, 13.01883589], [14.09181687, 14.59254133]],
                                   [[1.430641307, 1.502173372], [1.573705438, 1.645237503]]]], dtype=torch.float64)),
                   dim=0)
         ),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, True, True,
         torch.cat((torch.tensor([[[[-1.089768056, -1.089768056], [1.055712804, 2.52008863]],
                                   [[0.238386762, 0.408663021], [0.681105035, 0.817326042]]]], dtype=torch.float64),
                    torch.tensor([[[[7.59432114, 7.59432114], [7.59432114, 7.59432114]],
                                   [[1.191933811, 1.15787856], [1.123823308, 1.089768056]]]], dtype=torch.float64)),
                   dim=0)
         )
    ]
)
def test_conv_layer_wrapper(conv_input, conv_weights, mode, clip_acts, per_channel_wts, expected_output):
    layer = torch.nn.Conv2d(conv_input.shape[1], expected_output.shape[1], conv_weights.shape[-1],
                            padding=1, bias=False)
    layer.weight.data = conv_weights

    model = RangeLinearQuantParamLayerWrapper(layer, 8, 8, mode=mode, clip_acts=clip_acts,
                                              per_channel_wts=per_channel_wts)

    with pytest.raises(RuntimeError):
        model(conv_input)

    model.eval()

    output = model(conv_input)

    torch.testing.assert_allclose(output, expected_output)


@pytest.fixture()
def linear_input():
    return torch.tensor([[-7, 5, 2, -3]], dtype=torch.float64)


@pytest.fixture()
def linear_weights():
    return torch.tensor([[-1, 0.5, 0, 0.5],
                         [-0.05, 0, 0.05, 0.1],
                         [0.3, 0.6, -0.1, -0.2]], dtype=torch.float64)


@pytest.fixture()
def linear_bias():
    return torch.tensor([-0.3, 0.1, -0.5], dtype=torch.float64)


@pytest.mark.parametrize(
    "mode, clip_acts, per_channel_wts, expected_output",
    [
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, False, False,
         torch.tensor([[7.698556917, 0.262450804, 0.787352412]], dtype=torch.float64)),
        (LinearQuantMode.ASYMMETRIC_UNSIGNED, False, True,
         torch.tensor([[7.71233218, 0.262920415, 0.788761246]], dtype=torch.float64))
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

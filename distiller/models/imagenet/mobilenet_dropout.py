#
# Copyright (c) 2019 Intel Corporation
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

"""Source: https://github.com/mit-han-lab/amc-compressed-models/blob/master/models/mobilenet_v1.py

The code has been modified to remove code related to AMC.
"""

__all__ = ['mobilenet_v1_dropout']


import torch
import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileNet(nn.Module):
    def __init__(self, n_class=1000, channel_multiplier=1.):
        super(MobileNet, self).__init__()
        in_planes = int(32 * channel_multiplier)
        a = int(64 * channel_multiplier)
        cfg = [a, (a*2, 2), a*2, (a*4, 2), a*4, (a*8, 2), a*8, a*8, a*8, a*8, a*8, (a*16, 2), a*16]

        self.conv1 = conv_bn(3, in_planes, stride=2)
        self.features = self._make_layers(in_planes, cfg, conv_dw)
        #self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(cfg[-1], n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = x.mean(3).mean(2)  # global average pooling
        #x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _make_layers(self, in_planes, cfg, layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v1_dropout_025():
    return MobileNet(channel_multiplier=0.25)


def mobilenet_v1_dropout_050():
    return MobileNet(channel_multiplier=0.5)


def mobilenet_v1_dropout_075():
    return MobileNet(channel_multiplier=0.75)


def mobilenet_v1_dropout():
    return MobileNet()

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
AlexNet model with batch-norm layers.
Model configuration based on the AlexNet DoReFa example in TensorPack:
https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/alexnet-dorefa.py

Code based on the AlexNet PyTorch sample, with the required changes.
"""

import math
import torch.nn as nn

__all__ = ['AlexNetBN', 'alexnet_bn']


class AlexNetBN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetBN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=12, stride=4),                           # conv0 (224x224x3) -> (54x54x96)
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2, bias=False),   # conv1 (54x54x96)  -> (54x54x256)
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.9),                          # bn1   (54x54x256)
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),                # pool1 (54x54x256) -> (27x27x256)
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),            # conv2 (27x27x256) -> (27x27x384)
            nn.BatchNorm2d(384, eps=1e-4, momentum=0.9),                          # bn2   (27x27x384)
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),                     # pool2 (27x27x384) -> (14x14x384)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=False),  # conv3 (14x14x384) -> (14x14x384)
            nn.BatchNorm2d(384, eps=1e-4, momentum=0.9),                          # bn3   (14x14x384)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=False),  # conv4 (14x14x384) -> (14x14x256)
            nn.BatchNorm2d(256, eps=1e-4, momentum=0.9),                          # bn4   (14x14x256)
            nn.MaxPool2d(kernel_size=3, stride=2),                                # pool4 (14x14x256) -> (6x6x256)
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096, bias=False),       # fc0
            nn.BatchNorm1d(4096, eps=1e-4, momentum=0.9),   # bnfc0
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),              # fc1
            nn.BatchNorm1d(4096, eps=1e-4, momentum=0.9),   # bnfc1
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),                   # fct
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                fan_in, k_size = (m.in_channels, m.kernel_size[0] * m.kernel_size[1]) if isinstance(m, nn.Conv2d) \
                    else (m.in_features, 1)
                n = k_size * fan_in
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet_bn(**kwargs):
    r"""AlexNet model with batch-norm layers.
    Model configuration based on the AlexNet DoReFa example in `TensorPack
    <https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/alexnet-dorefa.py>`
    """
    model = AlexNetBN(**kwargs)
    return model

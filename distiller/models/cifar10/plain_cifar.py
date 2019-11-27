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

"""Plain for CIFAR10

Plain for CIFAR10, based on "Deep Residual Learning for Image Recognition".

@inproceedings{DBLP:conf/cvpr/HeZRS16,
  author    = {Kaiming He and
               Xiangyu Zhang and
               Shaoqing Ren and
               Jian Sun},
  title     = {Deep Residual Learning for Image Recognition},
  booktitle = {{CVPR}},
  pages     = {770--778},
  publisher = {{IEEE} Computer Society},
  year      = {2016}
}

"""
import torch.nn as nn
import math


__all__ = ['plain20_cifar', 'plain20_cifar_nobn']

NUM_CLASSES = 10


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, batch_norm=True):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes) if batch_norm else None
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if batch_norm else None
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out) if self.bn1 is not None else out
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out) if self.bn2 is not None else out
        out = self.relu2(out)
        return out


class PlainCifar(nn.Module):
    def __init__(self, block, blks_per_layer, num_classes=NUM_CLASSES, batch_norm=True):
        self.inplanes = 16
        super().__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes) if batch_norm else None
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, blks_per_layer[0], stride=1, batch_norm=batch_norm)
        self.layer2 = self._make_layer(block, 32, blks_per_layer[1], stride=2, batch_norm=batch_norm)
        self.layer3 = self._make_layer(block, 64, blks_per_layer[2], stride=2, batch_norm=batch_norm)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm=True):
        # Each layer is composed on 2*num_blocks blocks, and the first block usually
        # performs downsampling of the input, and doubling of the number of filters/feature-maps.
        blocks = []
        inplanes = self.inplanes
        # First block is special (downsamples and adds filters)
        blocks.append(block(inplanes, planes, stride, batch_norm=batch_norm))

        self.inplanes = planes * block.expansion
        for i in range(num_blocks - 1):
            blocks.append(block(self.inplanes, planes, stride=1, batch_norm=batch_norm))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def plain20_cifar(**kwargs):
    # Plain20 for CIFAR10
    model = PlainCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model
    #return plain20_cifar_nobn(**kwargs)


def plain20_cifar_nobn(**kwargs):
    # Plain20 for CIFAR10, without batch-normalization layers
    model = PlainCifar(BasicBlock, [3, 3, 3], batch_norm=False, **kwargs)
    return model
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

"""Pre-Activation ResNet for ImageNet

Pre-Activation ResNet for ImageNet, based on "Identity Mappings in Deep Residual Networks".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate changes for pre-activation.

@article{
  He2016,
  author = {Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun},
  title = {Identity Mappings in Deep Residual Networks},
  journal = {arXiv preprint arXiv:1603.05027},
  year = {2016}
}
"""

import torch.nn as nn
import math


__all__ = ['PreactResNet', 'preact_resnet18', 'preact_resnet34', 'preact_resnet50', 'preact_resnet101',
           'preact_resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, preactivate=True):
        super(PreactBasicBlock, self).__init__()
        self.pre_bn = self.pre_relu = None
        if preactivate:
            self.pre_bn = nn.BatchNorm2d(inplanes)
            self.pre_relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1_2 = nn.BatchNorm2d(planes)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.preactivate = preactivate

    def forward(self, x):
        if self.preactivate:
            preact = self.pre_bn(x)
            preact = self.pre_relu(preact)
        else:
            preact = x

        out = self.conv1(preact)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(preact)
        else:
            residual = x

        out += residual

        return out


class PreactBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, preactivate=True):
        super(PreactBottleneck, self).__init__()
        self.pre_bn = self.pre_relu = None
        if preactivate:
            self.pre_bn = nn.BatchNorm2d(inplanes)
            self.pre_relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(planes)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(planes)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.preactivate = preactivate

    def forward(self, x):
        if self.preactivate:
            preact = self.pre_bn(x)
            preact = self.pre_relu(preact)
        else:
            preact = x

        out = self.conv1(preact)
        out = self.bn1_2(out)
        out = self.relu1_2(out)

        out = self.conv2(out)
        out = self.bn2_3(out)
        out = self.relu2_3(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(preact)
        else:
            residual = x

        out += residual

        return out


class PreactResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(PreactResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.final_bn = nn.BatchNorm2d(512 * block.expansion)
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        # On the first residual block in the first residual layer we don't pre-activate,
        # because we take care of that (+ maxpool) after the initial conv layer
        preactivate_first = stride != 1

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, preactivate_first))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preact_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = PreactResNet(PreactBasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def preact_resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = PreactResNet(PreactBasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def preact_resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = PreactResNet(PreactBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def preact_resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = PreactResNet(PreactBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def preact_resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = PreactResNet(PreactBottleneck, [3, 8, 36, 3], **kwargs)
    return model

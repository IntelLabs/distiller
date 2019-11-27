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

"""Pre-Activation ResNet for CIFAR10

Pre-Activation ResNet for CIFAR10, based on "Identity Mappings in Deep Residual Networks".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for pre-activation and the 10-class Cifar-10 dataset.
This ResNet also has layer gates, to be able to dynamically remove layers.

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

__all__ = ['preact_resnet20_cifar', 'preact_resnet32_cifar', 'preact_resnet44_cifar', 'preact_resnet56_cifar',
           'preact_resnet110_cifar', 'preact_resnet20_cifar_conv_ds', 'preact_resnet32_cifar_conv_ds',
           'preact_resnet44_cifar_conv_ds', 'preact_resnet56_cifar_conv_ds', 'preact_resnet110_cifar_conv_ds']

NUM_CLASSES = 10


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PreactBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None, preact_downsample=True):
        super(PreactBasicBlock, self).__init__()
        self.block_gates = block_gates
        self.pre_bn = nn.BatchNorm2d(inplanes)
        self.pre_relu = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.preact_downsample = preact_downsample

    def forward(self, x):
        need_preact = self.block_gates[0] or self.block_gates[1] or self.downsample and self.preact_downsample
        if need_preact:
            preact = self.pre_bn(x)
            preact = self.pre_relu(preact)
            out = preact
        else:
            preact = out = x

        if self.block_gates[0]:
            out = self.conv1(out)
            out = self.bn(out)
            out = self.relu(out)

        if self.block_gates[1]:
            out = self.conv2(out)

        if self.downsample is not None:
            if self.preact_downsample:
                residual = self.downsample(preact)
            else:
                residual = self.downsample(x)
        else:
            residual = x

        out += residual

        return out


class PreactResNetCifar(nn.Module):
    def __init__(self, block, layers, num_classes=NUM_CLASSES, conv_downsample=False):
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.inplanes = 16  # 64
        super(PreactResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0],
                                       conv_downsample=conv_downsample)
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2,
                                       conv_downsample=conv_downsample)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2,
                                       conv_downsample=conv_downsample)
        self.final_bn = nn.BatchNorm2d(64 * block.expansion)
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1, conv_downsample=False):
        downsample = None
        outplanes = planes * block.expansion
        if stride != 1 or self.inplanes != outplanes:
            if conv_downsample:
                downsample = nn.Conv2d(self.inplanes, outplanes,
                                       kernel_size=1, stride=stride, bias=False)
            else:
                # Identity downsample uses strided average pooling + padding instead of convolution
                pad_amount = int(self.inplanes / 2)
                downsample = nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.ConstantPad3d((0, 0, 0, 0, pad_amount, pad_amount), 0)
                )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample, conv_downsample))
        self.inplanes = outplanes
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preact_resnet20_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [3, 3, 3], **kwargs)
    return model


def preact_resnet32_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [5, 5, 5], **kwargs)
    return model


def preact_resnet44_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [7, 7, 7], **kwargs)
    return model


def preact_resnet56_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [9, 9, 9], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet182_cifar(**kwargs):
    model = PreactResNetCifar(PreactBasicBlock, [30, 30, 30], **kwargs)
    return model


def preact_resnet20_cifar_conv_ds(**kwargs):
    return preact_resnet20_cifar(conv_downsample=True)


def preact_resnet32_cifar_conv_ds(**kwargs):
    return preact_resnet32_cifar(conv_downsample=True)


def preact_resnet44_cifar_conv_ds(**kwargs):
    return preact_resnet44_cifar(conv_downsample=True)


def preact_resnet56_cifar_conv_ds(**kwargs):
    return preact_resnet56_cifar(conv_downsample=True)


def preact_resnet110_cifar_conv_ds(**kwargs):
    return preact_resnet110_cifar(conv_downsample=True)


def preact_resnet182_cifar_conv_ds(**kwargs):
    return preact_resnet182_cifar(conv_downsample=True)

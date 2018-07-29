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

"""Resnet for CIFAR10

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.
This ResNet also has layer gates, to be able to dynamically remove layers.

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
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from .resnet_cifar import BasicBlock
from .resnet_cifar import ResNetCifar


__all__ = ['resnet20_cifar_earlyexit', 'resnet32_cifar_earlyexit', 'resnet44_cifar_earlyexit',
    'resnet56_cifar_earlyexit', 'resnet110_cifar_earlyexit', 'resnet1202_cifar_earlyexit']

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResNetCifarEarlyExit(ResNetCifar):

    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        super(ResNetCifarEarlyExit, self).__init__(block, layers, num_classes)

        # Define early exit layers
        self.linear_exit0 = nn.Linear(1600, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)

        # Add early exit layers
        exit0 = nn.functional.avg_pool2d(x, 3)
        exit0 = exit0.view(exit0.size(0), -1)
        exit0 = self.linear_exit0(exit0)

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # return a list of probabilities
        output = []
        output.append(exit0)
        output.append(x)
        return output


def resnet20_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet32_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet56_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet110_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet1202_cifar_earlyexit(**kwargs):
    model = ResNetCifarEarlyExit(BasicBlock, [200, 200, 200], **kwargs)
    return model
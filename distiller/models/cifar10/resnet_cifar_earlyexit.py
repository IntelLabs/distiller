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


class ExitBranch(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(3)
        self.linear = nn.Linear(1600, num_classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class BranchPoint(nn.Module):
    def __init__(self, original_m, exit_m):
        super().__init__()
        self.original_m = original_m
        self.exit_m = exit_m
        self.output = None

    def forward(self, x):
        x1 = self.original_m.forward(x)
        x2 = self.exit_m.forward(x1)
        self.output = x2
        return x1


class ResNetCifarEarlyExit(ResNetCifar):
    def __init__(self, block, layers, num_classes=NUM_CLASSES):
        super().__init__(block, layers, num_classes)

        # Define early exit branches and install them
        self.exit_branch = ExitBranch(num_classes)
        self.layer1 = BranchPoint(self.layer1, self.exit_branch)

    def forward(self, x):
        x = super().forward(x)

        # return a list of probabilities
        exit0 = self.layer1.output
        output = (exit0, x)
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
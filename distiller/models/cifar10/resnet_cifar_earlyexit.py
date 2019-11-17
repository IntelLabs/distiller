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

"""Resnet for CIFAR10 with Early Exit branches

Resnet for CIFAR10, based on "Deep Residual Learning for Image Recognition".
This is based on TorchVision's implementation of ResNet for ImageNet, with appropriate
changes for the 10-class Cifar-10 dataset.

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
from .resnet_cifar import BasicBlock
from .resnet_cifar import ResNetCifar
import torch.nn as nn
from distiller.modules import BranchPoint


__all__ = ['resnet20_cifar_earlyexit', 'resnet32_cifar_earlyexit', 'resnet44_cifar_earlyexit',
           'resnet56_cifar_earlyexit', 'resnet110_cifar_earlyexit', 'resnet1202_cifar_earlyexit']

NUM_CLASSES = 10

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def get_exits_def():
    exits_def = [('layer1.2.relu2', nn.Sequential(nn.AvgPool2d(3),
                            nn.Flatten(),
                            nn.Linear(1600, NUM_CLASSES)))]
    return exits_def


def find_module(model, mod_name):
    """Locate a module, given its full name"""
    for name, module in model.named_modules():
        if name == mod_name:
            return module
    return None


def split_module_name(mod_name):
    name_parts = mod_name.split('.')
    parent = '.'.join(name_parts[:-1])
    node = name_parts[-1]
    return parent, node


class ResNetCifarEarlyExit(ResNetCifar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exit_points = []
        self.attach_exits(get_exits_def())

    def attach_exits(self, exits_def):
        # For each exit point, we:
        # 1. Cache the name of the exit_point module (i.e. the name of the module
        #    whose output we forward to the exit branch).
        # 2. Override the exit_point module with an instance of BranchPoint
        for exit_point, exit_branch in exits_def:
            self.exit_points.append(exit_point)
            replaced_module = find_module(self, exit_point)
            parent_name, node_name = split_module_name(exit_point)
            parent_module = find_module(self, parent_name)
            parent_module.__setattr__(node_name, BranchPoint(replaced_module, exit_branch))

    def forward(self, x):
        # Run the input through the network
        x = super().forward(x)
        # Collect the outputs of all the exits and return them
        outputs = []
        for exit_point in self.exit_points:
            parent_name, node_name = split_module_name(exit_point)
            parent_module = find_module(self, parent_name)
            output = parent_module.__getattr__(node_name).output
            outputs.append(output)
        outputs += [x]
        return outputs


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
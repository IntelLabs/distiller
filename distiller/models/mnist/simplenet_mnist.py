#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""An implementation of a trivial MNIST model.
 
The original network definition is sourced here: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import torch.nn as nn
import torch.nn.functional as F


__all__ = ['simplenet_mnist', 'simplenet_v2_mnist']


class Simplenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(500, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


class Simplenet_v2(nn.Module):
    """
    This is Simplenet but with only one small Linear layer, instead of two Linear layers,
    one of which is large.
    26K parameters.
    python compress_classifier.py ${MNIST_PATH} --arch=simplenet_mnist --vs=0 --lr=0.01

    ==> Best [Top1: 98.970   Top5: 99.970   Sparsity:0.00   Params: 26000 on epoch: 54]
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(50, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def simplenet_mnist():
    model = Simplenet()
    return model

def simplenet_v2_mnist():
    model = Simplenet_v2()
    return model
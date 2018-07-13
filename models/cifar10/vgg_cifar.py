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

"""VGG for CIFAR10

VGG for CIFAR10, based on "Very Deep Convolutional Networks for Large-Scale
Image Recognition".
This is based on TorchVision's implementation of VGG for ImageNet, with
appropriate changes for the 10-class Cifar-10 dataset.
We replaced the three linear classifiers with a single one.
"""

import torch.nn as nn

__all__ = [
    'VGGCifar', 'vgg11_cifar', 'vgg11_bn_cifar', 'vgg13_cifar', 'vgg13_bn_cifar', 'vgg16_cifar', 'vgg16_bn_cifar',
    'vgg19_bn_cifar', 'vgg19_cifar',
]


class VGGCifar(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(VGGCifar, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_cifar(**kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGGCifar(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn_cifar(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGGCifar(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13_cifar(**kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGGCifar(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn_cifar(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGGCifar(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16_cifar(**kwargs):
    """VGG 16-layer model (configuration "D")
    """
    model = VGGCifar(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn_cifar(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGGCifar(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19_cifar(**kwargs):
    """VGG 19-layer model (configuration "E")
    """
    model = VGGCifar(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn_cifar(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGGCifar(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

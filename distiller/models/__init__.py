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

"""This package contains ImageNet and CIFAR image classification models for pytorch"""

import torch
import torchvision.models as torch_models
from . import cifar10 as cifar10_models
from . import imagenet as imagenet_extra_models

import logging
msglogger = logging.getLogger()

# ResNet special treatment: we have our own version of ResNet, so we need to over-ride
# TorchVision's version.
RESNET_SYMS = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

IMAGENET_MODEL_NAMES = sorted(name for name in torch_models.__dict__
                              if name.islower() and not name.startswith("__")
                              and callable(torch_models.__dict__[name]))
IMAGENET_MODEL_NAMES.extend(sorted(name for name in imagenet_extra_models.__dict__
                                   if name.islower() and not name.startswith("__")
                                   and callable(imagenet_extra_models.__dict__[name])))

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(), set(IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES)))


import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Conv2dWithMask(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):

        super(Conv2dWithMask, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.test_mask = None
        self.p_mask = 1.0
        self.frequency = 16

    def forward(self, input):
        if self.training:
            #prob = torch.distributions.binomial.Binomial(total_count=1, probs=[0.9]*self.out_channels)
            #mask = prob.sample()
            self.frequency -= 1
            if self.frequency == 0:
                sample = np.random.binomial(n=1, p=self.p_mask, size=self.out_channels)
                param = self.weight
                l1norm = param.detach().view(param.size(0), -1).norm(p=1, dim=1)
                mask = torch.tensor(sample)
                #print(mask.sum().item())

                mask = mask.expand(param.size(1) * param.size(2) * param.size(3), param.size(0)).t().contiguous()
                mask = mask.view(self.weight.shape).to(param.device)
                mask = mask.type(param.type())
                #print(mask.sum().item())
                #pruning_factor = self.p_mask
                masked_weights = self.weight * mask
                masked_l1norm = masked_weights.detach().view(param.size(0), -1).norm(p=1, dim=1)
                pruning_factor = (masked_l1norm.sum() / l1norm.sum()).item()
                # print(pruning_factor)
                pruning_factor = max(0.2, pruning_factor)
                weight = masked_weights / pruning_factor
                self.frequency = 16
            else:
                weight = self.weight
            #self.test_mask = mask
        # elif self.mask is not None:
        #     mask = self.mask.view(-1, 1, 1, 1)
        #     mask = mask.expand(self.weight.shape)
        #     mask = mask.to(self.weight.device)
        #     weight = self.weight * mask
        else:
            weight = self.weight# * self.test_mask

        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


# replaces all conv2d layers in target`s model with 'Conv2dWithMask'
def replace_conv2d(container):
    for name, module in container.named_children(): #for name, module in model.named_modules():
        if (isinstance(module, nn.Conv2d)):
            print("replacing: ", name)
            new_module = Conv2dWithMask(in_channels=module.in_channels,
                                        out_channels=module.out_channels,
                                        kernel_size=module.kernel_size, padding=module.padding,
                                        stride=module.stride, bias=module.bias)
            setattr(container, name, new_module)
        replace_conv2d(module)


def create_model(pretrained, dataset, arch, parallel=True, device_ids=None):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained: True is you wish to load a pretrained model.  Only torchvision models
          have a pretrained model.
        dataset:
        arch:
        parallel:
        device_ids: Devices on which model should be created -
            None - GPU if available, otherwise CPU
            -1 - CPU
            >=0 - GPU device IDs
    """
    msglogger.info('==> using %s dataset' % dataset)

    model = None
    if dataset == 'imagenet':
        str_pretrained = 'pretrained ' if pretrained else 
        msglogger.info("=> using %s%s model for ImageNet" % (str_pretrained, arch))
        assert arch in torch_models.__dict__ or arch in imagenet_extra_models.__dict__, \
            "Model %s is not supported for dataset %s" % (arch, 'ImageNet')
        if arch in RESNET_SYMS:
            model = imagenet_extra_models.__dict__[arch](pretrained=pretrained)
        elif arch in torch_models.__dict__:
            model = torch_models.__dict__[arch](pretrained=pretrained)
        else:
            assert not pretrained, "Model %s (ImageNet) does not have a pretrained model" % arch
            model = imagenet_extra_models.__dict__[arch]()
    elif dataset == 'cifar10':
        msglogger.info("=> creating %s model for CIFAR10" % arch)
        assert arch in cifar10_models.__dict__, "Model %s is not supported for dataset CIFAR10" % arch
        assert not pretrained, "Model %s (CIFAR10) does not have a pretrained model" % arch
        model = cifar10_models.__dict__[arch]()
    else:
        print("FATAL ERROR: create_model does not support models for dataset %s" % dataset)
        exit()

    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if (arch.startswith('alexnet') or arch.startswith('vgg')) and parallel:
            model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
        elif parallel:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = 'cpu'

    #replace_conv2d(model)
    return model.to(device)

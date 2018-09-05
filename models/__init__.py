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
import models.cifar10 as cifar10_models
import models.imagenet as imagenet_extra_models

import logging
msglogger = logging.getLogger()

IMAGENET_MODEL_NAMES = sorted(name for name in torch_models.__dict__
                              if name.islower() and not name.startswith("__")
                              and callable(torch_models.__dict__[name]))
IMAGENET_MODEL_NAMES.extend(sorted(name for name in imagenet_extra_models.__dict__
                                   if name.islower() and not name.startswith("__")
                                   and callable(imagenet_extra_models.__dict__[name])))

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))

ALL_MODEL_NAMES = sorted(set(IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES))


def create_model(pretrained, dataset, arch, parallel=True, device_ids=None):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained: True is you wish to load a pretrained model.  Only torchvision models
          have a pretrained model.
        dataset:
        arch:
        parallel:
    """
    msglogger.info('==> using %s dataset' % dataset)

    model = None
    if dataset == 'imagenet':
        str_pretrained = 'pretrained ' if pretrained else ''
        msglogger.info("=> using %s%s model for ImageNet" % (str_pretrained, arch))
        assert arch in torch_models.__dict__ or arch in imagenet_extra_models.__dict__, \
            "Model %s is not supported for dataset %s" % (arch, 'ImageNet')
        if arch in torch_models.__dict__:
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

    if (arch.startswith('alexnet') or arch.startswith('vgg')) and parallel:
        model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
    elif parallel:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    return model

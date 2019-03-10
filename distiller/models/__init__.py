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
import pretrainedmodels

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
IMAGENET_MODEL_NAMES.extend(pretrainedmodels.model_names)

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(),
                            set(IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES)))


def create_model(pretrained, dataset, arch, parallel=True, device_ids=None):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained: True is you wish to load a pretrained model.  Only torchvision models
          have a pretrained model.
        dataset: dataset name (only 'imagenet' and 'cifar10' are supported)
        arch: architecture name
        parallel (boolean): if set, use torch.nn.DataParallel
        device_ids: Devices on which model should be created -
            None - GPU if available, otherwise CPU
            -1 - CPU
            >=0 - GPU device IDs
    """
    msglogger.info('==> using %s dataset' % dataset)

    model = None
    dataset = dataset.lower()
    if dataset == 'imagenet':
        if arch in RESNET_SYMS:
            model = imagenet_extra_models.__dict__[arch](pretrained=pretrained)
        elif arch in torch_models.__dict__:
            model = torch_models.__dict__[arch](pretrained=pretrained)
        elif (arch in imagenet_extra_models.__dict__) and not pretrained:
            model = imagenet_extra_models.__dict__[arch]()
        elif arch in pretrainedmodels.model_names:
            model = pretrainedmodels.__dict__[arch](
                        num_classes=1000,
                        pretrained=(dataset if pretrained else None))
        else:
            error_message = ''
            if arch not in IMAGENET_MODEL_NAMES:
                error_message = "Model %s is not supported for dataset %s" % (arch, 'ImageNet')
            elif pretrained:
                error_message = "Model %s (ImageNet) does not have a pretrained model" % arch
            raise ValueError(error_message or 'Failed to find model {}'.format(arch))

        msglogger.info("=> using {p}{a} model for ImageNet".format(a=arch,
            p=('pretrained ' if pretrained else '')))
    elif dataset == 'cifar10':
        if pretrained:
            raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
        try:
            model = cifar10_models.__dict__[arch]()
        except KeyError:
            raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
        msglogger.info("=> creating %s model for CIFAR10" % arch)
    else:
        raise ValueError('Could not recognize dataset {}'.format(dataset))

    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if (arch.startswith('alexnet') or arch.startswith('vgg')) and parallel:
            model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
        elif parallel:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = 'cpu'

    return model.to(device)

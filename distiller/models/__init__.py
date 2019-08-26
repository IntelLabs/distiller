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

import copy

import torch
import torchvision.models as torch_models
from . import cifar10 as cifar10_models
from . import mnist as mnist_models
from . import imagenet as imagenet_extra_models
import pretrainedmodels

from distiller.utils import set_model_input_shape_attr

import logging
msglogger = logging.getLogger()

SUPPORTED_DATASETS = ('imagenet', 'cifar10', 'mnist')

# ResNet special treatment: we have our own version of ResNet, so we need to over-ride
# TorchVision's version.
RESNET_SYMS = ('ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')

TORCHVISION_MODEL_NAMES = sorted(
                            name for name in torch_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torch_models.__dict__[name]))

IMAGENET_MODEL_NAMES = copy.deepcopy(TORCHVISION_MODEL_NAMES)
IMAGENET_MODEL_NAMES.extend(sorted(name for name in imagenet_extra_models.__dict__
                                   if name.islower() and not name.startswith("__")
                                   and callable(imagenet_extra_models.__dict__[name])))
IMAGENET_MODEL_NAMES.extend(pretrainedmodels.model_names)

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))

MNIST_MODEL_NAMES = sorted(name for name in mnist_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(mnist_models.__dict__[name]))

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(),
                            set(IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES + MNIST_MODEL_NAMES)))

# A temporary monkey-patch to get past this Torchvision bug:
# https://github.com/pytorch/pytorch/issues/20516
from functools import partial
def patch_torchvision_mobilenet_v2_bug(model):
    def patched_forward(self, x):
        x = self.features(x)
        #x = x.mean([2, 3])
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x
    model.__class__.forward = patched_forward


_model_extensions = {}


def create_model(pretrained, dataset, arch, parallel=True, device_ids=None):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained [boolean]: True is you wish to load a pretrained model.
            Some models do not have a pretrained version.
        dataset: dataset name (only 'imagenet' and 'cifar10' are supported)
        arch: architecture name
        parallel [boolean]: if set, use torch.nn.DataParallel
        device_ids: Devices on which model should be created -
            None - GPU if available, otherwise CPU
            -1 - CPU
            >=0 - GPU device IDs
    """
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset {} is not supported'.format(dataset))

    model = None
    cadene = False
    try:
        if dataset == 'imagenet':
            model, cadene = _create_imagenet_model(arch, pretrained)
        elif dataset == 'cifar10':
            model = _create_cifar10_model(arch, pretrained)
        elif dataset == 'mnist':
            model = _create_mnist_model(arch, pretrained)
    except ValueError:
        if _is_registered_extension(arch, dataset, pretrained):
            model = _get_extension_model(arch, dataset)
        else:
            raise ValueError('Could not recognize dataset {} and model {} pair'.format(dataset, arch))

    msglogger.info("=> created a %s%s model with the %s dataset" % ('pretrained ' if pretrained else '',
                                                                     arch, dataset))
    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if parallel:
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            else:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        device = 'cpu'

    # Cache some attributes which describe the model
    _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene)
    model.arch = arch
    model.dataset = dataset
    model.is_parallel = parallel
    return model.to(device)


def _create_imagenet_model(arch, pretrained):
    dataset = "imagenet"
    cadene = False
    model = None
    if arch in RESNET_SYMS:
        model = imagenet_extra_models.__dict__[arch](pretrained=pretrained)
    elif arch in TORCHVISION_MODEL_NAMES:
        try:
            model = getattr(torch_models, arch)(pretrained=pretrained)
            if arch == "mobilenet_v2":
                patch_torchvision_mobilenet_v2_bug(model)
        except NotImplementedError:
            # In torchvision 0.3, trying to download a model that has no
            # pretrained image available will raise NotImplementedError
            if not pretrained:
                raise
    if model is None and (arch in imagenet_extra_models.__dict__) and not pretrained:
        model = imagenet_extra_models.__dict__[arch]()
    if model is None and (arch in pretrainedmodels.model_names):
        cadene = True
        model = pretrainedmodels.__dict__[arch](
            num_classes=1000,
            pretrained=(dataset if pretrained else None))
    if model is None:
        error_message = ''
        if arch not in IMAGENET_MODEL_NAMES:
            error_message = "Model {} is not supported for dataset ImageNet".format(arch)
        elif pretrained:
            error_message = "Model {} (ImageNet) does not have a pretrained model".format(arch)
        raise ValueError(error_message or 'Failed to find model {}'.format(arch))
    return model, cadene


def _create_cifar10_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = cifar10_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def _create_mnist_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (MNIST) does not have a pretrained model".format(arch))
    try:
        model = mnist_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset MNIST".format(arch))
    return model


def _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene):
    if cadene and pretrained:
        # When using pre-trained weights, Cadene models already have an input size attribute
        # We add the batch dimension to it
        input_size = model.module.input_size if isinstance(model, torch.nn.DataParallel) else model.input_size
        shape = tuple([1] + input_size)
        set_model_input_shape_attr(model, input_shape=shape)
    elif arch == 'inception_v3':
        set_model_input_shape_attr(model, input_shape=(1, 3, 299, 299))
    else:
        set_model_input_shape_attr(model, dataset=dataset)


def register_user_model(arch, dataset, model):
    """A simple mechanism to support models that are not part of distiller.models"""
    _model_extensions[(arch, dataset)] = model


def _is_registered_extension(arch, dataset, pretrained):
    try:
        return _model_extensions[(arch, dataset)]
    except KeyError:
        return None


def _get_extension_model(arch, dataset):
    return _model_extensions[(arch, dataset)]
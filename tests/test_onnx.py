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

import logging
import tempfile

import distiller

import pytest
import common  # common test code


# Logging configuration
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)


@pytest.mark.parametrize('arch',
    ['resnet18', 'resnet20_cifar', 'alexnet', 'vgg19', 'resnext101_32x4d'])
@pytest.mark.parametrize('add_softmax', [True, False])
def test_summary(arch, add_softmax):
    dataset = 'cifar10' if arch.endswith('cifar') else 'imagenet'
    model, _ = common.setup_test(arch, dataset, parallel=True)

    with tempfile.NamedTemporaryFile() as f:
        distiller.export_img_classifier_to_onnx(model, f.name, dataset, add_softmax=add_softmax)

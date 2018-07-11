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
import os
import sys
import pytest
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from models import create_model
from apputils import load_checkpoint

def test_load():
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.INFO)

    model = create_model(False, 'cifar10', 'resnet20_cifar')
    model, compression_scheduler, start_epoch = load_checkpoint(model, '../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar')
    assert compression_scheduler is not None
    assert start_epoch == 180

def test_load_negative():
    with pytest.raises(FileNotFoundError):
        model = create_model(False, 'cifar10', 'resnet20_cifar')
        model, compression_scheduler, start_epoch = load_checkpoint(model, 'THIS_IS_AN_ERROR/checkpoint_trained_dense.pth.tar')

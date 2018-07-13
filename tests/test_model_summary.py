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
import torch
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller
import pytest
import common  # common test code
import apputils

# Logging configuration
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)


def test_png_generation():
    DATASET = "cifar10"
    ARCH = "resnet20_cifar"
    model, zeros_mask_dict = common.setup_test(ARCH, DATASET, parallel=True)
    # 2 different ways to create a PNG
    apputils.draw_img_classifier_to_file(model, 'model.png', DATASET, True)
    apputils.draw_img_classifier_to_file(model, 'model.png', DATASET, False)


def test_negative():
    DATASET = "cifar10"
    ARCH = "resnet20_cifar"
    model, zeros_mask_dict = common.setup_test(ARCH, DATASET, parallel=True)

    with pytest.raises(ValueError):
        # png is not a supported summary type, so we expect this to fail with a ValueError
        distiller.model_summary(model, what='png', dataset=DATASET)


def test_summary():
    DATASET = "cifar10"
    ARCH = "resnet20_cifar"
    model, zeros_mask_dict = common.setup_test(ARCH, DATASET, parallel=True)

    distiller.model_summary(model, what='sparsity', dataset=DATASET)
    distiller.model_summary(model, what='compute', dataset=DATASET)
    distiller.model_summary(model, what='model', dataset=DATASET)
    distiller.model_summary(model, what='modules', dataset=DATASET)

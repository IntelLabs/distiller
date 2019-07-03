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
import distiller
import pytest
import os
import common  # common test code


# Logging configuration
logging.basicConfig(level=logging.INFO)
fh = logging.FileHandler('test.log')
logger = logging.getLogger()
logger.addHandler(fh)


SUMMARY_CHOICES = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params']


@pytest.mark.parametrize('display_param_nodes', [True, False])
def test_png_generation(display_param_nodes):
    dataset = "cifar10"
    arch = "resnet20_cifar"
    model, _ = common.setup_test(arch, dataset, parallel=True)
    # 2 different ways to create a PNG
    png_fname = os.path.join(common.PYTEST_COLLATERALS_DIR, 'model.png')
    distiller.draw_img_classifier_to_file(model, png_fname, dataset, display_param_nodes)


def test_compute_summary():
    dataset = "cifar10"
    arch = "simplenet_cifar"
    model, _ = common.setup_test(arch, dataset, parallel=True)
    df_compute = distiller.model_performance_summary(model, distiller.get_dummy_input(dataset))
    module_macs = df_compute.loc[:, 'MACs'].to_list()
    #                     [conv1,  conv2,  fc1,   fc2,   fc3]
    assert module_macs == [352800, 240000, 48000, 10080, 840]

    dataset = "imagenet"
    arch = "mobilenet"
    model, _ = common.setup_test(arch, dataset, parallel=True)
    df_compute = distiller.model_performance_summary(model, distiller.get_dummy_input(dataset))
    module_macs = df_compute.loc[:, 'MACs'].to_list()
    expected_macs = [10838016, 3612672, 25690112, 1806336, 25690112, 3612672, 51380224, 903168, 
                     25690112, 1806336, 51380224, 451584, 25690112, 903168, 51380224, 903168, 
                     51380224, 903168, 51380224, 903168, 51380224, 903168, 51380224, 225792, 
                     25690112, 451584, 51380224, 1024000]
    assert module_macs == expected_macs


@pytest.mark.parametrize('what', SUMMARY_CHOICES)
def test_summary(what):
    dataset = "cifar10"
    arch = "resnet20_cifar"
    model, _ = common.setup_test(arch, dataset, parallel=True)
    distiller.model_summary(model, what, dataset=dataset, logdir=common.PYTEST_COLLATERALS_DIR)


@pytest.mark.parametrize('what', SUMMARY_CHOICES)
def test_mnist(what):
    dataset = "mnist"
    arch = "simplenet_mnist"
    model, _ = common.setup_test(arch, dataset, parallel=True)
    distiller.model_summary(model, what, dataset=dataset, logdir=common.PYTEST_COLLATERALS_DIR)

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
import tempfile

import torch
import pytest
try:
    import distiller
except ImportError:
    module_path = os.path.abspath(os.path.join('..'))
    if module_path not in sys.path:
        sys.path.append(module_path)
    import distiller
import distiller
from distiller.apputils import save_checkpoint, load_checkpoint
from distiller.models import create_model


def test_load():
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.INFO)

    model = create_model(False, 'cifar10', 'resnet20_cifar')
    model, compression_scheduler, start_epoch = load_checkpoint(model, '../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar')
    assert compression_scheduler is not None
    assert start_epoch == 180


def test_load_state_dict():
    # prepare lean checkpoint
    state_dict_arrays = torch.load('../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar').get('state_dict')

    with tempfile.NamedTemporaryFile() as tmpfile:
        torch.save({'state_dict': state_dict_arrays}, tmpfile.name)
        model = create_model(False, 'cifar10', 'resnet20_cifar')
        model, compression_scheduler, start_epoch = load_checkpoint(model, tmpfile.name)

    assert len(list(model.named_modules())) >= len([x for x in state_dict_arrays if x.endswith('weight')]) > 0
    assert compression_scheduler is None
    assert start_epoch == 0


def test_load_dumb_checkpoint():
    # prepare lean checkpoint
    state_dict_arrays = torch.load('../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar').get('state_dict')

    with tempfile.NamedTemporaryFile() as tmpfile:
        torch.save(state_dict_arrays, tmpfile.name)
        model = create_model(False, 'cifar10', 'resnet20_cifar')
        with pytest.raises(ValueError):
            model, compression_scheduler, start_epoch = load_checkpoint(model, tmpfile.name)


def test_load_negative():
    with pytest.raises(FileNotFoundError):
        model = create_model(False, 'cifar10', 'resnet20_cifar')
        model, compression_scheduler, start_epoch = load_checkpoint(model, 'THIS_IS_AN_ERROR/checkpoint_trained_dense.pth.tar')


def test_load_gpu_model_on_cpu():
    # Issue #148
    CPU_DEVICE_ID = -1
    model = create_model(False, 'cifar10', 'resnet20_cifar', device_ids=CPU_DEVICE_ID)
    model, compression_scheduler, start_epoch = load_checkpoint(model,
                                                                '../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar')
    assert compression_scheduler is not None
    assert start_epoch == 180
    assert distiller.model_device(model) == 'cpu'


def test_load_gpu_model_on_cpu_with_thinning():
    # Issue #148
    # 1. create a GPU model and remove 50% of the filters in one of the layers (thninning)
    # 2. save the thinned model in a checkpoint file
    # 3. load the checkpoint and place it on the CPU
    CPU_DEVICE_ID = -1
    gpu_model = create_model(False, 'cifar10', 'resnet20_cifar')
    conv_pname = "module.layer1.0.conv1.weight"
    conv_p = distiller.model_find_param(gpu_model, conv_pname)
    pruner = distiller.pruning.L1RankedStructureParameterPruner("test_pruner", group_type="Filters",
                                                                desired_sparsity=0.5, weights=conv_pname)
    zeros_mask_dict = distiller.create_model_masks_dict(gpu_model)
    pruner.set_param_mask(conv_p, conv_pname, zeros_mask_dict, meta=None)

    # Use the mask to prune
    zeros_mask_dict[conv_pname].apply_mask(conv_p)
    distiller.remove_filters(gpu_model, zeros_mask_dict, 'resnet20_cifar', 'cifar10', optimizer=None)
    assert hasattr(gpu_model, 'thinning_recipes')
    scheduler = distiller.CompressionScheduler(gpu_model)
    save_checkpoint(epoch=0, arch='resnet20_cifar', model=gpu_model, scheduler=scheduler, optimizer=None)

    CPU_DEVICE_ID = -1
    cpu_model = create_model(False, 'cifar10', 'resnet20_cifar', device_ids=CPU_DEVICE_ID)
    load_checkpoint(cpu_model, "checkpoint.pth.tar")
    assert distiller.model_device(cpu_model) == 'cpu'


if __name__ == '__main__':
    test_load_gpu_model_on_cpu()

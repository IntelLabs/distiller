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

import torch
import pytest
import distiller
from distiller.apputils import save_checkpoint, load_checkpoint, load_lean_checkpoint
from distiller.models import create_model
import pretrainedmodels


def test_create_model_cifar():
    pretrained = False
    model = create_model(pretrained, 'cifar10', 'resnet20_cifar')
    with pytest.raises(ValueError):
        # only cifar _10_ is currently supported
        model = create_model(pretrained, 'cifar100', 'resnet20_cifar')
    with pytest.raises(ValueError):
        model = create_model(pretrained, 'cifar10', 'no_such_model!')

    pretrained = True
    with pytest.raises(ValueError):
        # no pretrained models of cifar10
        model = create_model(pretrained, 'cifar10', 'resnet20_cifar')


def test_create_model_imagenet():
    model = create_model(False, 'imagenet', 'alexnet')
    model = create_model(False, 'imagenet', 'resnet50')
    model = create_model(True, 'imagenet', 'resnet50')

    with pytest.raises(ValueError):
        model = create_model(False, 'imagenet', 'no_such_model!')


def test_create_model_pretrainedmodels():
    premodel_name = 'resnext101_32x4d'
    model = create_model(True, 'imagenet', premodel_name)

    with pytest.raises(ValueError):
        model = create_model(False, 'imagenet', 'no_such_model!')


def _is_similar_param_groups(opt_a, opt_b):
    for k in opt_a['param_groups'][0]:
        val_a = opt_a['param_groups'][0][k]
        val_b = opt_b['param_groups'][0][k]
        if (val_a != val_b) and (k != 'params'):
            return False
    return True


def test_load():
    logger = logging.getLogger('simple_example')
    logger.setLevel(logging.INFO)

    checkpoint_filename = 'checkpoints/resnet20_cifar10_checkpoint.pth.tar'
    src_optimizer_state_dict = torch.load(checkpoint_filename)['optimizer_state_dict']

    model = create_model(False, 'cifar10', 'resnet20_cifar', 0)
    model, compression_scheduler, optimizer, start_epoch = load_checkpoint(
        model, checkpoint_filename)
    assert compression_scheduler is not None
    assert optimizer is not None, 'Failed to load the optimizer'
    if not _is_similar_param_groups(src_optimizer_state_dict, optimizer.state_dict()):
        assert src_optimizer_state_dict == optimizer.state_dict() # this will always fail
    assert start_epoch == 1


def test_load_state_dict_implicit():
    # prepare lean checkpoint
    state_dict_arrays = torch.load('../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar').get('state_dict')

    with tempfile.NamedTemporaryFile() as tmpfile:
        torch.save({'state_dict': state_dict_arrays}, tmpfile.name)
        model = create_model(False, 'cifar10', 'resnet20_cifar')
        model, compression_scheduler, optimizer, start_epoch = load_checkpoint(model, tmpfile.name)

    assert compression_scheduler is None
    assert optimizer is None
    assert start_epoch == 0


def test_load_lean_checkpoint_1():
    # prepare lean checkpoint
    state_dict_arrays = torch.load('../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar').get('state_dict')

    with tempfile.NamedTemporaryFile() as tmpfile:
        torch.save({'state_dict': state_dict_arrays}, tmpfile.name)
        model = create_model(False, 'cifar10', 'resnet20_cifar')
        model, compression_scheduler, optimizer, start_epoch = load_checkpoint(
            model, tmpfile.name, lean_checkpoint=True)

    assert compression_scheduler is None
    assert optimizer is None
    assert start_epoch == 0


def test_load_lean_checkpoint_2():
    checkpoint_filename = '../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar'

    model = create_model(False, 'cifar10', 'resnet20_cifar', 0)
    model = load_lean_checkpoint(model, checkpoint_filename)


def test_load_dumb_checkpoint():
    # prepare lean checkpoint
    state_dict_arrays = torch.load('../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar').get('state_dict')

    with tempfile.NamedTemporaryFile() as tmpfile:
        torch.save(state_dict_arrays, tmpfile.name)
        model = create_model(False, 'cifar10', 'resnet20_cifar')
        with pytest.raises(ValueError):
            model, compression_scheduler, optimizer, start_epoch = load_checkpoint(model, tmpfile.name)


def test_load_negative():
    with pytest.raises(FileNotFoundError):
        model = create_model(False, 'cifar10', 'resnet20_cifar')
    with pytest.raises(FileNotFoundError):
        load_checkpoint(model, 'THIS_IS_AN_ERROR/checkpoint_trained_dense.pth.tar')


def test_load_gpu_model_on_cpu():
    # Issue #148
    CPU_DEVICE_ID = -1
    CPU_DEVICE_NAME = 'cpu'
    checkpoint_filename = 'checkpoints/resnet20_cifar10_checkpoint.pth.tar'

    model = create_model(False, 'cifar10', 'resnet20_cifar', device_ids=CPU_DEVICE_ID)
    model, compression_scheduler, optimizer, start_epoch = load_checkpoint(
        model, checkpoint_filename)

    assert compression_scheduler is not None
    assert optimizer is not None
    assert distiller.utils.optimizer_device_name(optimizer) == CPU_DEVICE_NAME
    assert start_epoch == 1
    assert distiller.model_device(model) == CPU_DEVICE_NAME


def test_load_gpu_model_on_cpu_lean_checkpoint():
    CPU_DEVICE_ID = -1
    CPU_DEVICE_NAME = 'cpu'
    checkpoint_filename = '../examples/ssl/checkpoints/checkpoint_trained_dense.pth.tar'

    model = create_model(False, 'cifar10', 'resnet20_cifar', device_ids=CPU_DEVICE_ID)
    model = load_lean_checkpoint(model, checkpoint_filename,
                                 model_device=CPU_DEVICE_NAME)
    assert distiller.model_device(model) == CPU_DEVICE_NAME


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
    save_checkpoint(epoch=0, arch='resnet20_cifar', model=gpu_model, scheduler=scheduler, optimizer=None,
        dir='checkpoints')

    CPU_DEVICE_ID = -1
    cpu_model = create_model(False, 'cifar10', 'resnet20_cifar', device_ids=CPU_DEVICE_ID)
    load_lean_checkpoint(cpu_model, "checkpoints/checkpoint.pth.tar")
    assert distiller.model_device(cpu_model) == 'cpu'

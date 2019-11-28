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
import os
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
    save_checkpoint(epoch=0, arch='resnet20_cifar', model=gpu_model,
                    scheduler=scheduler, optimizer=None, dir='checkpoints')

    CPU_DEVICE_ID = -1
    cpu_model = create_model(False, 'cifar10', 'resnet20_cifar', device_ids=CPU_DEVICE_ID)
    load_lean_checkpoint(cpu_model, "checkpoints/checkpoint.pth.tar")
    assert distiller.model_device(cpu_model) == 'cpu'


def test_validate_input_shape():
    with pytest.raises(ValueError):
        distiller.utils._validate_input_shape('', None)
    with pytest.raises(ValueError):
        distiller.utils._validate_input_shape('not_a_dataset', None)
    with pytest.raises(TypeError):
        distiller.utils._validate_input_shape('', 'non_numeric_shape')
    with pytest.raises(TypeError):
        distiller.utils._validate_input_shape('', ('blah', 2))
    with pytest.raises(TypeError):
        distiller.utils._validate_input_shape('', (1.5, 2))
    with pytest.raises(TypeError):
        # Mix "flattened" shape and tuple
        distiller.utils._validate_input_shape('', (1, 2, (3, 4)))

    s = distiller.utils._validate_input_shape('imagenet', None)
    assert s == (1, 3, 224, 224)
    s = distiller.utils._validate_input_shape('imagenet', (1, 2))
    assert s == (1, 3, 224, 224)
    s = distiller.utils._validate_input_shape('', (1, 2))
    assert s == (1, 2)
    s = distiller.utils._validate_input_shape('', ((1, 2), (3, 4)))
    assert s == ((1, 2), (3, 4))
    s = distiller.utils._validate_input_shape('', ((1, 2), ((3, 4), (5, 6))))
    assert s == ((1, 2), ((3, 4), (5, 6)))


@pytest.mark.parametrize('device', [None, 'cpu', 'cuda:0'])
def test_get_dummy_input(device):
    def check_shape_device(t, exp_shape, exp_device):
        assert t.shape == exp_shape
        assert str(t.device) == exp_device

    if device is None:
        expected_device = 'cpu'
    else:
        if 'cuda' in device and not torch.cuda.is_available():
            return
        expected_device = device

    with pytest.raises(ValueError):
        distiller.utils.get_dummy_input('', None)
    with pytest.raises(ValueError):
        distiller.utils.get_dummy_input(dataset='not_a_dataset')
    with pytest.raises(TypeError):
        distiller.utils.get_dummy_input(input_shape='non_numeric_shape')
    with pytest.raises(TypeError):
        distiller.utils.get_dummy_input(input_shape=('blah', 2))
    with pytest.raises(TypeError):
        distiller.utils.get_dummy_input(input_shape=(1.5, 2))
    with pytest.raises(TypeError):
        # Mix "flattened" shape and tuple
        distiller.utils.get_dummy_input(input_shape=(1, 2, (3, 4)))

    t = distiller.utils.get_dummy_input(dataset='imagenet', device=device)
    check_shape_device(t, (1, 3, 224, 224), expected_device)

    t = distiller.utils.get_dummy_input(dataset='imagenet', device=device, input_shape=(1, 2))
    check_shape_device(t, (1, 3, 224, 224), expected_device)

    shape = 1, 2
    t = distiller.utils.get_dummy_input(dataset='', device=device, input_shape=shape)
    check_shape_device(t, shape, expected_device)

    shape = ((1, 2), (3, 4))
    t = distiller.utils.get_dummy_input(device=device, input_shape=shape)
    assert isinstance(t, tuple)
    check_shape_device(t[0], shape[0], expected_device)
    check_shape_device(t[1], shape[1], expected_device)

    shape = ((1, 2), ((3, 4), (5, 6)))
    t = distiller.utils.get_dummy_input(device=device, input_shape=shape)
    assert isinstance(t, tuple)
    assert isinstance(t[0], torch.Tensor)
    assert isinstance(t[1], tuple)
    check_shape_device(t[0], shape[0], expected_device)
    check_shape_device(t[1][0], shape[1][0], expected_device)
    check_shape_device(t[1][1], shape[1][1], expected_device)


def test_load_checkpoint_without_model():
    checkpoint_filename = 'checkpoints/resnet20_cifar10_checkpoint.pth.tar'
    # Load a checkpoint w/o specifying the model: this should fail because the loaded
    # checkpoint is old and does not have the required metadata to create a model.
    with pytest.raises(ValueError):
        load_checkpoint(model=None, chkpt_file=checkpoint_filename)

    for model_device in (None, 'cuda', 'cpu'):
        # Now we create a new model, save a checkpoint, and load it w/o specifying the model.
        # This should succeed because the checkpoint has enough metadata to create model.
        model = create_model(False, 'cifar10', 'resnet20_cifar', 0)
        model, compression_scheduler, optimizer, start_epoch = load_checkpoint(model, checkpoint_filename)
        save_checkpoint(epoch=0, arch='resnet20_cifar', model=model, name='eraseme',
                        scheduler=compression_scheduler, optimizer=None, dir='checkpoints')
        temp_checkpoint = os.path.join("checkpoints", "eraseme_checkpoint.pth.tar")
        model, compression_scheduler, optimizer, start_epoch = load_checkpoint(model=None,
                                                                               chkpt_file=temp_checkpoint,
                                                                               model_device=model_device)
        assert compression_scheduler is not None
        assert optimizer is None
        assert start_epoch == 1
        assert model
        assert model.arch == "resnet20_cifar"
        assert model.dataset == "cifar10"
        os.remove(temp_checkpoint)


if __name__ == '__main__':
    test_load_gpu_model_on_cpu_with_thinning()
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

""" Helper code for checkpointing models, with support for saving the pruning schedule.

Adding the schedule information in the model checkpoint is helpful in resuming
a pruning session, or for querying the pruning schedule of a sparse model.
"""

import contextlib
import os
import shutil
from errno import ENOENT
import logging
from numbers import Number
from tabulate import tabulate
import torch
import distiller
from distiller.utils import normalize_module_name, getModuleFromModel, inferDatasetNameFromImageClassifierModel

msglogger = logging.getLogger()


def save_checkpoint(model, optimizer=None, compression_sched=None,
                    arch=None, dataset=None, is_best=False,
                    name=None, dir='.', file_ext='pth.tar',
                    **extras):
    """Save a pytorch training checkpoint

    Args:
        model: a pytorch model
        optimizer: the optimizer used in the training session
        compression_sched: the CompressionScheduler instance used for training, if any
        arch [str]: name of the network architecture/topology. e.g. 'resnet18'
        dataset [str]: dataset. e.g. 'imagenet'
        is_best [bool]: If true, will save a copy of the checkpoint with the suffix 'best'
        name [str]: the name of the checkpoint file
        dir [str]: directory in which to save the checkpoint
        file_ext [str]: file extension, defaults to 'pth.tar'
        extras: optional values with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
    """
    if not os.path.isdir(dir):
        raise IOError(ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(dir))

    # the current method to extract the arch argument from model isn't perfected yet
    # thus, arch is mandatory argument for some topologies
    if arch is None:
        model_arch_list = type(getModuleFromModel(model)).__name__.lower().split()
        if model_arch_list[-1] not in distiller.models.ALL_MODEL_NAMES:
            raise NotImplementedError(
                'Implicit arch is not supported for this model. Please specify arch=ARCH')

    checkpoint = {'extras': extras}
    checkpoint['arch'] = arch or type(getModuleFromModel(model))
    checkpoint['dataset'] = dataset or distiller.apputils.classification_dataset_str_from_arch(checkpoint['arch'])
    checkpoint['state_dict'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer)
    if compression_sched is not None:
        checkpoint['compression_sched'] = compression_sched.state_dict()
    if hasattr(model, 'thinning_recipes'):
        checkpoint['thinning_recipes'] = model.thinning_recipes
    if hasattr(model, 'quantizer_metadata'):
        checkpoint['quantizer_metadata'] = model.quantizer_metadata

    checkpoint_names = ['checkpoint']
    if is_best:
        checkpoint_names.append('best')
    for s in checkpoint_names:
        # construct full path
        filename = s if name is None else '_'.join((name, s))
        filename = '.'.join((filename, file_ext))
        fullpath = os.path.join(dir, filename)

        torch.save(checkpoint, fullpath)
        msglogger.info("Saving checkpoint to: %s" % fullpath)

def load_lean_checkpoint(chkpt_path, map_location=None, model=None,
                         model_create_params=None):
    return load_checkpoint(chkpt_path, map_location, model,
        model_create_params, lean_checkpoint=True)['model']


def get_contents_table(d):
    def inspect_val(val):
        if isinstance(val, (Number, str)):
            return val
        elif isinstance(val, type):
            return val.__name__
        return None

    contents = [[k, type(d[k]).__name__, inspect_val(d[k])] for k in d.keys()]
    contents = sorted(contents, key=lambda entry: entry[0])
    return tabulate(contents, headers=["Key", "Type", "Value"], tablefmt="psql")


def load_checkpoint(chkpt_path, model_device=None, model=None, model_create_params=None, *, 
                    lean_checkpoint=False, strict=False):
    """Load a pytorch training checkpoint.

    Args:
        chkpt_path: path to checkpoint file
        model_device [str]: if set, call model.to(model_device)
                This should be set to either 'cpu' or 'cuda'.
        model: the pytorch model to which we will load the parameters
        model_create_params [dict] - parameters to pass to create_model()
        lean_checkpoint: if set, read into model only 'state_dict' field
    :returns: dict(updated model, compression_scheduler, optimizer, start_epoch, ...)
    """
    if not os.path.isfile(chkpt_path):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_path)

    msglogger.info("=> loading checkpoint %s", chkpt_path)
    checkpoint = torch.load(chkpt_path, map_location=lambda storage, loc: storage)
    msglogger.info('=> Checkpoint contents:\n%s\n' % get_contents_table(checkpoint))

    if 'extras' in checkpoint:
        msglogger.info("=> Checkpoint['extras'] contents:\n{}\n".format(get_contents_table(checkpoint['extras'])))

    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")

    model_created_during_load = model is None
    if model_created_during_load:
        if 'arch' not in checkpoint:
            raise ValueError('Failed to recreate model from checkpoint.')
        model_arch_name = checkpoint['arch'].split('.')[-1]
        model_dataset = checkpoint.get('dataset',
            distiller.apputils.classification_dataset_str_from_arch(model_arch_name))
        model = distiller.models.create_model(False, model_dataset, model_arch_name,
                                              **(model_create_params or dict()))

    try:
        compression_scheduler = None
        normalize_dataparallel_keys = False
        if 'compression_sched' in checkpoint:
            compression_scheduler = distiller.CompressionScheduler(model)
            try:
                compression_scheduler.load_state_dict(checkpoint['compression_sched'], normalize_dataparallel_keys)
            except KeyError as e:
                # A very common source of this KeyError is loading a GPU model on the CPU.
                # We rename all of the DataParallel keys because DataParallel does not execute on the CPU.
                normalize_dataparallel_keys = True
                compression_scheduler.load_state_dict(checkpoint['compression_sched'], normalize_dataparallel_keys)
            msglogger.info("Loaded compression schedule from checkpoint")
        else:
            msglogger.info("Warning: compression schedule data does not exist in the checkpoint")

        if 'thinning_recipes' in checkpoint:
            if 'compression_sched' not in checkpoint:
                raise KeyError("Found thinning_recipes key, but missing mandatory key compression_sched")
            msglogger.info("Loaded a thinning recipe from the checkpoint")
            # Cache the recipes in case we need them later
            model.thinning_recipes = checkpoint['thinning_recipes']
            if normalize_dataparallel_keys:
                model.thinning_recipes = [distiller.get_normalized_recipe(recipe) for recipe in model.thinning_recipes]
            distiller.execute_thinning_recipes_list(model,
                                                    compression_scheduler.zeros_mask_dict,
                                                    model.thinning_recipes)

        if 'quantizer_metadata' in checkpoint:
            msglogger.info('Loaded quantizer metadata from the checkpoint')
            qmd = checkpoint['quantizer_metadata']
            quantizer = qmd['type'](model, **qmd['params'])
            quantizer.prepare_model()

        if normalize_dataparallel_keys:
                checkpoint['state_dict'] = {normalize_module_name(k): v for k, v in checkpoint['state_dict'].items()}
        anomalous_keys = model.load_state_dict(checkpoint['state_dict'], strict)
        if anomalous_keys:
            # This is pytorch 1.1+
            missing_keys, unexpected_keys = anomalous_keys
            if unexpected_keys:
                msglogger.warning("Warning: the loaded checkpoint (%s) contains %d unexpected state keys" % (chkpt_path, len(unexpected_keys)))
            if missing_keys:
                raise ValueError("The loaded checkpoint (%s) is missing %d state keys" % (chkpt_path, len(missing_keys)))
                
        if model_device is not None:
            model.to(model_device)

        if lean_checkpoint:
            msglogger.info("=> loaded 'state_dict' from checkpoint '{}'".format(str(chkpt_path)))
            return {'model': model}

        checkpoint_epoch = -1
        with contextlib.suppress(KeyError):
            checkpoint_epoch = checkpoint['extras']['epoch']

        def _load_optimizer(cls, src_state_dict, model):
            """Initiate optimizer with model parameters and load src_state_dict"""
            # initiate the dest_optimizer with a dummy learning rate,
            # this is required to support SGD.__init__()
            dest_optimizer = cls(model.parameters(), lr=1)
            dest_optimizer.load_state_dict(src_state_dict)
            return dest_optimizer

        try:
            optimizer = _load_optimizer(checkpoint['optimizer_type'],
                checkpoint['optimizer_state_dict'], model)
        except KeyError:
            # Older checkpoints do support optimizer loading: They either had an 'optimizer' field 
            # (different name) which was not used during the load, or they didn't even checkpoint
            # the optimizer. 
            optimizer = None

        if optimizer is not None:
            msglogger.info('Optimizer of type {type} was loaded from checkpoint'.format(
                type=type(optimizer)))
            msglogger.info('Optimizer Args: {}'.format(
                dict((k,v) for k,v in optimizer.state_dict()['param_groups'][0].items()
                                if k != 'params')))
        else:
            msglogger.warning('Optimizer could not be loaded from checkpoint.')

        msglogger.info("=> loaded checkpoint '{f}' (epoch {e})".format(f=str(chkpt_path),
                                                                       e=checkpoint_epoch))

        # extract additional optional fields
        res = checkpoint.get('extras', dict())
        excluded_keys = ['state_dict', 'optimizer_state_dict', 'optimizer_type',
                         'compression_sched', 'thinning_recipes', 'quantizer_metadata',
                         'epoch']
        res.update({k:v for k,v in checkpoint.items() if k not in excluded_keys})
        res.update({
            'model': model,
            'compression_sched': compression_scheduler,
            'optimizer': optimizer,
            'start_epoch': checkpoint_epoch+1,
        })

        return res
    except Exception:
        if model_created_during_load:
            # clean up
            del model
        raise

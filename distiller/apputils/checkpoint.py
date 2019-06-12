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

import os
import shutil
from errno import ENOENT
import logging
from numbers import Number
from tabulate import tabulate
import torch
import distiller
from distiller.utils import normalize_module_name
msglogger = logging.getLogger()


def save_checkpoint(epoch, arch, model, optimizer=None, scheduler=None,
                    extras=None, is_best=False, name=None, dir='.'):
    """Save a pytorch training checkpoint

    Args:
        epoch: current epoch number
        arch: name of the network architecture/topology
        model: a pytorch model
        optimizer: the optimizer used in the training session
        scheduler: the CompressionScheduler instance used for training, if any
        extras: optional dict with additional user-defined data to be saved in the checkpoint.
            Will be saved under the key 'extras'
        is_best: If true, will save a copy of the checkpoint with the suffix 'best'
        name: the name of the checkpoint file
        dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(dir):
        raise IOError(ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(dir))

    if extras is None:
        extras = {}
    if not isinstance(extras, dict):
        raise TypeError('extras must be either a dict or None')

    filename = 'checkpoint.pth.tar' if name is None else name + '_checkpoint.pth.tar'
    fullpath = os.path.join(dir, filename)
    msglogger.info("Saving checkpoint to: %s" % fullpath)
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    fullpath_best = os.path.join(dir, filename_best)

    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['arch'] = arch
    checkpoint['state_dict'] = model.state_dict()
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        checkpoint['optimizer_type'] = type(optimizer)
    if scheduler is not None:
        checkpoint['compression_sched'] = scheduler.state_dict()
    if hasattr(model, 'thinning_recipes'):
        checkpoint['thinning_recipes'] = model.thinning_recipes
    if hasattr(model, 'quantizer_metadata'):
        checkpoint['quantizer_metadata'] = model.quantizer_metadata

    checkpoint['extras'] = extras

    torch.save(checkpoint, fullpath)
    if is_best:
        shutil.copyfile(fullpath, fullpath_best)


def load_lean_checkpoint(model, chkpt_file, model_device=None):
    return load_checkpoint(model, chkpt_file, model_device=model_device,
                           lean_checkpoint=True)[0]


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


def load_checkpoint(model, chkpt_file, optimizer=None, model_device=None, *, lean_checkpoint=False):
    """Load a pytorch training checkpoint.

    Args:
        model: the pytorch model to which we will load the parameters
        chkpt_file: the checkpoint file
        lean_checkpoint: if set, read into model only 'state_dict' field
        optimizer: [deprecated argument]
        model_device [str]: if set, call model.to($model_device)
                This should be set to either 'cpu' or 'cuda'.
    :returns: updated model, compression_scheduler, optimizer, start_epoch
    """
    if not os.path.isfile(chkpt_file):
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)

    msglogger.info("=> loading checkpoint %s", chkpt_file)
    checkpoint = torch.load(chkpt_file, map_location=lambda storage, loc: storage)
    msglogger.info('=> Checkpoint contents:\n%s\n' % get_contents_table(checkpoint))
    if 'extras' in checkpoint:
        msglogger.info("=> Checkpoint['extras'] contents:\n{}\n".format(get_contents_table(checkpoint['extras'])))

    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain the model parameters under the key 'state_dict'")

    checkpoint_epoch = checkpoint.get('epoch', None)
    start_epoch = checkpoint_epoch + 1 if checkpoint_epoch is not None else 0

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
        msglogger.info("Loaded compression schedule from checkpoint (epoch {})".format(
            checkpoint_epoch))
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
    model.load_state_dict(checkpoint['state_dict'])
    if model_device is not None:
        model.to(model_device)

    if lean_checkpoint:
        msglogger.info("=> loaded 'state_dict' from checkpoint '{}'".format(str(chkpt_file)))
        return (model, None, None, 0)

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

    msglogger.info("=> loaded checkpoint '{f}' (epoch {e})".format(f=str(chkpt_file),
                                                                   e=checkpoint_epoch))
    return (model, compression_scheduler, optimizer, start_epoch)

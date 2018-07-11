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
import torch
import distiller
msglogger = logging.getLogger()


def save_checkpoint(epoch, arch, model, optimizer=None, scheduler=None,
                    best_top1=None, is_best=False, name=None, dir='.'):
    """Save a pytorch training checkpoint

    Args:
        epoch: current epoch
        arch: name of the network arechitecture/topology
        model: a pytorch model
        optimizer: the optimizer used in the training session
        scheduler: the CompressionScheduler instance used for training, if any
        best_top1: the best top1 score seen so far
        is_best: True if this is the best (top1 accuracy) model so far
        name: the name of the checkpoint file
        dir: directory in which to save the checkpoint
    """
    if not os.path.isdir(dir):
        raise IOError(ENOENT, 'Checkpoint directory does not exist at', os.path.abspath(dir))

    filename = 'checkpoint.pth.tar' if name is None else name + '_checkpoint.pth.tar'
    fullpath = os.path.join(dir, filename)
    msglogger.info("Saving checkpoint to: %s" % fullpath)
    filename_best = 'best.pth.tar' if name is None else name + '_best.pth.tar'
    fullpath_best = os.path.join(dir, filename_best)
    checkpoint = {}
    checkpoint['epoch'] = epoch
    checkpoint['arch'] = arch
    checkpoint['state_dict'] = model.state_dict()
    if best_top1 is not None:
        checkpoint['best_top1'] = best_top1
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint['compression_sched'] = scheduler.state_dict()
    if hasattr(model, 'thinning_recipes'):
        checkpoint['thinning_recipes'] = model.thinning_recipes
    if hasattr(model, 'quantizer_metadata'):
        checkpoint['quantizer_metadata'] = model.quantizer_metadata

    torch.save(checkpoint, fullpath)
    if is_best:
        shutil.copyfile(fullpath, fullpath_best)


def load_checkpoint(model, chkpt_file, optimizer=None):
    """Load a pytorch training checkpoint

    Args:
        model: the pytorch model to which we will load the parameters
        chkpt_file: the checkpoint file
        optimizer: the optimizer to which we will load the serialized state
    """
    compression_scheduler = None
    start_epoch = 0

    if os.path.isfile(chkpt_file):
        msglogger.info("=> loading checkpoint %s", chkpt_file)
        checkpoint = torch.load(chkpt_file)
        msglogger.info("Checkpoint keys:\n{}".format("\n\t".join(k for k in checkpoint.keys())))
        start_epoch = checkpoint['epoch'] + 1
        best_top1 = checkpoint.get('best_top1', None)
        if best_top1 is not None:
            msglogger.info("   best top@1: %.3f", best_top1)

        if 'compression_sched' in checkpoint:
            compression_scheduler = distiller.CompressionScheduler(model)
            compression_scheduler.load_state_dict(checkpoint['compression_sched'])
            msglogger.info("Loaded compression schedule from checkpoint (epoch %d)",
                           checkpoint['epoch'])
        else:
            msglogger.info("Warning: compression schedule data does not exist in the checkpoint")

        if 'thinning_recipes' in checkpoint:
            if 'compression_sched' not in checkpoint:
                raise KeyError("Found thinning_recipes key, but missing mandatory key compression_sched")
            msglogger.info("Loaded a thinning recipe from the checkpoint")
            # Cache the recipes in case we need them later
            model.thinning_recipes = checkpoint['thinning_recipes']
            distiller.execute_thinning_recipes_list(model,
                                              compression_scheduler.zeros_mask_dict,
                                              model.thinning_recipes)

        if 'quantizer_metadata' in checkpoint:
            msglogger.info('Loaded quantizer metadata from the checkpoint')
            qmd = checkpoint['quantizer_metadata']
            quantizer = qmd['type'](model, **qmd['params'])
            quantizer.prepare_model()

        msglogger.info("=> loaded checkpoint '%s' (epoch %d)", chkpt_file, checkpoint['epoch'])

        model.load_state_dict(checkpoint['state_dict'])
        return model, compression_scheduler, start_epoch
    else:
        raise IOError(ENOENT, 'Could not find a checkpoint file at', chkpt_file)

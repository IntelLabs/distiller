#
# Copyright (c) 2019 Intel Corporation
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

"""Lottery Ticket Hypothesis"""


import logging
import distiller.apputils as apputils
import distiller.models
import torch
import argparse
from tabulate import tabulate
import distiller


def print_sparsities(masks_dict):
    mask_sparsities = [(param_name, distiller.sparsity(mask)) for param_name, mask in masks_dict.items()
                       if mask is not None]
    print(tabulate(mask_sparsities, headers=["Module", "Mask Sparsity"], tablefmt="fancy_grid"))


def add_args(argparser):
    """
    Helper function which defines command-line arguments specific to Lottery Ticket Hypothesis training.

    Arguments:
        argparser (argparse.ArgumentParser): Existing parser to which to add the arguments
    """
    group = argparser.add_argument_group('AutoML Compression Arguments')
    group.add_argument('--lt-untrained-ckpt', type=str, action='store',
                        help='Checkpoint file of the untrained network (randomly initialized)')
    group.add_argument('--lt-pruned-ckpt', type=str, action='store',
                        help='Checkpoint file of the pruned (but not thinned) network')
    return argparser


def extract_lottery_ticket(args, untrained_ckpt_name, pruned_ckpt_name):
    untrained_ckpt = apputils.load_checkpoint(model=None, chkpt_file=untrained_ckpt_name, model_device='cpu')
    untrained_model, _, optimizer, start_epoch = untrained_ckpt

    pruned_ckpt = apputils.load_checkpoint(model=None, chkpt_file=pruned_ckpt_name, model_device='cpu')
    pruned_model, pruned_scheduler, optimizer, start_epoch = pruned_ckpt

    # create a dictionary of masks by inferring the masks from the parameter sparsity
    masks_dict = {pname: (torch.ne(param, 0)).type(param.type())
                  for pname, param in pruned_model.named_parameters()
                  if pname in pruned_scheduler.zeros_mask_dict.keys()}
    for pname, mask in masks_dict.items():
        untrained_model.state_dict()[pname].mul_(mask)

    sparsities = {pname: distiller.sparsity(mask) for pname, mask in masks_dict.items()}
    print(sparsities)
    pruned_scheduler.init_from_masks_dict(masks_dict)

    apputils.save_checkpoint(0, pruned_model.arch, untrained_model, optimizer=optimizer,
                             scheduler=pruned_scheduler,
                             name='_'.join([untrained_ckpt_name, 'masked']))

    # pruned_ckpt = torch.load(pruned_ckpt_name, map_location='cpu')
    #
    # assert 'extras' in pruned_ckpt and pruned_ckpt['extras']
    # print("\nContents of Checkpoint['extras']:")
    # print(get_contents_table(pruned_ckpt['extras']))
    # masks_dict = pruned_ckpt["extras"]["creation_masks"]
    # print_sparsities(masks_dict)
    # compression_scheduler = distiller.CompressionScheduler(model)
    # compression_scheduler.load_state_dict(masks_dict)
    #
    # model = distiller.models.create_model(False, args.dataset, args.arch, device_ids=[-1])
    # model = apputils.load_lean_checkpoint(model, args.load_model_path, model_device=args.device)
    #
    # apputils.save_checkpoint(0, args.arch, model, optimizer=None, scheduler=scheduler,
    #                      name='_'.join([args.name, checkpoint_name]) if args.name else checkpoint_name,
    #                      dir=msglogger.logdir, extras={'quantized_top1': top1})

msglogger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Lottery Ticket Hypothesis')
    add_args(parser)
    args = parser.parse_args()
    extract_lottery_ticket(args, args.lt_untrained_ckpt, args.lt_pruned_ckpt)
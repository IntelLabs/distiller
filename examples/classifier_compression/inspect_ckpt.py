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

"""A small utility to inspect the contents of checkpoint files.

Sometimes it is useful to look at the contents of a checkpoint file, and this utility is meant to help
with this.
By default this utility will print just the names and types of the keys it finds in the checkpoint
file.  If the key type is simple (i.e. integer, float, or string), then the value is printed as well.

You can also print the model keys (i.e. the names of the parameters tensors in the model), and the
weight tensor masks in the schedule).

$ python3 inspect_ckpt.py checkpoint.pth.tar --model --schedule
"""
import torch
import argparse
from tabulate import tabulate
import distiller
from distiller.apputils.checkpoint import get_contents_table


def print_sparsities(masks_dict):
    mask_sparsities = [(param_name, distiller.sparsity(mask)) for param_name, mask in masks_dict.items()
                       if mask is not None]
    print(tabulate(mask_sparsities, headers=["Module", "Mask Sparsity"], tablefmt="fancy_grid"))


def inspect_checkpoint(chkpt_file, args):
    print("Inspecting checkpoint file: ", chkpt_file)
    # force loading on the CPU which always has more memory than the GPU(s)
    checkpoint = torch.load(chkpt_file, map_location='cpu')

    print(get_contents_table(checkpoint))

    if 'extras' in checkpoint and checkpoint['extras']:
        print("\nContents of Checkpoint['extras']:")
        print(get_contents_table(checkpoint['extras']))
        try:
            print_sparsities(checkpoint["extras"]["creation_masks"])
        except KeyError:
            pass

    if args.model and "state_dict" in checkpoint:
        print("\nModel keys (state_dict):\n{}".format(", ".join(list(checkpoint["state_dict"].keys()))))

    if args.schedule and "compression_sched" in checkpoint:
        compression_sched = checkpoint["compression_sched"]
        print("\nSchedule keys (compression_sched):\n{}\n".format("\n\t".join(list(compression_sched.keys()))))
        sched_keys = [[k, type(compression_sched[k]).__name__] for k in compression_sched.keys()]
        print(tabulate(sched_keys, headers=["Key", "Type"], tablefmt="fancy_grid"))
        try:
            masks_dict = compression_sched["masks_dict"]
            print("compression_sched[\"masks_dict\"] keys:\n{}".format(list(masks_dict.keys())))
            print_sparsities(masks_dict)
        except KeyError:
            pass

    if args.thinning and "thinning_recipes" in checkpoint:
        for recipe in checkpoint["thinning_recipes"]:
            print(recipe)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distiller checkpoint inspection')
    parser.add_argument('chkpt_file', help='path to the checkpoint file')
    parser.add_argument('-m', '--model', action='store_true', help='print the model keys')
    parser.add_argument('-s', '--schedule', action='store_true', help='print the schedule keys')
    parser.add_argument('-t', '--thinning', action='store_true', help='print the thinning keys')
    args = parser.parse_args()
    inspect_checkpoint(args.chkpt_file, args)

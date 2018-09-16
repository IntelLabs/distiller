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


def inspect_checkpoint(chkpt_file, args):
    def inspect_val(val):
        if isinstance(val, (int, float, str)):
            return val
        return None

    print("Inspecting checkpoint file: ", chkpt_file)
    checkpoint = torch.load(chkpt_file)

    chkpt_keys = [[k, type(checkpoint[k]).__name__, inspect_val(checkpoint[k])] for k in checkpoint.keys()]
    print(tabulate(chkpt_keys, headers=["Key", "Type", "Value"], tablefmt="fancy_grid"))

    if args.model and "state_dict" in checkpoint:
        print("\nModel keys (state_dict):\n{}".format(", ".join(list(checkpoint["state_dict"].keys()))))

    if args.schedule and "compression_sched" in checkpoint:
        compression_sched = checkpoint["compression_sched"]
        print("\nSchedule keys (compression_sched):\n{}\n".format("\n\t".join(list(compression_sched.keys()))))
        sched_keys = [[k, type(compression_sched[k]).__name__] for k in compression_sched.keys()]
        print(tabulate(sched_keys, headers=["Key", "Type"], tablefmt="fancy_grid"))
        if "masks_dict" in checkpoint["compression_sched"]:
            print("compression_sched[\"masks_dict\"] keys:\n{}".format(", ".join(
                  list(compression_sched["masks_dict"].keys()))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Distiller checkpoint inspection')
    parser.add_argument('chkpt_file', help='path to the checkpoint file')
    parser.add_argument('-m', '--model', action='store_true', help='print the model keys')
    parser.add_argument('-s', '--schedule', action='store_true', help='print the schedule keys')
    args = parser.parse_args()
    inspect_checkpoint(args.chkpt_file, args)

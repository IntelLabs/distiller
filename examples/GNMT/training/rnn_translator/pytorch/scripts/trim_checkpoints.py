import argparse
import glob
import os
import subprocess
import sys
import time

import torch


def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser(description='Trim training checkpoints')
    parser.add_argument('--path', required=True,
                        help='path to directory with checkpoints (*.pth)')
    parser.add_argument('--suffix', default='trim',
                        help='suffix appended to the name of output file')
    return parser.parse_args()


def get_checkpoints(path):
    """
    Gets all *.pth checkpoints from a given directory.

    :param path:
    """
    checkpoints = glob.glob(os.path.join(path, '*.pth'))
    return checkpoints

def main():
    # Add parent folder to sys.path
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    args = parse_args()

    checkpoints = get_checkpoints(args.path)
    print('All checkpoints:', checkpoints)

    for checkpoint in checkpoints:
        print('Processing ', checkpoint)
        chkpt = torch.load(checkpoint)
        chkpt['optimizer'] = None
        output_file = checkpoint.replace('pth', args.suffix + '.pth')
        torch.save(chkpt, output_file)

if __name__ == "__main__":
    main()

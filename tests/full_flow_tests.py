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

import subprocess
import os
import errno
import re
from collections import namedtuple
import argparse
import time

DS_CIFAR = 'cifar10'

distiller_root = os.path.realpath('..')
examples_root = os.path.join(distiller_root, 'examples')
script_path = os.path.realpath(os.path.join(examples_root, 'classifier_compression',
                                            'compress_classifier.py'))

###########
# Some Basic Logging Mechanisms
###########

class Colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[37m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_PURPLE = '\033[45m'
    BG_CYAN = '\033[30;46m'
    BG_WHITE = '\x1b[30;47m'
    BG_RESET = '\033[49m'
    BOLD = '\033[1m'
    UNDERLINE_ON = '\033[4m'
    UNDERLINE_OFF = '\033[24m'
    END = '\033[0m'


def colorize(string, color):
    return color + string + Colors.END


def error(string):
    print(colorize('ERROR: ' + string, Colors.RED))


def test_progress(string):
    print(colorize(string, Colors.BLUE))


def success(string):
    print(colorize(string, Colors.GREEN))


###########
# Checkers
###########

def compare_values(name, expected, actual):
    print('Comparing {0}: Expected = {1} ; Actual = {2}'.format(name, expected, actual))
    if expected != actual:
        error('Mismatch on {0}'.format(name))
        return False
    else:
        return True


def accuracy_checker(log, expected_top1, expected_top5):
    tops = re.findall(r"Top1: (?P<top1>\d*\.\d*) *Top5: (?P<top5>\d*\.\d*)", log)
    if not tops:
        error('No accuracy results in log')
        return False
    if not compare_values('Top-1', expected_top1, float(tops[-1][0])):
        return False
    return compare_values('Top-5', expected_top5, float(tops[-1][1]))


###########
# Test Configurations
###########
TestConfig = namedtuple('TestConfig', ['args', 'dataset', 'checker_fn', 'checker_args'])

test_configs = [
    TestConfig('--arch simplenet_cifar --epochs 2', DS_CIFAR, accuracy_checker, [48.340, 92.630]),
    TestConfig('-a resnet20_cifar --resume {0} --quantize --evaluate'.
               format(os.path.join(examples_root, 'ssl', 'checkpoints', 'checkpoint_trained_dense.pth.tar')),
               DS_CIFAR, accuracy_checker, [91.620, 99.630]),
    TestConfig('-a preact_resnet20_cifar --epochs 2 --compress {0}'.
               format(os.path.join('full_flow_tests', 'preact_resnet20_cifar_pact_test.yaml')),
               DS_CIFAR, accuracy_checker, [48.290, 94.460])
]


###########
# Tests Execution
###########

def process_failure(msg, test_idx, cmd, log_path, failed_tests, log):
    error(msg)
    if not log_path:
        test_progress('Log file not created. Full output from test:')
        print(log)
    else:
        test_progress('Test log file: {0}'.format(colorize(log_path, Colors.YELLOW)))
    failed_tests.append((test_idx, cmd, log_path))


def validate_dataset_path(path, default, name):
    if path:
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            error("Path provided to {0} dataset doesn't exist".format(name))
            exit(1)
        return path

    test_progress('Path to {0} dataset not provided, defaulting to: {1}'.format(name,
                                                                                colorize(os.path.abspath(default),
                                                                                         Colors.WHITE)))
    try:
        os.makedirs(default)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    return default


def run_tests():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cifar10-path', dest='cifar10_path', metavar='DIR', help='Path to CIFAR-10 dataset')
    args = parser.parse_args()

    cifar10_path = validate_dataset_path(args.cifar10_path, default='data.cifar10', name='CIFAR-10')

    datasets = {DS_CIFAR: cifar10_path}
    total_configs = len(test_configs)
    failed_tests = []
    for idx, tc in enumerate(test_configs):
        print('')
        test_progress('-------------------------------------------------')
        test_progress('Running Test {0} / {1}'.format(idx + 1, total_configs))
        dataset_dir = datasets[tc.dataset]
        # Run with '--det -j 1' to ensure deterministic results
        # Run with single GPU (lowest common denominator...)
        cmd = 'python3 {script} {tc_args} --det -j 1 --gpus 0 {data}'.format(script=script_path, tc_args=tc.args,
                                                                    data=dataset_dir)

        test_progress('Executing command: ' + colorize(cmd, Colors.YELLOW))
        p = subprocess.Popen(cmd.split(' '), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        # Poll for completion
        waiting_chars = ['-', '\\', '|', '/']
        cnt = 0
        while p.poll() is None:
            print(waiting_chars[cnt] * 5, end='\r', flush=True)
            cnt = (cnt + 1) % len(waiting_chars)
            time.sleep(0.5)

        log = p.stdout.read()
        log_path = re.match(r"Log file for this run: (.*)", log)
        log_path = log_path.groups()[0] if log_path else ''

        if p.returncode != 0:
            process_failure('Command returned with exit status {0}'.
                            format(p.returncode), idx, cmd, log_path, failed_tests, log)
            continue
        test_progress('Running checker: ' + colorize(tc.checker_fn.__name__, Colors.YELLOW))
        if not tc.checker_fn(log, *tc.checker_args):
            process_failure('Checker failed', idx, cmd, log_path, failed_tests, log)
            continue
        success('TEST PASSED')
        test_progress('Test log file: {0}'.format(colorize(log_path, Colors.YELLOW)))

    print('')
    test_progress('-------------------------------------------------')
    test_progress('-------------------------------------------------')
    test_progress('All tests completed')
    test_progress('# Tests run: {0} ; # Tests passed {1} ; # Tests failed: {2}'.
                  format(total_configs, total_configs - len(failed_tests), len(failed_tests)))
    if failed_tests:
        print('')
        print(colorize('Failed tests summary:', Colors.RED))
        for idx, cmd, log_path in failed_tests:
            print(colorize('  Test Index:', Colors.YELLOW), idx + 1)
            print(colorize('    Command Line:', Colors.YELLOW), cmd)
            print(colorize('    Log File Path:', Colors.YELLOW), log_path)
        exit(1)
    print('')
    success('ALL TESTS PASSED')
    print('')
    exit(0)


if __name__ == '__main__':
    run_tests()
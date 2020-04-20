#
# Copyright (c) 2020 Intel Corporation
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

"""Compression schedule specification using Distiller API.

This script shows how to specify a compression-schedule directly using Distiller's API, instead of using a YAML
specification.
distiller.CompressionScheduler uses a declarative specification of the compression-schedule, to control the
compression process. A YAML specification helps us clearly separate the compression-schedule from the rest of the
application, and makes it easy to invoke a compression application with different schedules. However, a YAML
specification is not mandatory and can be replaced by building (declaring) the compression specification from
code.

The schedule specified in the code is equivalent to this YAML schedule:
    pruners:
      filter_pruner:
        class: 'L1RankedStructureParameterPruner'
        group_type: Filters
        desired_sparsity: 0.1
        weights: [module.conv1.weight]

      filter_pruner_agp:
        class: 'L1RankedStructureParameterPruner_AGP'
        group_type: Filters
        initial_sparsity: 0.05
        final_sparsity: 0.20
        weights: [module.conv2.weight]

      gemm_pruner_agp:
        class: 'AutomatedGradualPruner'
        initial_sparsity: 0.02
        final_sparsity: 0.15
        weights: [module.fc2.weight]

    extensions:
      net_thinner:
          class: 'FilterRemover'
          thinning_func_str: remove_filters
          arch: 'simplenet_mnist'
          dataset: 'mnist'

    policies:
      - pruner:
          instance_name: filter_pruner
        epochs: [0,1]

      - pruner:
          instance_name: filter_pruner_agp
        starting_epoch: 0
        ending_epoch: 2
        frequency: 1

      - pruner:
          instance_name: gemm_pruner_agp
        starting_epoch: 0
        ending_epoch: 2
        frequency: 1

      - extension:
          instance_name: net_thinner
        epochs: [2]

To invoke:
    $ python3 <DISTILLER HOME>/examples/scheduling_api/direct_api_pruning.py --arch simplenet_mnist --epochs 3 -p=50  --det -j 1 --gpus 0 /datasets/mnist

This is equivalent to the following:
    $ python3 <DISTILLER HOME>/examples/classifier_compression/compress_classifier.py --arch simplenet_mnist --epochs 3 -p=50 --compress=full_flow_tests/simplenet_mnist_pruning.yaml --det -j 1 --gpus 0 /datasets/mnist
"""


import os
import distiller.apputils.image_classifier as classifier
import distiller


def train_model(app, nepochs):
    best = [float("-inf"), float("-inf"), float("inf")]
    for epoch in range(nepochs):
        validate = True
        top1, top5, loss = app.train_validate_with_scheduling(epoch,
                                                              validate=validate,
                                                              verbose=True)
        if validate:
            if top1 > best[0]:
                best = [top1, top5, loss]
    return best


if __name__ == '__main__':
    argparser = classifier.init_classifier_compression_arg_parser()
    app_args = argparser.parse_args()
    assert app_args.compress is None
    assert app_args.arch == "simplenet_mnist"

    app = classifier.ClassifierCompressor(app_args, script_dir=os.path.dirname(__file__))
    compression_scheduler = distiller.CompressionScheduler(app.model)

    # Pruners
    filter_pruner = distiller.L1RankedStructureParameterPruner(
        name='filter_pruner',
        group_type='Filters',
        desired_sparsity=0.1,
        weights=['module.conv1.weight'])

    filter_pruner_agp = distiller.L1RankedStructureParameterPruner_AGP(
        name='filter_pruner_agp',
        group_type='Filters',
        initial_sparsity=0.05,
        final_sparsity=0.20,
        weights=['module.conv2.weight'])

    gemm_pruner_agp = distiller.AutomatedGradualPruner(
        name='gemm_pruner_agp',
        initial_sparsity=0.02,
        final_sparsity=0.15,
        weights=['module.fc2.weight'])

    net_thinner = distiller.FilterRemover("remove_filters",
                                          arch='simplenet_mnist',
                                          dataset='mnist')

    # Policies
    policy1 = distiller.PruningPolicy(filter_pruner, pruner_args=None)
    compression_scheduler.add_policy(policy1, epochs=(0,1))

    policy2 = distiller.PruningPolicy(filter_pruner_agp, pruner_args=None)
    compression_scheduler.add_policy(policy2, starting_epoch=0, ending_epoch=2, frequency=1)

    policy3 = distiller.PruningPolicy(gemm_pruner_agp, pruner_args=None)
    compression_scheduler.add_policy(policy3, starting_epoch=0, ending_epoch=2, frequency=1)

    compression_scheduler.add_policy(net_thinner, epochs=(2,))

    # Plug the new compression scheduler into the sample application
    app.compression_scheduler = compression_scheduler

    validation_best_results = train_model(app, app.args.epochs)
    test_results = app.test()
    print(test_results)


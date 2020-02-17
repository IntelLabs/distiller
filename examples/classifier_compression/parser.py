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

import argparse
import distiller
import distiller.pruning
import distiller.models as models


def add_cmdline_args(parser):
    parser.add_argument('--save-untrained-model',  action='store_true', default=False,
                        help='Save the randomly-initialized model before training (useful for lottery-ticket method)')
    parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None,
                        help='List of loss weights for early exits (e.g. --earlyexit_lossweights 0.1 0.3)')
    parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None,
                        help='List of EarlyExit thresholds (e.g. --earlyexit_thresholds 1.2 0.9)')
    distiller.knowledge_distillation.add_distillation_args(parser, models.ALL_MODEL_NAMES, True)
    distiller.pruning.greedy_filter_pruning.add_greedy_pruner_args(parser)
    return parser

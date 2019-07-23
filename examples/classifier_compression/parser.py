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
import distiller.quantization
import examples.automated_deep_compression as adc
import distiller.models as models
#from utils_cnn_classifier import init_classifier_compression_arg_parser

def get_parser(parser):
    #parser = init_classifier_compression_arg_parser()
    parser.add_argument('--earlyexit_lossweights', type=float, nargs='*', dest='earlyexit_lossweights', default=None,
                        help='List of loss weights for early exits (e.g. --earlyexit_lossweights 0.1 0.3)')
    parser.add_argument('--earlyexit_thresholds', type=float, nargs='*', dest='earlyexit_thresholds', default=None,
                        help='List of EarlyExit thresholds (e.g. --earlyexit_thresholds 1.2 0.9)')
    distiller.knowledge_distillation.add_distillation_args(parser, models.ALL_MODEL_NAMES, True)
    distiller.quantization.add_post_train_quant_args(parser)
    distiller.pruning.greedy_filter_pruning.add_greedy_pruner_args(parser)
    adc.automl_args.add_automl_args(parser)
    return parser

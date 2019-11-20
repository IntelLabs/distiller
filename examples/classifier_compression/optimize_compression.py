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

"""This is an initial example of integration of FB's Ax (https://github.com/facebook/Ax)


time python3 optimize_compression.py --arch resnet20_cifar  $CIFAR10_PATH -p=50 --lr=0.4 --epochs=180 --compress=../agp-pruning/resnet20_filters.schedule_agp.yaml  --resume-from=../ssl/checkpoints/checkpoint_trained_dense.pth.tar --vs=0 --reset-optimizer --gpu=

"""
import traceback
import logging
import ax
from examples.classifier_compression.compress_classifier import ClassifierCompressorSampleApp
import distiller.apputils.image_classifier as classifier
import parser
import os


# Logger handle
msglogger = logging.getLogger()


def main():
    def train_evaluate_distiller(parameters):
        args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser()).parse_args()
        args.lr = parameters.get("lr", args.lr)
        app = ClassifierCompressorSampleApp(args, script_dir=os.path.dirname(__file__))
        if app.handle_subapps():
            return
        app.ending_epoch = args.epochs
        net = app.model
        app.args.lr = parameters.get("lr", app.args.lr)
        for param_group in app.optimizer.param_groups:
            param_group['lr'] = app.args.lr
        perf_scores_history = app.run_training_loop()
        return perf_scores_history[0].top1

    best_parameters, values, experiment, model = ax.service.managed_loop.optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [1e-2, 0.4], "log_scale": True},
            {"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        ],
        evaluation_function=train_evaluate_distiller,
        objective_name='accuracy',
    )
    msglogger.info(best_parameters)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    except Exception as e:
        if msglogger is not None:
            # We catch unhandled exceptions here in order to log them to the log file
            # However, using the msglogger as-is to do that means we get the trace twice in stdout - once from the
            # logging operation and once from re-raising the exception. So we remove the stdout logging handler
            # before logging the exception
            handlers_bak = msglogger.handlers
            msglogger.handlers = [h for h in msglogger.handlers if type(h) != logging.StreamHandler]
            msglogger.error(traceback.format_exc())
            msglogger.handlers = handlers_bak
        raise
    finally:
        if msglogger is not None and hasattr(msglogger, 'log_filename'):
            msglogger.info()
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))

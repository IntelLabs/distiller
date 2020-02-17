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

import os
import torch
from copy import deepcopy
import logging
from collections import OrderedDict

import distiller
import distiller.apputils as apputils
import distiller.apputils.image_classifier as classifier
import distiller.quantization.ptq_coordinate_search as lapq


msglogger = logging.getLogger()


def image_classifier_ptq_lapq(model, criterion, loggers, args):
    args = deepcopy(args)

    effective_test_size_bak = args.effective_test_size
    args.effective_test_size = args.lapq_eval_size
    eval_data_loader = classifier.load_data(args, load_train=False, load_val=False, load_test=True, fixed_subset=True)

    args.effective_test_size = effective_test_size_bak
    test_data_loader = classifier.load_data(args, load_train=False, load_val=False, load_test=True)

    model = model.eval()
    device = next(model.parameters()).device

    if args.lapq_eval_memoize_dataloader:
        images_batches = []
        targets_batches = []
        for images, targets in eval_data_loader:
            images_batches.append(images.to(device))
            targets_batches.append(targets.to(device))
        memoized_data_loader = [(torch.cat(images_batches), torch.cat(targets_batches))]
    else:
        memoized_data_loader = None

    def eval_fn(model):
        if memoized_data_loader:
            loss = 0
            for images, targets in memoized_data_loader:
                outputs = model(images)
                loss += criterion(outputs, targets).item()
            loss = loss / len(memoized_data_loader)
        else:
            _, _, loss = classifier.test(eval_data_loader, model, criterion, loggers, None, args)
        return loss

    def test_fn(model):
        top1, top5, loss = classifier.test(test_data_loader, model, criterion, loggers, None, args)
        return OrderedDict([('top-1', top1), ('top-5', top5), ('loss', loss)])

    args.device = device
    if args.resumed_checkpoint_path:
        args.load_model_path = args.resumed_checkpoint_path
    if args.load_model_path:
        msglogger.info("Loading checkpoint from %s" % args.load_model_path)
        model = apputils.load_lean_checkpoint(model, args.load_model_path,
                                              model_device=args.device)

    quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(model, args)

    dummy_input = torch.rand(*model.input_shape, device=args.device)
    model, qp_dict = lapq.ptq_coordinate_search(quantizer, dummy_input, eval_fn, test_fn=test_fn,
                                                **lapq.cmdline_args_to_dict(args))

    results = test_fn(quantizer.model)
    msglogger.info("Arch: %s \tTest: \t top1 = %.3f \t top5 = %.3f \t loss = %.3f" %
                   (args.arch, results['top-1'], results['top-5'], results['loss']))
    distiller.yaml_ordered_save('%s.quant_params_dict.yaml' % args.arch, qp_dict)

    distiller.apputils.save_checkpoint(0, args.arch, model,
                                       extras={'top1': results['top-1'], 'qp_dict': qp_dict}, name=args.name,
                                       dir=msglogger.logdir)


if __name__ == "__main__":
    parser = classifier.init_classifier_compression_arg_parser(include_ptq_lapq_args=True)
    args = parser.parse_args()
    args.epochs = float('inf')  # hack for args parsing so there's no error in epochs
    cc = classifier.ClassifierCompressor(args, script_dir=os.path.dirname(__file__))
    image_classifier_ptq_lapq(cc.model, cc.criterion, [cc.pylogger, cc.tflogger], cc.args)

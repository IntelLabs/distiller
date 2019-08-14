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
"""This package contains the implementation of a greedy algorithm that is similar to NetAdapt and CAR.

We score improvement by means of the computation load (MACs/FLOPs) and do not use direct metrics because this is
an open-source package that runs on many different hardware platforms.

TODO: this code requires refactoring as is makes assumptions about applicative layers (e.g. the names of certain
      members of the application `args` variable) and this is both tight-coupling and reverse-dependency.

An example invocation:
$ time python3 compress_classifier.py --arch=mobilenet PATH-TO-IMAGENET_DS  --resume=mobilenet_sgd_68.848.pth.tar
--greedy --greedy-target-density=0.5 --vs=0 -p=50 --lr=0.1 --gpu=0 --greedy-pruning-step=0.1 --effective-train-size=0.1


References:
[1] Structural Compression of Convolutional Neural Networks Based on Greedy Filter Pruning
    Reza Abbasi-Asl, Bin Yu https://arxiv.org/abs/1705.07356

[2] NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications
    Tien-Ju Yang, Andrew Howard, Bo Chen, Xiao Zhang, Alec Go, Mark Sandler, Vivienne Sze, Hartwig Adam
    https://arxiv.org/abs/1804.03230, ECCV 2018
"""

from copy import deepcopy
import numpy as np
import logging
import torch
import torch.multiprocessing as mp
from multiprocessing import Process, Queue
import multiprocessing
import csv
import os
import distiller
from distiller.apputils import save_checkpoint
from distiller.data_loggers import PythonLogger
from distiller.summary_graph import SummaryGraph
from distiller import normalize_module_name

__all__ = ['add_greedy_pruner_args', 'greedy_pruner']
msglogger = logging.getLogger()


def add_greedy_pruner_args(argparser, arch_choices=None, enable_pretrained=False):
    """
    Helper function to make it easier to add command-line arguments for Greedy Pruning
    to any application.

    Arguments:
        argparser (argparse.ArgumentParser): Existing parser to which to add the arguments
    """
    argparser.add_argument('--greedy', dest='greedy', action='store_true', help='greedy filter pruning')
    group = argparser.add_argument_group('Greedy Pruning')
    group.add_argument('--greedy-ft-epochs', type=int, default=1,
                       help='number of epochs to fine-tune each discovered network')
    group.add_argument('--greedy-target-density', type=float,
                       help='target density of the network we are seeking')
    group.add_argument('--greedy-pruning-step', type=float, default=0.10,
                       help='size of each pruning step (as a fraction in [0..1])')
    group.add_argument('--greedy-finetuning-policy', choices=["constant", "linear-grow"], default="constant",
                       help='policy used for determining how long to fine-tune')


def prune_tensor(param, param_name, fraction_to_prune, zeros_mask_dict):
    """Prune filters from a parameter tensor.

    Returns the filter-sparsity of the tensor.
    """
    # Create a filter-ranking pruner
    pruner = distiller.pruning.L1RankedStructureParameterPruner(name=None,
                                                                group_type="Filters",
                                                                desired_sparsity=fraction_to_prune,
                                                                weights=param_name)
    pruner.set_param_mask(param, param_name, zeros_mask_dict, meta=None)
    # Use the mask to prune
    zeros_mask_dict[param_name].apply_mask(param)
    return distiller.sparsity_3D(param)


def prune_finetune_test(iteration, model_cpy, pruning_step, test_fn, train_fn,
                        app_args, param_name, effective_train_size):
    pylogger = PythonLogger(msglogger)
    zeros_mask_dict = distiller.create_model_masks_dict(model_cpy)
    param = model_cpy.state_dict()[param_name]
    if 0 == prune_tensor(param, param_name, pruning_step, zeros_mask_dict):
        return (-1, -1, -1, zeros_mask_dict)  # Did not prune anything

    if train_fn is not None:
        # Fine-tune
        optimizer = torch.optim.SGD(model_cpy.parameters(), lr=app_args.lr,
                                    momentum=app_args.momentum, weight_decay=app_args.weight_decay)
        app_args.effective_train_size = effective_train_size
        train_fn(model=model_cpy, compression_scheduler=create_scheduler(model_cpy, zeros_mask_dict),
                 optimizer=optimizer, epoch=iteration, loggers=[pylogger])

    # Physically remove filters
    dataset = app_args.dataset
    arch = app_args.arch
    distiller.remove_filters(model_cpy, zeros_mask_dict, arch, dataset, optimizer=None)

    # Test and record the performance of the pruned model
    prec1, prec5, loss = test_fn(model=model_cpy, loggers=None)
    return (prec1, prec5, loss, zeros_mask_dict)


def find_most_robust_layer(iteration, model, pruning_step, test_fn, train_fn,
                           app_args, tensors_to_prune=None, effective_train_size=1):
    """Find the layer that is most robust to pruning 'pruning_step' filters.

    For each layer: prune 'step' percent of the filters, fine-tune, and measure top1 accuracy
    """
    if tensors_to_prune is None:
        tensors_to_prune = [param_name for param_name, _ in model.named_parameters()]
    # best_layer format: (prec1, prec5, param_name, pruned_model, zeros_mask_dict)
    best_layer = (-np.inf, -np.inf, None, None, None, np.inf)
    for param_name in tensors_to_prune:
        if model.state_dict()[param_name].dim() != 4:
            continue
        # Make a copy of the model, because when we prune the parameter tensor the model's weights are altered
        torch.cuda.empty_cache()
        model_cpy = deepcopy(model)
        (prec1, prec5, loss, zeros_mask_dict) = prune_finetune_test(iteration, model_cpy, pruning_step, test_fn,
                                                                    train_fn, app_args, param_name,
                                                                    effective_train_size)
        if (prec1, prec5, -loss) > (best_layer[0], best_layer[1], -best_layer[-1]):
            best_layer = (prec1, prec5, param_name, model_cpy, zeros_mask_dict, loss)
        del model_cpy
        del zeros_mask_dict
    return best_layer[:-1]


def find_most_robust_layer_mp(iteration, model, pruning_step, test_fn, train_fn,
                              app_args, tensors_to_prune=None, effective_train_size=1):
    """Multiprocessing version of find_most_robust_layer
    """
    if tensors_to_prune is None:
        tensors_to_prune = [param_name for param_name, _ in model.named_parameters()]
    # best_layer format: (prec1, prec5, param_name, pruned_model, zeros_mask_dict)
    best_layer = (-np.inf, -np.inf, None, None, None, np.inf)
    pqueue = Queue()
    processes = []
    for param_name in tensors_to_prune:
        if model.state_dict()[param_name].dim() != 4:
            continue
        # Make a copy of the model, because when we prune the parameter tensor the model's weights are altered
        model_cpy = deepcopy(model)
        model_cpy.share_memory()
        (prec1, prec5, loss, zeros_mask_dict) = prune_finetune_test(iteration, model_cpy, pruning_step,
                                                                    test_fn, train_fn, app_args,
                                                                    param_name, effective_train_size)
        param = model_cpy.state_dict()[param_name]
        p = mp.Process(target=prune_finetune_test, args=(pqueue, model_cpy, pruning_step, test_fn,
                                                         train_fn, app_args, param_name, param))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        (prec1, prec5, loss, zeros_mask_dict) = pqueue.get()
        if (prec1, prec5, -loss) > (best_layer[0], best_layer[1], -best_layer[-1]):
            best_layer = (prec1, prec5, param_name, model_cpy, zeros_mask_dict, loss)
    return best_layer[:-1]


def get_model_compute_budget(model, dataset, layers_to_prune=None):
    """Return the compute budget of the Convolution layers in an image-classifier.
    """
    dummy_input = distiller.get_dummy_input(dataset)
    g = SummaryGraph(model, dummy_input)
    total_macs = 0
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            # Use the SummaryGraph to obtain some other details of the models
            conv_op = g.find_op(normalize_module_name(name))
            assert conv_op is not None
            total_macs += conv_op['attrs']['MACs']
    del g
    return total_macs


def get_param_densities(model, compressed_model, tensors_to_prune):
    if tensors_to_prune is None:
        tensors_to_prune = [param_name for param_name, _ in model.named_parameters()]

    densities = []
    for param_name in tensors_to_prune:
        if model.state_dict()[param_name].dim() != 4:
            continue
        param = model.state_dict()[param_name]
        compressed_param = compressed_model.state_dict()[param_name]
        densities.append(torch.numel(compressed_param) / torch.numel(param))
    return densities


def create_scheduler(model, zeros_mask_dict):
    scheduler = distiller.CompressionScheduler(model)
    masks = {param_name: masker.mask for param_name, masker in zeros_mask_dict.items()}
    scheduler.load_state_dict(state={'masks_dict': masks})
    return scheduler


def create_network_record_file():
    """Create the CSV file and write the column names"""
    fields = ['iteration', 'top1', 'param_name', 'normalized_macs', 'total_macs', 'densities']
    with open(os.path.join(msglogger.logdir, 'greedy.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def record_network_details(fields):
    """Write the details of one network to a CSV file and create a checkpoint file"""
    with open(os.path.join(msglogger.logdir, 'greedy.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


# This is a temporary hack!
resnet50_params = ["module.layer1.0.conv1.weight", "module.layer1.0.conv2.weight",
                   "module.layer1.1.conv1.weight", "module.layer1.1.conv2.weight",
                   "module.layer1.2.conv1.weight", "module.layer1.2.conv2.weight",
                   "module.layer2.0.conv1.weight", "module.layer2.0.conv2.weight",
                   "module.layer2.1.conv1.weight", "module.layer2.1.conv2.weight",
                   "module.layer2.2.conv1.weight", "module.layer2.2.conv2.weight",
                   "module.layer2.3.conv1.weight", "module.layer2.3.conv2.weight",
                   "module.layer3.0.conv1.weight", "module.layer3.0.conv2.weight",
                   "module.layer3.1.conv1.weight", "module.layer3.1.conv2.weight",
                   "module.layer3.2.conv1.weight", "module.layer3.2.conv2.weight",
                   "module.layer3.3.conv1.weight", "module.layer3.3.conv2.weight",
                   "module.layer3.4.conv1.weight", "module.layer3.4.conv2.weight",
                   "module.layer3.5.conv1.weight", "module.layer3.5.conv2.weight",
                   "module.layer4.0.conv1.weight", "module.layer4.0.conv2.weight",
                   "module.layer4.1.conv1.weight", "module.layer4.1.conv2.weight",
                   "module.layer4.2.conv1.weight", "module.layer4.2.conv2.weight"]

resnet20_params = ["module.layer1.0.conv1.weight", "module.layer1.1.conv1.weight", "module.layer1.2.conv1.weight",
                   "module.layer2.0.conv1.weight", "module.layer2.1.conv1.weight", "module.layer2.2.conv1.weight",
                   "module.layer3.0.conv1.weight", "module.layer3.1.conv1.weight", "module.layer3.2.conv1.weight"]

resnet56_params = ["module.layer1.0.conv1.weight", "module.layer1.1.conv1.weight", "module.layer1.2.conv1.weight",
                   "module.layer1.3.conv1.weight", "module.layer1.4.conv1.weight", "module.layer1.5.conv1.weight",
                   "module.layer1.6.conv1.weight", "module.layer1.7.conv1.weight", "module.layer1.8.conv1.weight",

                   "module.layer2.0.conv1.weight", "module.layer2.1.conv1.weight", "module.layer2.2.conv1.weight",
                   "module.layer2.3.conv1.weight", "module.layer2.4.conv1.weight", "module.layer2.5.conv1.weight",
                   "module.layer2.6.conv1.weight", "module.layer2.7.conv1.weight", "module.layer2.8.conv1.weight",

                   "module.layer3.0.conv1.weight", "module.layer3.1.conv1.weight", "module.layer3.2.conv1.weight",
                   "module.layer3.3.conv1.weight", "module.layer3.4.conv1.weight", "module.layer3.5.conv1.weight",
                   "module.layer3.6.conv1.weight", "module.layer3.7.conv1.weight", "module.layer3.8.conv1.weight"]

mobilenet_params = [
                    #"module.model.0.0.weight", 
                    "module.model.1.3.weight",  "module.model.2.3.weight",
                    "module.model.3.3.weight",  "module.model.4.3.weight",  "module.model.5.3.weight",
                    "module.model.6.3.weight",  "module.model.7.3.weight",  "module.model.8.3.weight",
                    "module.model.9.3.weight",  "module.model.10.3.weight", "module.model.11.3.weight",
                    "module.model.12.3.weight", "module.model.13.3.weight"]

def greedy_pruner(pruned_model, app_args, fraction_to_prune, pruning_step, test_fn, train_fn):
    dataset = app_args.dataset
    arch = app_args.arch
    create_network_record_file()

    # Temporary ugly hack!
    model_layers, model_params = None, None
    if arch == "resnet20_cifar":
        model_params = resnet20_params
    elif arch == "resnet56_cifar":
        model_params = resnet56_params
    elif arch == "resnet50":
        model_params = resnet50_params
    elif arch == "mobilenet":
        model_params = mobilenet_params
    if model_params is not None:
        model_layers = [param[:-len(".weight")] for param in model_params]

    total_macs = dense_total_macs = get_model_compute_budget(pruned_model, dataset, model_layers)
    iteration = 0
    model = pruned_model

    while total_macs > fraction_to_prune * dense_total_macs:
        iteration += 1
        if app_args.greedy_finetuning_policy == "constant":
            effective_train_size = app_args.effective_train_size
        elif app_args.greedy_finetuning_policy == "linear-grow":
            effective_train_size = 1 - (total_macs / dense_total_macs)

        prec1, prec5, param_name, pruned_model, zeros_mask_dict = find_most_robust_layer(iteration, pruned_model,
                                                                                         pruning_step,
                                                                                         test_fn, train_fn,
                                                                                         app_args, model_params,
                                                                                         effective_train_size)
        total_macs = get_model_compute_budget(pruned_model, dataset, model_layers)
        densities = get_param_densities(model, pruned_model, model_params)
        compute_density = total_macs/dense_total_macs
        results = (iteration, prec1, param_name, compute_density, total_macs, densities)
        record_network_details(results)
        scheduler = create_scheduler(pruned_model, zeros_mask_dict)
        save_checkpoint(0, arch, pruned_model, optimizer=None, scheduler=scheduler, extras={'top1': prec1},
                        name="greedy__{}__{:.1f}__{:.1f}".format(str(iteration).zfill(3), compute_density*100, prec1),
                        dir=msglogger.logdir)
        del scheduler
        del zeros_mask_dict
        msglogger.info("Iteration {}: top1-{:.2f} {} compute-{:.2f}".format(*results[0:4]))

    assert iteration > 0
    prec1, prec5, loss = test_fn(model=pruned_model)
    print(prec1, prec5, loss)
    scheduler = create_scheduler(pruned_model, zeros_mask_dict)
    save_checkpoint(0, arch, pruned_model, optimizer=None, scheduler=scheduler, extras={'top1': prec1},
                    name='_'.join(("greedy", str(fraction_to_prune))),
                    dir=msglogger.logdir)

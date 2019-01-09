"""This package contains the implementation of a greedy algorithm which is very similar to NetAdapt and CAR.

We score improvement by means of the computation load (MACs/FLOPs) and do not use direct metrics because this is
an open-source package that runs on many different hardware platforms.


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
import csv
import os
import distiller
from apputils import SummaryGraph, save_checkpoint
from distiller import normalize_module_name


__all__ = ['greedy_pruner']
msglogger = logging.getLogger()


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


def find_most_robust_layer(model, pruning_step, test_func, train_fn, app_args):
    """Find the layer that is most robust to pruning 'pruning_step' filters.

    For each layer: prune 'step' percent of the filters, fine-tune, and measure top1 accuracy
    """
    net_params = [param_name for param_name, _ in model.named_parameters()]
    best_layer = (-np.inf, -np.inf, None, None, None)  # format: (prec1, prec5, param_name, pruned_model, zeros_mask_dict)
    for param_name in net_params:
        if model.state_dict()[param_name].dim() != 4:
            continue

        # Make a copy of the model, because when we prune the parameter tensor the model's weights are altered
        model_cpy = deepcopy(model)
        param = model_cpy.state_dict()[param_name]
        zeros_mask_dict = distiller.create_model_masks_dict(model_cpy)
        if 0 == prune_tensor(param, param_name, pruning_step, zeros_mask_dict):
            continue  # Did not prune anything

        if train_fn is not None:
            # Fine-tune
            optimizer = torch.optim.SGD(model_cpy.parameters(), lr=app_args.lr,
                                        momentum=app_args.momentum, weight_decay=app_args.weight_decay)
            train_fn(model=model_cpy, compression_scheduler=create_scheduler(model_cpy, zeros_mask_dict),
                     optimizer=optimizer, epoch=-1)

        # Test and record the performance of the pruned model
        prec1, prec5, loss = test_func(model=model_cpy, loggers=None)
        if (prec1, prec5) > (best_layer[0], best_layer[1]):
            best_layer = (prec1, prec5, param_name, model_cpy, zeros_mask_dict)
    return best_layer


def get_model_compute_budget(model, dataset):
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
    return total_macs


def create_scheduler(model, zeros_mask_dict):
    scheduler = distiller.CompressionScheduler(model)
    masks = {param_name: masker.mask for param_name, masker in zeros_mask_dict.items()}
    scheduler.load_state_dict(state={'masks_dict': masks})
    return scheduler


def create_network_record_file():
    """Create the CSV file and write the column names"""
    fields = ['iteration', 'top1', 'param_name', 'normalized_macs', 'total_macs']
    with open(os.path.join(msglogger.logdir, 'netadapt.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def record_network_details(fields):
    """Write the details of one network to a CSV file and create a checkpoint file"""
    with open(os.path.join(msglogger.logdir, 'netadapt.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)


def greedy_pruner(pruned_model, app_args, fraction_to_prune, pruning_step, test_fn, train_fn):
    dataset = app_args.dataset
    arch = app_args.arch
    create_network_record_file()
    total_macs = dense_total_macs = get_model_compute_budget(pruned_model, dataset)
    iteration = 0
    while total_macs > fraction_to_prune * dense_total_macs:
        iteration += 1
        prec1, prec5, param_name, pruned_model, zeros_mask_dict = find_most_robust_layer(pruned_model, pruning_step,
                                                                                         test_fn, train_fn, app_args)
        assert distiller.sparsity_3D(zeros_mask_dict[param_name].mask) > 0
        # Physically remove filters
        distiller.remove_filters(pruned_model, zeros_mask_dict, arch, dataset, optimizer=None)

        total_macs = get_model_compute_budget(pruned_model, dataset)
        results = (iteration, prec1, param_name, total_macs/dense_total_macs, total_macs)
        record_network_details(results)
        msglogger.info("Iteration {}: {} {} {}".format(*results[0:4]))

    prec1, prec5, loss = test_fn(model=pruned_model)
    print(prec1, prec5, loss)
    scheduler = create_scheduler(pruned_model, zeros_mask_dict)
    save_checkpoint(0, arch, pruned_model, optimizer=None, best_top1=prec1, scheduler=scheduler,
                    name='_'.join(("netadapt", str(fraction_to_prune))),
                    dir=msglogger.logdir)

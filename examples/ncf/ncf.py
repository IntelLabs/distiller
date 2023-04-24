import os
import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser
import sys

import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import multiprocessing as mp

import utils
from neumf import NeuMF
from dataset import CFTrainDataset, load_test_ratings, load_test_negs
from convert import (TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                     TRAIN_RATINGS_FILENAME)

import distiller
import distiller.quantization as quantization
import distiller.apputils as apputils
from distiller.data_loggers import TensorBoardLogger, PythonLogger

msglogger = None


def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('data', type=str,
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of examples for each iteration')
    parser.add_argument('-f', '--factors', type=int, default=8,
                        help='number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[64, 32, 16, 8],
                        help='size of hidden layers for MLP')
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float,
                        help='stop training early at threshold')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes for evaluating model')
    parser.add_argument('--workers', '-w', type=int, default=8,
                        help='Number of workers for training DataLoader')

    # Distiller Args
    # summary_choices = ['sparsity', 'compute', 'model', 'modules', 'png', 'png_w_params', 'onnx']
    # parser.add_argument('--summary', type=str, choices=summary_choices,
    #                     help='print a summary of the model, and exit - options: ' +
    #                          ' | '.join(summary_choices))
    parser.add_argument('--load', type=str, metavar='PATH')
    parser.add_argument('--reset-optimizer', action='store_true')
    parser.add_argument('--eval', '--evaluate', action='store_true')
    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')
    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used '
                             '(default is to use all available devices)')
    parser.add_argument('--out-dir', '-o', dest='output_dir', default=os.path.join('run', 'neumf'),
                        help='Path to dump logs and checkpoints')
    parser.add_argument('--name', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--log-freq', '--lf', default=100, type=int, metavar='N', help='Logging frequency')
    parser.add_argument('--param-hist', dest='log_params_histograms', action='store_true', default=False,
                        help='log the parameter tensors histograms to file '
                             '(WARNING: this can use significant disk space)')
    parser.add_argument('--split-final', '--sf', action='store_true')
    parser.add_argument('--eval-fp16', action='store_true')
    parser.add_argument('--activation-histograms', '--act-hist',
                        type=distiller.utils.float_range_argparse_checker(exc_min=True),
                        metavar='PORTION_OF_TEST_SET',
                        help='Run the model in evaluation mode on the specified portion of the test dataset and '
                             'generate activation histograms. NOTE: This slows down evaluation significantly')
    quantization.add_post_train_quant_args(parser)

    return parser.parse_args()


def predict(model, users, items, batch_size=1024, use_cuda=True):
    with torch.no_grad():
        batches = [(users[i:i + batch_size], items[i:i + batch_size])
                   for i in range(0, len(users), batch_size)]
        preds = []
        for user, item in batches:
            def proc(x):
                x = np.array(x)
                x = torch.from_numpy(x)
                if use_cuda:
                    x = x.cuda(async=True)
                return torch.autograd.Variable(x)
            outp = model(proc(user), proc(item), torch.tensor([True], dtype=torch.bool))
            outp = outp.data.cpu().numpy()
            preds += list(outp.flatten())
        return preds


def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.


def eval_one(rating, items, model, K, use_cuda=True):
    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    # items.insert(0, test_item)
    users = [user] * len(items)
    predictions = predict(model, users, items, use_cuda=use_cuda)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    # return user, hit, ndcg
    return hit, ndcg


def val_epoch(model, ratings, negs, K, use_cuda=True, output=None, epoch=None,
              processes=1, num_users=-1):
    if epoch is None:
        msglogger.info("Initial evaluation")
    else:
        msglogger.info("Epoch {} evaluation".format(epoch))
    start = datetime.now()
    model.eval()

    if num_users > 0:
        ratings = ratings[:num_users]
        negs = negs[:num_users]

    if processes > 1:
        context = mp.get_context('spawn')
        _eval_one = partial(eval_one, model=model, K=K, use_cuda=use_cuda)
        with context.Pool(processes=processes) as workers:
            hits_and_ndcg = workers.starmap(_eval_one, zip(ratings, negs))
        hits, ndcgs = zip(*hits_and_ndcg)
    else:
        hits, ndcgs = [], []
        with tqdm.tqdm(zip(ratings, negs), total=len(ratings)) as t:
            for rating, items in t:
                hit, ndcg = eval_one(rating, items, model, K, use_cuda=use_cuda)
                hits.append(hit)
                ndcgs.append(ndcg)
                steps_completed = len(hits) + 1
                if steps_completed % 100 == 0:
                    t.set_description('HR@10 = {0:.4f}, NDCG = {1:.4f}'.format(np.mean(hits), np.mean(ndcgs)))

    hits = np.array(hits, dtype=np.float32)
    ndcgs = np.array(ndcgs, dtype=np.float32)

    end = datetime.now()
    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = np.mean(hits)
        result['NDCG'] = np.mean(ndcgs)
        utils.save_result(result, output)

    return hits, ndcgs


def main():
    global msglogger

    script_dir = os.path.dirname(__file__)

    args = parse_args()

    # Distiller loggers
    msglogger = apputils.config_pylogger('logging.conf', args.name, output_dir=args.output_dir)
    tflogger = TensorBoardLogger(msglogger.logdir)
    # tflogger.log_gradients = True
    # pylogger = PythonLogger(msglogger)

    if args.seed is not None:
        msglogger.info("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed)
        np.random.seed(seed=args.seed)

    args.qe_mode = str(args.qe_mode).split('.')[1]
    args.qe_clip_acts = str(args.qe_clip_acts).split('.')[1]

    apputils.log_execution_env_state(sys.argv)

    if args.gpus is not None:
        try:
            args.gpus = [int(s) for s in args.gpus.split(',')]
        except ValueError:
            msglogger.error('ERROR: Argument --gpus must be a comma-separated list of integers only')
            exit(1)
        if len(args.gpus) > 1:
            msglogger.error('ERROR: Only single GPU supported for NCF')
            exit(1)
        available_gpus = torch.cuda.device_count()
        for dev_id in args.gpus:
            if dev_id >= available_gpus:
                msglogger.error('ERROR: GPU device ID {0} requested, but only {1} devices available'
                                .format(dev_id, available_gpus))
                exit(1)
        # Set default device in case the first one on the list != 0
        torch.cuda.set_device(args.gpus[0])

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())
    run_dir = msglogger.logdir
    msglogger.info("Saving config and results to {}".format(run_dir))
    if not os.path.exists(run_dir) and run_dir != '':
        os.makedirs(run_dir)
    utils.save_config(config, run_dir)

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    t1 = time.time()
    # Load Data
    training = not (args.eval or args.qe_calibration or args.activation_histograms)
    msglogger.info('Loading data')
    if training:
        train_dataset = CFTrainDataset(
            os.path.join(args.data, TRAIN_RATINGS_FILENAME), args.negative_samples)
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
        nb_users, nb_items = train_dataset.nb_users, train_dataset.nb_items
    else:
        train_dataset = None
        train_dataloader = None
        nb_users, nb_items = (138493, 26744)

    test_ratings = load_test_ratings(os.path.join(args.data, TEST_RATINGS_FILENAME))  # noqa: E501
    test_negs = load_test_negs(os.path.join(args.data, TEST_NEG_FILENAME))

    msglogger.info('Load data done [%.1f s]. #user=%d, #item=%d, #train=%s, #test=%d'
              % (time.time()-t1, nb_users, nb_items, str(train_dataset.mat.nnz) if training else 'N/A',
                 len(test_ratings)))

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors, mf_reg=0.,
                  mlp_layer_sizes=args.layers,
                  mlp_layer_regs=[0. for i in args.layers],
                  split_final=args.split_final)
    if use_cuda:
        model = model.cuda()
    msglogger.info(model)
    msglogger.info("{} parameters".format(utils.count_parameters(model)))

    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))

    compression_scheduler = None
    start_epoch = 0
    optimizer = None
    if args.load:
        if training:
            model, compression_scheduler, optimizer, start_epoch = apputils.load_checkpoint(model, args.load)
            if args.reset_optimizer:
                start_epoch = 0
                optimizer = None
        else:
            model = apputils.load_lean_checkpoint(model, args.load)

    # Add loss to graph
    criterion = nn.BCEWithLogitsLoss()

    if use_cuda:
        criterion = criterion.cuda()

    if training and optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        msglogger.info('Optimizer Type: %s', type(optimizer))
        msglogger.info('Optimizer Args: %s', optimizer.defaults)

    if args.compress:
        compression_scheduler = distiller.file_config(model, optimizer, args.compress)
        model.cuda()

    # Create files for tracking training
    valid_results_file = os.path.join(run_dir, 'valid_results.csv')

    if args.qe_calibration or args.activation_histograms:
        calib = {'portion': args.qe_calibration,
                 'desc_str': 'quantization calibration stats',
                 'collect_func': partial(distiller.data_loggers.collect_quant_stats, inplace_runtime_check=True,
                                         disable_inplace_attrs=True)}
        hists = {'portion': args.activation_histograms,
                 'desc_str': 'activation histograms',
                 'collect_func': partial(distiller.data_loggers.collect_histograms, activation_stats=None, nbins=2048,
                                         save_hist_imgs=True)}
        d = calib if args.qe_calibration else hists

        distiller.utils.assign_layer_fq_names(model)
        num_users = int(np.floor(len(test_ratings) * d['portion']))
        msglogger.info(
            "Generating {} based on {:.1%} of the test-set ({} users)".format(d['desc_str'], d['portion'], num_users))

        test_fn = partial(val_epoch, ratings=test_ratings, negs=test_negs, K=args.topk, use_cuda=use_cuda,
                          processes=args.processes, num_users=num_users)
        d['collect_func'](model=model, test_fn=test_fn, save_dir=run_dir, classes=None)

        return 0

    if args.eval:
        if args.quantize_eval and args.qe_calibration is None:
            model.cpu()
            quantizer = quantization.PostTrainLinearQuantizer.from_args(model, args)
            dummy_input = (torch.tensor([1]), torch.tensor([1]), torch.tensor([True], dtype=torch.bool))
            quantizer.prepare_model(dummy_input)
            model.cuda()

        distiller.utils.assign_layer_fq_names(model)

        if args.eval_fp16:
            model = model.half()

        # Calculate initial Hit Ratio and NDCG
        begin = time.time()
        hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                                use_cuda=use_cuda, processes=args.processes)
        val_time = time.time() - begin
        hit_rate = np.mean(hits)
        msglogger.info('Initial HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}, val_time = {val_time:.2f}'
                       .format(K=args.topk, hit_rate=hit_rate, ndcg=np.mean(ndcgs), val_time=val_time))
        hit_rate = 0

        if args.quantize_eval:
            checkpoint_name = 'quantized'
            apputils.save_checkpoint(0, 'NCF', model, optimizer=None, extras={'quantized_hr@10': hit_rate},
                                     name='_'.join([args.name, 'quantized']) if args.name else checkpoint_name,
                                     dir=msglogger.logdir)
        return 0

    total_samples = len(train_dataloader.sampler)
    steps_per_epoch = math.ceil(total_samples / args.batch_size)
    best_hit_rate = 0
    best_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        msglogger.info('')
        model.train()
        losses = utils.AverageMeter()

        begin = time.time()

        if compression_scheduler:
            compression_scheduler.on_epoch_begin(epoch, optimizer)

        loader = tqdm.tqdm(train_dataloader)
        for batch_index, (user, item, label) in enumerate(loader):
            user = torch.autograd.Variable(user, requires_grad=False)
            item = torch.autograd.Variable(item, requires_grad=False)
            label = torch.autograd.Variable(label, requires_grad=False)
            if use_cuda:
                user = user.cuda(async=True)
                item = item.cuda(async=True)
                label = label.cuda(async=True)

            if compression_scheduler:
                compression_scheduler.on_minibatch_begin(epoch, batch_index, steps_per_epoch, optimizer)

            outputs = model(user, item, torch.tensor([False], dtype=torch.bool))
            loss = criterion(outputs, label)

            if compression_scheduler:
                compression_scheduler.before_backward_pass(epoch, batch_index, steps_per_epoch, loss, optimizer,
                                                           return_loss_components=False)

            losses.update(loss.data.item(), user.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if compression_scheduler:
                compression_scheduler.on_minibatch_end(epoch, batch_index, steps_per_epoch, optimizer)

            # Save stats to file
            description = ('Epoch {} Loss {loss.val:.4f} ({loss.avg:.4f})'
                           .format(epoch, loss=losses))
            loader.set_description(description)

            steps_completed = batch_index + 1
            if steps_completed % args.log_freq == 0:
                stats_dict = OrderedDict()
                stats_dict['Loss'] = losses.avg
                stats = ('Performance/Training/', stats_dict)
                params = model.named_parameters() if args.log_params_histograms else None
                distiller.log_training_progress(stats, params, epoch, steps_completed, steps_per_epoch, args.log_freq,
                                                [tflogger])

                tflogger.log_model_buffers(model, ['tracked_min', 'tracked_max'], 'Quant/Train/Acts/TrackedMinMax',
                                           epoch, steps_completed, steps_per_epoch, args.log_freq)

        train_time = time.time() - begin
        begin = time.time()
        hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                                use_cuda=use_cuda, output=valid_results_file,
                                epoch=epoch, processes=args.processes)
        val_time = time.time() - begin

        if compression_scheduler:
            compression_scheduler.on_epoch_end(epoch, optimizer)

        hit_rate = np.mean(hits)
        mean_ndcgs = np.mean(ndcgs)

        stats_dict = OrderedDict()
        stats_dict['HR@{0}'.format(args.topk)] = hit_rate
        stats_dict['NDCG@{0}'.format(args.topk)] = mean_ndcgs
        stats = ('Performance/Validation/', stats_dict)
        distiller.log_training_progress(stats, None, epoch, steps_completed=0, total_steps=1, log_freq=1,
                                        loggers=[tflogger])

        msglogger.info('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}, AvgTrainLoss = {loss.avg:.4f}, '
                       'train_time = {train_time:.2f}, val_time = {val_time:.2f}'.format(
                            epoch=epoch, K=args.topk, hit_rate=hit_rate, ndcg=mean_ndcgs,
                            loss=losses, train_time=train_time, val_time=val_time))

        is_best = False
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            is_best = True
            best_epoch = epoch
        extras = {'current_hr@10': hit_rate,
                  'best_hr@10': best_hit_rate,
                  'best_epoch': best_epoch}
        apputils.save_checkpoint(epoch, 'NCF', model, optimizer, compression_scheduler, extras, is_best, dir=run_dir)

        if args.threshold is not None:
            if np.mean(hits) >= args.threshold:
                msglogger.info("Hit threshold of {}".format(args.threshold))
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n-- KeyboardInterrupt --")
    finally:
        if msglogger is not None:
            msglogger.info('')
            msglogger.info('Log file for this run: ' + os.path.realpath(msglogger.log_filename))

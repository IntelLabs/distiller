#!/usr/bin/env python
import argparse
import os
import logging
from ast import literal_eval
import subprocess

import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
import torch.optim

from mlperf_compliance import mlperf_log

from seq2seq import models
from seq2seq.train.smoothing import LabelSmoothing
from seq2seq.data.dataset import ParallelDataset
from seq2seq.data.tokenizer import Tokenizer
from seq2seq.utils import setup_logging
import seq2seq.data.config as config
import seq2seq.train.trainer as trainers
from seq2seq.inference.inference import Translator


def parse_args():
    parser = argparse.ArgumentParser(description='GNMT training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # dataset
    dataset = parser.add_argument_group('dataset setup')
    dataset.add_argument('--dataset-dir', default=None, required=True,
                         help='path to directory with training/validation data')
    dataset.add_argument('--max-size', default=None, type=int,
                         help='use at most MAX_SIZE elements from training \
                        dataset (useful for benchmarking), by default \
                        uses entire dataset')

    # results
    results = parser.add_argument_group('results setup')
    results.add_argument('--results-dir', default='../results',
                         help='path to directory with results, it it will be \
                        automatically created if does not exist')
    results.add_argument('--save', default='gnmt_wmt16',
                         help='defines subdirectory within RESULTS_DIR for \
                        results from this training run')
    results.add_argument('--print-freq', default=10, type=int,
                         help='print log every PRINT_FREQ batches')

    # model
    model = parser.add_argument_group('model setup')
    model.add_argument('--model-config',
                       default="{'hidden_size': 1024,'num_layers': 4, \
                        'dropout': 0.2, 'share_embedding': True}",
                       help='GNMT architecture configuration')
    model.add_argument('--smoothing', default=0.1, type=float,
                       help='label smoothing, if equal to zero model will use \
                        CrossEntropyLoss, if not zero model will be trained \
                        with label smoothing loss based on KLDivLoss')

    # setup
    general = parser.add_argument_group('general setup')
    general.add_argument('--math', default='fp32', choices=['fp32', 'fp16'],
                         help='arithmetic type')
    general.add_argument('--seed', default=None, type=int,
                         help='set random number generator seed')
    general.add_argument('--disable-eval', action='store_true', default=False,
                         help='disables validation after every epoch')
    general.add_argument('--workers', default=0, type=int,
                         help='number of workers for data loading')

    cuda_parser = general.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true',
                             help='enables cuda (use \'--no-cuda\' to disable)')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                             help=argparse.SUPPRESS)
    cuda_parser.set_defaults(cuda=True)

    cudnn_parser = general.add_mutually_exclusive_group(required=False)
    cudnn_parser.add_argument('--cudnn', dest='cudnn', action='store_true',
                              help='enables cudnn (use \'--no-cudnn\' to disable)')
    cudnn_parser.add_argument('--no-cudnn', dest='cudnn', action='store_false',
                              help=argparse.SUPPRESS)
    cudnn_parser.set_defaults(cudnn=True)

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--batch-size', default=128, type=int,
                          help='batch size for training')
    training.add_argument('--epochs', default=8, type=int,
                          help='number of total epochs to run')
    training.add_argument('--optimization-config',
                          default="{'optimizer': 'Adam', 'lr': 5e-4}", type=str,
                          help='optimizer config')
    training.add_argument('--grad-clip', default=5.0, type=float,
                          help='enabled gradient clipping and sets maximum \
                        gradient norm value')
    training.add_argument('--max-length-train', default=50, type=int,
                          help='maximum sequence length for training')
    training.add_argument('--min-length-train', default=0, type=int,
                          help='minimum sequence length for training')
    training.add_argument('--target-bleu', default=None, type=float,
                          help='target accuracy')

    bucketing_parser = training.add_mutually_exclusive_group(required=False)
    bucketing_parser.add_argument('--bucketing', dest='bucketing', action='store_true',
                             help='enables bucketing (use \'--no-bucketing\' to disable)')
    bucketing_parser.add_argument('--no-bucketing', dest='bucketing', action='store_false',
                             help=argparse.SUPPRESS)
    bucketing_parser.set_defaults(bucketing=True)

    # validation
    validation = parser.add_argument_group('validation setup')
    validation.add_argument('--eval-batch-size', default=32, type=int,
                            help='batch size for validation')
    validation.add_argument('--max-length-val', default=150, type=int,
                            help='maximum sequence length for validation')
    validation.add_argument('--min-length-val', default=0, type=int,
                            help='minimum sequence length for validation')

    validation.add_argument('--beam-size', default=5, type=int,
                        help='beam size')
    validation.add_argument('--len-norm-factor', default=0.6, type=float,
                        help='length normalization factor')
    validation.add_argument('--cov-penalty-factor', default=0.1, type=float,
                        help='coverage penalty factor')
    validation.add_argument('--len-norm-const', default=5.0, type=float,
                        help='length normalization constant')


    # checkpointing
    checkpoint = parser.add_argument_group('checkpointing setup')
    checkpoint.add_argument('--start-epoch', default=0, type=int,
                            help='manually set initial epoch counter')
    checkpoint.add_argument('--resume', default=None, type=str, metavar='PATH',
                            help='resumes training from checkpoint from PATH')
    checkpoint.add_argument('--save-all', action='store_true', default=False,
                            help='saves checkpoint after every epoch')
    checkpoint.add_argument('--save-freq', default=5000, type=int,
                            help='save checkpoint every SAVE_FREQ batches')
    checkpoint.add_argument('--keep-checkpoints', default=0, type=int,
                            help='keep only last KEEP_CHECKPOINTS checkpoints, \
                        affects only checkpoints controlled by --save-freq \
                        option')

    # distributed support
    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--rank', default=0, type=int,
                             help='rank of the process, do not set! Done by multiproc module')
    distributed.add_argument('--world-size', default=1, type=int,
                             help='number of processes, do not set! Done by multiproc module')
    distributed.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                             help='url used to set up distributed training')

    return parser.parse_args()


def build_criterion(vocab_size, padding_idx, smoothing):
    if smoothing == 0.:
        logging.info(f'building CrossEntropyLoss')
        loss_weight = torch.ones(vocab_size)
        loss_weight[padding_idx] = 0
        criterion = nn.CrossEntropyLoss(weight=loss_weight, size_average=False)
        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                              value='Cross Entropy')
    else:
        logging.info(f'building SmoothingLoss (smoothing: {smoothing})')
        criterion = LabelSmoothing(padding_idx, smoothing)
        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_LOSS_FN,
                              value='Cross Entropy with label smoothing')
        mlperf_log.gnmt_print(key=mlperf_log.MODEL_HP_LOSS_SMOOTHING,
                              value=smoothing)

    return criterion


def main():
    mlperf_log.ROOT_DIR_GNMT = os.path.dirname(os.path.abspath(__file__))
    mlperf_log.LOGGER.propagate = False
    mlperf_log.gnmt_print(key=mlperf_log.RUN_START)

    args = parse_args()
    print(args)

    if not args.cudnn:
        torch.backends.cudnn.enabled = False
    mlperf_log.gnmt_print(key=mlperf_log.RUN_SET_RANDOM_SEED)
    if args.seed:
        torch.manual_seed(args.seed + args.rank)

    # initialize distributed backend
    distributed = args.world_size > 1
    if distributed:
        backend = 'nccl' if args.cuda else 'gloo'
        dist.init_process_group(backend=backend, rank=args.rank,
                                init_method=args.dist_url,
                                world_size=args.world_size)

    # create directory for results
    save_path = os.path.join(args.results_dir, args.save)
    os.makedirs(save_path, exist_ok=True)

    # setup logging
    log_filename = f'log_gpu_{args.rank}.log'
    setup_logging(os.path.join(save_path, log_filename))

    logging.info(f'Saving results to: {save_path}')
    logging.info(f'Run arguments: {args}')

    if args.cuda:
        torch.cuda.set_device(args.rank)

    # build tokenizer
    tokenizer = Tokenizer(os.path.join(args.dataset_dir, config.VOCAB_FNAME))

    # build datasets
    mlperf_log.gnmt_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
    mlperf_log.gnmt_print(key=mlperf_log.TRAIN_HP_MAX_SEQ_LEN,
                          value=args.max_length_train)

    train_data = ParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TRAIN_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_TRAIN_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_train,
        max_len=args.max_length_train,
        sort=False,
        max_size=args.max_size)

    mlperf_log.gnmt_print(key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES,
                          value=len(train_data))

    val_data = ParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_VAL_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_VAL_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_val,
        max_len=args.max_length_val,
        sort=True)

    mlperf_log.gnmt_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

    test_data = ParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TEST_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_TEST_FNAME),
        tokenizer=tokenizer,
        min_len=args.min_length_val,
        max_len=args.max_length_val,
        sort=False)

    mlperf_log.gnmt_print(key=mlperf_log.PREPROC_NUM_EVAL_EXAMPLES,
                          value=len(test_data))

    vocab_size = tokenizer.vocab_size
    mlperf_log.gnmt_print(key=mlperf_log.PREPROC_VOCAB_SIZE, value=vocab_size)

    # build GNMT model
    model_config = dict(vocab_size=vocab_size, math=args.math,
                        **literal_eval(args.model_config))
    model = models.GNMT(**model_config)
    logging.info(model)

    batch_first = model.batch_first

    # define loss function (criterion) and optimizer
    criterion = build_criterion(vocab_size, config.PAD, args.smoothing)
    opt_config = literal_eval(args.optimization_config)
    logging.info(f'Training optimizer: {opt_config}')

    # create trainer
    trainer_options = dict(
        criterion=criterion,
        grad_clip=args.grad_clip,
        save_path=save_path,
        save_freq=args.save_freq,
        save_info={'config': args, 'tokenizer': tokenizer},
        opt_config=opt_config,
        batch_first=batch_first,
        keep_checkpoints=args.keep_checkpoints,
        math=args.math,
        print_freq=args.print_freq,
        cuda=args.cuda,
        distributed=distributed)

    trainer_options['model'] = model
    trainer = trainers.Seq2SeqTrainer(**trainer_options)

    translator = Translator(model,
                            tokenizer,
                            beam_size=args.beam_size,
                            max_seq_len=args.max_length_val,
                            len_norm_factor=args.len_norm_factor,
                            len_norm_const=args.len_norm_const,
                            cov_penalty_factor=args.cov_penalty_factor,
                            cuda=args.cuda)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info(f'Number of parameters: {num_parameters}')

    # optionally resume from a checkpoint
    if args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth')
        if os.path.isfile(checkpoint_file):
            trainer.load(checkpoint_file)
        else:
            logging.error(f'No checkpoint found at {args.resume}')

    # get data loaders
    train_loader = train_data.get_loader(batch_size=args.batch_size,
                                         batch_first=batch_first,
                                         shuffle=True,
                                         bucket=args.bucketing,
                                         num_workers=args.workers,
                                         drop_last=True,
                                         distributed=distributed)

    mlperf_log.gnmt_print(key=mlperf_log.INPUT_BATCH_SIZE,
                          value=args.batch_size * args.world_size)
    mlperf_log.gnmt_print(key=mlperf_log.INPUT_SIZE,
                          value=train_loader.sampler.num_samples)


    val_loader = val_data.get_loader(batch_size=args.eval_batch_size,
                                     batch_first=batch_first,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     drop_last=False,
                                     distributed=False)

    test_loader = test_data.get_loader(batch_size=args.eval_batch_size,
                                       batch_first=batch_first,
                                       shuffle=False,
                                       num_workers=0,
                                       drop_last=False,
                                       distributed=False)

    mlperf_log.gnmt_print(key=mlperf_log.EVAL_SIZE,
                          value=len(test_loader.sampler))

    # training loop
    best_loss = float('inf')
    mlperf_log.gnmt_print(key=mlperf_log.TRAIN_LOOP)
    for epoch in range(args.start_epoch, args.epochs):
        mlperf_log.gnmt_print(key=mlperf_log.TRAIN_EPOCH,
                              value=epoch)
        logging.info(f'Starting epoch {epoch}')

        if distributed:
            train_loader.sampler.set_epoch(epoch)

        trainer.epoch = epoch
        train_loss = trainer.optimize(train_loader)

        # evaluate on validation set
        if args.rank == 0 and not args.disable_eval:
            logging.info(f'Running validation on dev set')
            val_loss = trainer.evaluate(val_loader)

            # remember best prec@1 and save checkpoint
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            mlperf_log.gnmt_print(key=mlperf_log.TRAIN_CHECKPOINT)
            trainer.save(save_all=args.save_all, is_best=is_best)

            logging.info(f'Epoch: {epoch}\t'
                         f'Training Loss {train_loss:.4f}\t'
                         f'Validation Loss {val_loss:.4f}')
        else:
            logging.info(f'Epoch: {epoch}\t'
                         f'Training Loss {train_loss:.4f}')

        if args.cuda:
            break_training = torch.cuda.LongTensor([0])
        else:
            break_training = torch.LongTensor([0])

        if args.rank == 0 and not args.disable_eval:
            logging.info(f'Running evaluation on test set')
            mlperf_log.gnmt_print(key=mlperf_log.EVAL_START, value=epoch)

            model.eval()
            torch.cuda.empty_cache()

            eval_path = os.path.join(save_path, f'eval_epoch_{epoch}')
            eval_file = open(eval_path, 'w')

            for i, (src, tgt, indices) in enumerate(test_loader):
                src, src_length = src

                if translator.batch_first:
                    batch_size = src.size(0)
                else:
                    batch_size = src.size(1)
                beam_size = args.beam_size

                bos = [translator.insert_target_start] * (batch_size * beam_size)
                bos = torch.LongTensor(bos)
                if translator.batch_first:
                    bos = bos.view(-1, 1)
                else:
                    bos = bos.view(1, -1)

                src_length = torch.LongTensor(src_length)

                if args.cuda:
                    src = src.cuda()
                    src_length = src_length.cuda()
                    bos = bos.cuda()

                with torch.no_grad():
                    context = translator.model.encode(src, src_length)
                    context = [context, src_length, None]

                    if beam_size == 1:
                        generator = translator.generator.greedy_search
                    else:
                        generator = translator.generator.beam_search
                    preds, lengths, counter = generator(batch_size, bos, context)

                preds = preds.cpu()
                lengths = lengths.cpu()

                output = []
                for idx, pred in enumerate(preds):
                    end = lengths[idx] - 1
                    pred = pred[1: end]
                    pred = pred.tolist()
                    out = translator.tok.detokenize(pred)
                    output.append(out)

                output = [output[indices.index(i)] for i in range(len(output))]
                for line in output:
                    eval_file.write(line)
                    eval_file.write('\n')

            eval_file.close()

            # run moses detokenizer
            detok_path = os.path.join(args.dataset_dir, config.DETOKENIZER)
            detok_eval_path = eval_path + '.detok'

            with open(detok_eval_path, 'w') as detok_eval_file,  \
                    open(eval_path, 'r') as eval_file:
                subprocess.run(['perl', f'{detok_path}'], stdin=eval_file,
                               stdout=detok_eval_file, stderr=subprocess.DEVNULL)

            # run sacrebleu
            reference_path = os.path.join(args.dataset_dir, config.TGT_TEST_TARGET_FNAME)
            sacrebleu = subprocess.run([f'sacrebleu --input {detok_eval_path} \
                                        {reference_path} --score-only -lc --tokenize intl'],
                                       stdout=subprocess.PIPE, shell=True)
            bleu = float(sacrebleu.stdout.strip())
            logging.info(f'Finished evaluation on test set')
            logging.info(f'BLEU on test dataset: {bleu}')

            if args.target_bleu:
                if bleu >= args.target_bleu:
                    logging.info(f'Target accuracy reached')
                    break_training[0] = 1

            torch.cuda.empty_cache()
            mlperf_log.gnmt_print(key=mlperf_log.EVAL_ACCURACY,
                                  value={"epoch": epoch, "value": bleu})
            mlperf_log.gnmt_print(key=mlperf_log.EVAL_TARGET,
                                  value=args.target_bleu)
            mlperf_log.gnmt_print(key=mlperf_log.EVAL_STOP)

        if distributed:
            dist.broadcast(break_training, 0)

        logging.info(f'Finished epoch {epoch}')
        if break_training:
            break

    mlperf_log.gnmt_print(key=mlperf_log.RUN_STOP,
                         value={"success": bool(break_training)})
    mlperf_log.gnmt_print(key=mlperf_log.RUN_FINAL)

if __name__ == '__main__':
    main()

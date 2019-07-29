
# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import codecs
import time
import warnings
from ast import literal_eval
from itertools import zip_longest

import torch

from seq2seq import models
from seq2seq.inference.inference import Translator
from seq2seq.utils import AverageMeter
import subprocess
import os
import seq2seq.data.config as config
from seq2seq.data.dataset import ParallelDataset
import logging
from seq2seq.utils import AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='GNMT Translate',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # data
    dataset = parser.add_argument_group('data setup')
    dataset.add_argument('--dataset-dir', default=None, required=True,
                         help='path to directory with input data')
    dataset.add_argument('-i', '--input', required=True,
                         help='input file (tokenized)')
    dataset.add_argument('-o', '--output', required=True,
                         help='output file (tokenized)')
    dataset.add_argument('-m', '--model', required=True,
                         help='model checkpoint file')
    dataset.add_argument('-r', '--reference', default=None,
                         help='full path to the file with reference \
                         translations (for sacrebleu)')

    # parameters
    params = parser.add_argument_group('inference setup')
    params.add_argument('--batch-size', default=128, type=int,
                        help='batch size')
    params.add_argument('--beam-size', default=5, type=int,
                        help='beam size')
    params.add_argument('--max-seq-len', default=80, type=int,
                        help='maximum prediciton sequence length')
    params.add_argument('--len-norm-factor', default=0.6, type=float,
                        help='length normalization factor')
    params.add_argument('--cov-penalty-factor', default=0.1, type=float,
                        help='coverage penalty factor')
    params.add_argument('--len-norm-const', default=5.0, type=float,
                        help='length normalization constant')
    # general setup
    general = parser.add_argument_group('general setup')

    general.add_argument('--mode', default='accuracy', choices=['accuracy',
            'performance'], help='test in accuracy or performance mode')

    general.add_argument('--math', default='fp16', choices=['fp32', 'fp16'],
                         help='arithmetic type')

    batch_first_parser = general.add_mutually_exclusive_group(required=False)
    batch_first_parser.add_argument('--batch-first', dest='batch_first',
                                    action='store_true',
                                    help='uses (batch, seq, feature) data \
                                    format for RNNs')
    batch_first_parser.add_argument('--seq-first', dest='batch_first',
                                    action='store_false',
                                    help='uses (seq, batch, feature) data \
                                    format for RNNs')
    batch_first_parser.set_defaults(batch_first=True)

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

    general.add_argument('--print-freq', '-p', default=1, type=int,
                         help='print log every PRINT_FREQ batches')

    return parser.parse_args()


def grouper(iterable, size, fillvalue=None):
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=fillvalue)


def write_output(output_file, lines):
    for line in lines:
        output_file.write(line)
        output_file.write('\n')


def checkpoint_from_distributed(state_dict):
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


def unwrap_distributed(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = value

    return new_state_dict


def main():
    args = parse_args()
    print(args)

    if args.cuda:
        torch.cuda.set_device(0)
    if not args.cuda and torch.cuda.is_available():
        warnings.warn('cuda is available but not enabled')
    if args.math == 'fp16' and not args.cuda:
        raise RuntimeError('fp16 requires cuda')
    if not args.cudnn:
        torch.backends.cudnn.enabled = False

    checkpoint = torch.load(args.model, map_location={'cuda:0': 'cpu'})

    vocab_size = checkpoint['tokenizer'].vocab_size
    model_config = dict(vocab_size=vocab_size, math=checkpoint['config'].math,
                        **literal_eval(checkpoint['config'].model_config))
    model_config['batch_first'] = args.batch_first
    model = models.GNMT(**model_config)

    state_dict = checkpoint['state_dict']
    if checkpoint_from_distributed(state_dict):
        state_dict = unwrap_distributed(state_dict)

    model.load_state_dict(state_dict)

    if args.math == 'fp32':
        dtype = torch.FloatTensor
    if args.math == 'fp16':
        dtype = torch.HalfTensor

    model.type(dtype)
    if args.cuda:
        model = model.cuda()
    model.eval()

    tokenizer = checkpoint['tokenizer']


    test_data = ParallelDataset(
        src_fname=os.path.join(args.dataset_dir, config.SRC_TEST_FNAME),
        tgt_fname=os.path.join(args.dataset_dir, config.TGT_TEST_FNAME),
        tokenizer=tokenizer,
        min_len=0,
        max_len=150,
        sort=False)

    test_loader = test_data.get_loader(batch_size=args.batch_size,
                                       batch_first=True,
                                       shuffle=False,
                                       num_workers=0,
                                       drop_last=False,
                                       distributed=False)

    translator = Translator(model,
                                   tokenizer,
                                   beam_size=args.beam_size,
                                   max_seq_len=args.max_seq_len,
                                   len_norm_factor=args.len_norm_factor,
                                   len_norm_const=args.len_norm_const,
                                   cov_penalty_factor=args.cov_penalty_factor,
                                   cuda=args.cuda)

    model.eval()
    torch.cuda.empty_cache()

    # only write the output to file in accuracy mode
    if args.mode == 'accuracy':
        test_file = open(args.output, 'w', encoding='UTF-8')

    batch_time = AverageMeter(False)
    tot_tok_per_sec = AverageMeter(False)
    iterations = AverageMeter(False)
    enc_seq_len = AverageMeter(False)
    dec_seq_len = AverageMeter(False)
    stats = {}

    for i, (src, tgt, indices) in enumerate(test_loader):
        translate_timer = time.time()
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
        stats['total_enc_len'] = int(src_length.sum())

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

        stats['total_dec_len'] = lengths.sum().item()
        stats['iters'] = counter

        preds = preds.cpu()
        lengths = lengths.cpu()

        output = []
        for idx, pred in enumerate(preds):
            end = lengths[idx] - 1
            pred = pred[1: end]
            pred = pred.tolist()
            out = translator.tok.detokenize(pred)
            output.append(out)

        # only write the output to file in accuracy mode
        if args.mode == 'accuracy':
            output = [output[indices.index(i)] for i in range(len(output))]
            for line in output:
                test_file.write(line)
                test_file.write('\n')


        # Get timing
        elapsed = time.time() - translate_timer
        batch_time.update(elapsed, batch_size)

        total_tokens = stats['total_dec_len'] + stats['total_enc_len']
        ttps = total_tokens / elapsed
        tot_tok_per_sec.update(ttps, batch_size)

        iterations.update(stats['iters'])
        enc_seq_len.update(stats['total_enc_len'] / batch_size, batch_size)
        dec_seq_len.update(stats['total_dec_len'] / batch_size, batch_size)

        if i % 5 == 0:
            log = []
            log += 'TEST '
            log += 'Time {:.3f} ({:.3f})\t'.format(batch_time.val, batch_time.avg)
            log += 'Decoder iters {:.1f} ({:.1f})\t'.format(iterations.val, iterations.avg)
            log += 'Tok/s {:.0f} ({:.0f})'.format(tot_tok_per_sec.val, tot_tok_per_sec.avg)
            log = ''.join(log)
            print(log)


    # summary timing
    time_per_sentence = (batch_time.avg / batch_size)
    log = []
    log += 'TEST SUMMARY:\n'
    log += 'Lines translated: {}\t'.format(len(test_loader.dataset))
    log += 'Avg total tokens/s: {:.0f}\n'.format(tot_tok_per_sec.avg)
    log += 'Avg time per batch: {:.3f} s\t'.format(batch_time.avg)
    log += 'Avg time per sentence: {:.3f} ms\n'.format(1000 * time_per_sentence)
    log += 'Avg encoder seq len: {:.2f}\t'.format(enc_seq_len.avg)
    log += 'Avg decoder seq len: {:.2f}\t'.format(dec_seq_len.avg)
    log += 'Total decoder iterations: {}'.format(int(iterations.sum))
    log = ''.join(log)
    print(log)

    # only write the output to file in accuracy mode
    if args.mode == 'accuracy':
        test_file.close()

        test_path = args.output
        # run moses detokenizer
        detok_path = os.path.join(args.dataset_dir, config.DETOKENIZER)
        detok_test_path = test_path + '.detok'

        with open(detok_test_path, 'w') as detok_test_file, \
                open(test_path, 'r') as test_file:
            subprocess.run(['perl', detok_path], stdin=test_file,
                           stdout=detok_test_file, stderr=subprocess.DEVNULL)


        # run sacrebleu
        reference_path = os.path.join(args.dataset_dir,
                                      config.TGT_TEST_TARGET_FNAME)
        sacrebleu = subprocess.run(['sacrebleu --input {} {} --score-only -lc --tokenize intl'.format(detok_test_path,
                                                                                                      reference_path)],
                                   stdout=subprocess.PIPE, shell=True)
        bleu = float(sacrebleu.stdout.strip())

        print('BLEU on test dataset: {}'.format(bleu))

        print('Finished evaluation on test set')

if __name__ == '__main__':
    main()

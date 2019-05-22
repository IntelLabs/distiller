from contextlib import contextmanager
import os
import logging.config

import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence

import seq2seq.data.config as config


def barrier():
    """ Calls all_reduce on dummy tensor."""
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(torch.cuda.FloatTensor(1))
        torch.cuda.synchronize()


def get_rank():
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    return rank

def get_world_size():
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1
    return world_size


@contextmanager
def sync_workers():
    """ Gets distributed rank and synchronizes workers at exit"""
    rank = get_rank()
    yield rank
    barrier()


def setup_logging(log_file='log.log'):
    """Setup logging configuration
    """
    class RankFilter(logging.Filter):
        def __init__(self, rank):
            self.rank = rank

        def filter(self, record):
            record.rank = self.rank
            return True

    rank = get_rank()
    rank_filter = RankFilter(rank)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(rank)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(rank)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('').addFilter(rank_filter)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, skip_first=True):
        self.reset()
        self.skip = skip_first

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val

        if self.skip:
            self.skip = False
        else:
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


def batch_padded_sequences(seq, batch_first=False, sort=False):
    if sort:
        key = lambda item: len(item[1])
        indices, seq = zip(*sorted(enumerate(seq), key=key, reverse=True))
    else:
        indices = range(len(seq))

    lengths = [len(sentence) for sentence in seq]
    batch_length = max(lengths)
    seq_tensor = torch.LongTensor(batch_length, len(seq)).fill_(config.PAD)
    for idx, sentence in enumerate(seq):
        end_seq = lengths[idx]
        seq_tensor[:end_seq, idx].copy_(sentence[:end_seq])
    if batch_first:
        seq_tensor = seq_tensor.t()
    return seq_tensor, lengths, indices


def debug_tensor(tensor, name):
    logging.info(name)
    tensor = tensor.float().cpu().numpy()
    logging.info(f'MIN: {tensor.min()} MAX: {tensor.max()} '
                 f'AVG: {tensor.mean()} STD: {tensor.std()} '
                 f'NAN: {np.isnan(tensor).sum()} INF: {np.isinf(tensor).sum()}')

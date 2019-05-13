import logging

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from seq2seq.data.sampler import BucketingSampler
from torch.utils.data import DataLoader

import seq2seq.data.config as config


def build_collate_fn(batch_first=False, sort=False):
    def collate_seq(seq):
        lengths = [len(s) for s in seq]
        batch_length = max(lengths)

        shape = (batch_length, len(seq))
        seq_tensor = torch.full(shape, config.PAD, dtype=torch.int64)

        for i, s in enumerate(seq):
            end_seq = lengths[i]
            seq_tensor[:end_seq, i].copy_(s[:end_seq])

        if batch_first:
            seq_tensor = seq_tensor.t()

        return (seq_tensor, lengths)

    def collate(seqs):
        src_seqs, tgt_seqs = zip(*seqs)
        if sort:
            key = lambda item: len(item[1])
            indices, src_seqs = zip(*sorted(enumerate(src_seqs), key=key,
                                        reverse=True))
            tgt_seqs = [tgt_seqs[idx] for idx in indices]
        else:
            indices = range(len(src_seqs))

        return tuple([collate_seq(s) for s in [src_seqs, tgt_seqs]] + [indices])

    return collate


class ParallelDataset(Dataset):
    def __init__(self, src_fname, tgt_fname, tokenizer,
                 min_len, max_len, sort=False, max_size=None):

        self.min_len = min_len
        self.max_len = max_len

        self.src = self.process_data(src_fname, tokenizer, max_size)
        self.tgt = self.process_data(tgt_fname, tokenizer, max_size)
        assert len(self.src) == len(self.tgt)

        self.filter_data(min_len, max_len)
        assert len(self.src) == len(self.tgt)

        lengths = [len(s) + len(t) for (s, t) in zip(self.src, self.tgt)]
        self.lengths = torch.tensor(lengths)

        if sort:
            self.sort_by_length()

    def sort_by_length(self):
        self.lengths, indices = self.lengths.sort(descending=True)

        self.src = [self.src[idx] for idx in indices]
        self.tgt = [self.tgt[idx] for idx in indices]

    def filter_data(self, min_len, max_len):
        logging.info(f'filtering data, min len: {min_len}, max len: {max_len}')

        initial_len = len(self.src)

        filtered_src = []
        filtered_tgt = []
        for src, tgt in zip(self.src, self.tgt):
            if min_len <= len(src) <= max_len and \
                    min_len <= len(tgt) <= max_len:
                filtered_src.append(src)
                filtered_tgt.append(tgt)

        self.src = filtered_src
        self.tgt = filtered_tgt

        filtered_len = len(self.src)
        logging.info(f'pairs before: {initial_len}, after: {filtered_len}')

    def process_data(self, fname, tokenizer, max_size):
        logging.info(f'processing data from {fname}')
        data = []
        with open(fname) as dfile:
            for idx, line in enumerate(dfile):
                if max_size and idx == max_size:
                    break
                entry = tokenizer.segment(line)
                entry = torch.tensor(entry)
                data.append(entry)
        return data

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

    def get_loader(self, batch_size=1, shuffle=False, num_workers=0, batch_first=False,
                   drop_last=False, distributed=False, bucket=True):

        collate_fn = build_collate_fn(batch_first, sort=True)

        if shuffle:
            sampler = BucketingSampler(self, batch_size, bucket)
        else:
            sampler = SequentialSampler(self)

        return DataLoader(self,
                          batch_size=batch_size,
                          collate_fn=collate_fn,
                          sampler=sampler,
                          num_workers=num_workers,
                          pin_memory=False,
                          drop_last=drop_last)

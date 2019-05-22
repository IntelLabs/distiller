import logging
from collections import defaultdict

import seq2seq.data.config as config

def default():
    return config.UNK

class Tokenizer:
    def __init__(self, vocab_fname, separator='@@'):

        self.separator = separator

        logging.info(f'building vocabulary from {vocab_fname}')
        vocab = [config.PAD_TOKEN, config.UNK_TOKEN,
                 config.BOS_TOKEN, config.EOS_TOKEN]

        with open(vocab_fname) as vfile:
            for line in vfile:
                vocab.append(line.strip())

        logging.info(f'size of vocabulary: {len(vocab)}')
        self.vocab_size = len(vocab)


        self.tok2idx = defaultdict(default)
        for idx, token in enumerate(vocab):
            self.tok2idx[token] = idx

        self.idx2tok = {}
        for key, value in self.tok2idx.items():
            self.idx2tok[value] = key

    def segment(self, line):
        line = line.strip().split()
        entry = [self.tok2idx[i] for i in line]
        entry = [config.BOS] + entry + [config.EOS]
        return entry

    def detokenize(self, inputs, delim=' '):
        detok = delim.join([self.idx2tok[idx] for idx in inputs])
        detok = detok.replace(
            self.separator+ ' ', '').replace(self.separator, '')
        return detok

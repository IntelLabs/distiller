import torch

from seq2seq.data.config import BOS
from seq2seq.data.config import EOS
from seq2seq.inference.beam_search import SequenceGenerator
from seq2seq.utils import batch_padded_sequences


class Translator(object):

    def __init__(self, model, tok,
                 beam_size=5,
                 len_norm_factor=0.6,
                 len_norm_const=5.0,
                 cov_penalty_factor=0.1,
                 max_seq_len=50,
                 cuda=False):

        self.model = model
        self.tok = tok
        self.insert_target_start = [BOS]
        self.insert_src_start = [BOS]
        self.insert_src_end = [EOS]
        self.batch_first = model.batch_first
        self.cuda = cuda
        self.beam_size = beam_size

        self.generator = SequenceGenerator(
            model=self.model,
            beam_size=beam_size,
            max_seq_len=max_seq_len,
            cuda=cuda,
            len_norm_factor=len_norm_factor,
            len_norm_const=len_norm_const,
            cov_penalty_factor=cov_penalty_factor)

    def translate(self, input_sentences):
        stats = {}
        batch_size = len(input_sentences)
        beam_size = self.beam_size

        src_tok = [torch.tensor(self.tok.segment(line)) for line in input_sentences]

        bos = [self.insert_target_start] * (batch_size * beam_size)
        bos = torch.LongTensor(bos)
        if self.batch_first:
            bos = bos.view(-1, 1)
        else:
            bos = bos.view(1, -1)

        src = batch_padded_sequences(src_tok, self.batch_first, sort=True)
        src, src_length, indices = src

        src_length = torch.LongTensor(src_length)
        stats['total_enc_len'] = int(src_length.sum())

        if self.cuda:
            src = src.cuda()
            src_length = src_length.cuda()
            bos = bos.cuda()

        with torch.no_grad():
            context = self.model.encode(src, src_length)
            context = [context, src_length, None]

            if beam_size == 1:
                generator = self.generator.greedy_search
            else:
                generator = self.generator.beam_search

            preds, lengths, counter = generator(batch_size, bos, context)

        preds = preds.cpu()
        lengths = lengths.cpu()

        output = []
        for idx, pred in enumerate(preds):
            end = lengths[idx] - 1
            pred = pred[1: end]
            pred = pred.tolist()
            out = self.tok.detokenize(pred)
            output.append(out)

        stats['total_dec_len'] = int(lengths.sum())
        stats['iters'] = counter

        output = [output[indices.index(i)] for i in range(len(output))]
        return output, stats

import torch.nn as nn
from torch.nn.functional import log_softmax


class Seq2Seq(nn.Module):
    def __init__(self, encoder=None, decoder=None, batch_first=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_first = batch_first

    def encode(self, inputs, lengths):
        return self.encoder(inputs, lengths)

    def decode(self, inputs, context, inference=False):
        return self.decoder(inputs, context, inference)

    def generate(self, inputs, context, beam_size):
        logits, scores, new_context = self.decode(inputs, context, True)
        logprobs = log_softmax(logits, dim=-1)
        logprobs, words = logprobs.topk(beam_size, dim=-1)
        return words, logprobs, scores, new_context

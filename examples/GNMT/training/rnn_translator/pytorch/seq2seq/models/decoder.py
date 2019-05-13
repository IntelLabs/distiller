import itertools

import torch
import torch.nn as nn

from seq2seq.models.attention import BahdanauAttention
import seq2seq.data.config as config


class RecurrentAttention(nn.Module):

    def __init__(self, input_size, context_size, hidden_size, num_layers=1,
                 bias=True, batch_first=False, dropout=0):

        super(RecurrentAttention, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bias,
                           batch_first)

        self.attn = BahdanauAttention(hidden_size, context_size, context_size,
                                      normalize=True, batch_first=batch_first)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden, context, context_len):
        # set attention mask, sequences have different lengths, this mask
        # allows to include only valid elements of context in attention's
        # softmax
        self.attn.set_mask(context_len, context)

        rnn_outputs, hidden = self.rnn(inputs, hidden)
        attn_outputs, scores = self.attn(rnn_outputs, context)
        rnn_outputs = self.dropout(rnn_outputs)

        return rnn_outputs, hidden, attn_outputs, scores


class Classifier(nn.Module):

    def __init__(self, in_features, out_features, math='fp32'):
        super(Classifier, self).__init__()

        self.out_features = out_features

        # padding required to trigger HMMA kernels
        if math == 'fp16':
            out_features = (out_features + 7) // 8 * 8

        self.classifier = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.classifier(x)
        out = out[..., :self.out_features]
        return out


class ResidualRecurrentDecoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=128, num_layers=8, bias=True,
                 dropout=0, batch_first=False, math='fp32', embedder=None):

        super(ResidualRecurrentDecoder, self).__init__()

        self.num_layers = num_layers

        self.att_rnn = RecurrentAttention(hidden_size, hidden_size,
                                          hidden_size, num_layers=1,
                                          batch_first=batch_first)

        self.rnn_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.rnn_layers.append(
                nn.LSTM(2 * hidden_size, hidden_size, num_layers=1, bias=bias,
                        batch_first=batch_first))

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size,
                                        padding_idx=config.PAD)

        self.classifier = Classifier(hidden_size, vocab_size, math)
        self.dropout = nn.Dropout(p=dropout)

    def init_hidden(self, hidden):
        if hidden is not None:
            # per-layer chunks
            hidden = hidden.chunk(self.num_layers)
            # (h, c) chunks for LSTM layer
            hidden = tuple(i.chunk(2) for i in hidden)
        else:
            hidden = [None] * self.num_layers

        self.next_hidden = []
        return hidden

    def append_hidden(self, h):
        if self.inference:
            self.next_hidden.append(h)

    def package_hidden(self):
        if self.inference:
            hidden = torch.cat(tuple(itertools.chain(*self.next_hidden)))
        else:
            hidden = None
        return hidden

    def forward(self, inputs, context, inference=False):
        self.inference = inference

        enc_context, enc_len, hidden = context
        hidden = self.init_hidden(hidden)

        x = self.embedder(inputs)

        x, h, attn, scores = self.att_rnn(x, hidden[0], enc_context, enc_len)
        self.append_hidden(h)

        x = self.dropout(x)
        x = torch.cat((x, attn), dim=2)
        x, h = self.rnn_layers[0](x, hidden[1])
        self.append_hidden(h)

        for i in range(1, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)
            x = torch.cat((x, attn), dim=2)
            x, h = self.rnn_layers[i](x, hidden[i + 1])
            self.append_hidden(h)
            x = x + residual

        x = self.classifier(x)
        hidden = self.package_hidden()

        return x, scores, [enc_context, enc_len, hidden]

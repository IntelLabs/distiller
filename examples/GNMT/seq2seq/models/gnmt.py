import torch.nn as nn

import seq2seq.data.config as config
from .seq2seq_base import Seq2Seq
from .decoder import ResidualRecurrentDecoder
from .encoder import ResidualRecurrentEncoder


class GNMT(Seq2Seq):
    def __init__(self, vocab_size, hidden_size=512, num_layers=8, bias=True,
                 dropout=0.2, batch_first=False, math='fp32',
                 share_embedding=False):

        super(GNMT, self).__init__(batch_first=batch_first)

        if share_embedding:
            embedder = nn.Embedding(vocab_size, hidden_size, padding_idx=config.PAD)
        else:
            embedder = None

        self.encoder = ResidualRecurrentEncoder(vocab_size, hidden_size,
                                                num_layers, bias, dropout,
                                                batch_first, embedder)

        self.decoder = ResidualRecurrentDecoder(vocab_size, hidden_size,
                                                num_layers, bias, dropout,
                                                batch_first, math, embedder)



    def forward(self, input_encoder, input_enc_len, input_decoder):
        context = self.encode(input_encoder, input_enc_len)
        context = (context, input_enc_len, None)
        output, _, _ = self.decode(input_decoder, context)

        return output

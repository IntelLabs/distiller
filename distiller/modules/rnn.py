#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
from .eltwise import EltwiseAdd, EltwiseMult
from itertools import product


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Treat f,i,o,c_ex as one single object:
        self.fc_gate_x = nn.Linear(input_size, hidden_size * 4, bias=bias)
        self.fc_gate_h = nn.Linear(hidden_size, hidden_size * 4, bias=bias)
        self.eltwiseadd_gate = EltwiseAdd()
        # Apply activations separately:
        self.act_f = nn.Sigmoid()
        self.act_i = nn.Sigmoid()
        self.act_o = nn.Sigmoid()
        self.act_c_ex = nn.Tanh()
        # Calculate cell:
        self.eltwisemult_cell_forget = EltwiseMult()
        self.eltwisemult_cell_input = EltwiseMult()
        self.eltwiseadd_cell = EltwiseAdd()
        # Calculate hidden:
        self.act_h = nn.Tanh()
        self.eltwisemult_hidden = EltwiseMult()
        self.init_weights()

    def forward(self, x, h):
        h_prev, c_prev = h
        fc_gate = self.eltwiseadd_gate(self.fc_gate_x(x), self.fc_gate_h(h_prev))
        i, f, c_ex, o = torch.chunk(fc_gate, 4, dim=1)
        i, f, c_ex, o = self.act_i(i), self.act_f(f), self.act_c_ex(c_ex), self.act_o(o)
        cf, ci = self.eltwisemult_cell_forget(f, c_prev), self.eltwisemult_cell_input(i, c_ex)
        c = self.eltwiseadd_cell(cf, ci)
        h = self.eltwisemult_hidden(o, self.act_h(c))
        return h, c

    def sequence_forward(self, x, h):
        results = []
        for step in x:
            y, h = self.forward(step, h)
            results.append(y)
            h = (y, h)  # h, c
        return torch.stack(results), h

    def init_hidden(self, batch_size, device='cuda:0'):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        return h_0, c_0

    def init_weights(self):
        initrange = 0.1
        self.fc_gate_x.weight.data.uniform_(-initrange, initrange)
        self.fc_gate_h.weight.data.uniform_(-initrange, initrange)

    def to_pytorch_impl(self):
        module = nn.LSTMCell(self.input_size, self.hidden_size, self.bias)
        module.weight_hh, module.weight_ih = \
            nn.Parameter(self.fc_gate_h.weight.clone().detach()), \
            nn.Parameter(self.fc_gate_x.weight.clone().detach())
        if self.bias:
            module.bias_hh, module.bias_ih = \
                nn.Parameter(self.fc_gate_h.bias.clone().detach()), \
                nn.Parameter(self.fc_gate_x.bias.clone().detach())
        return module

    @staticmethod
    def from_pytorch_impl(lstmcell: nn.LSTMCell):
        module = LSTMCell(input_size=lstmcell.input_size, hidden_size=lstmcell.hidden_size, bias=lstmcell.bias)
        module.fc_gate_x.weight = nn.Parameter(lstmcell.weight_ih.clone().detach())
        module.fc_gate_h.weight = nn.Parameter(lstmcell.weight_hh.clone().detach())
        if lstmcell.bias:
            module.fc_gate_x.bias = nn.Parameter(lstmcell.bias_ih.clone().detach())
            module.fc_gate_h.bias = nn.Parameter(lstmcell.bias_hh.clone().detach())

        return module

    def __repr__(self):
        return "%s(%d, %d)" % (self.__class__.__name__, self.input_size, self.hidden_size)


def _reformat_hidden(h):
    h_all = [t[0] for t in h]
    c_all = [t[1] for t in h]
    return torch.stack(h_all, 0), torch.stack(c_all, 0)


def _reformat_hidden_bidirectional(h_result):
    h_all = [t[0] for t in h_result]
    c_all = [t[1] for t in h_result]
    return torch.cat(h_all, dim=0), torch.cat(c_all, dim=0)


def _repackage_bidirectional_input_h(h):
    h_t, c_t = h
    h_front, h_back = h_t[::2], h_t[1::2]
    c_front, c_back = c_t[::2], c_t[1::2]
    h_front = (h_front, c_front)
    h_back = (h_back, c_back)
    return h_front, h_back


class LSTM(nn.Module):
    """
    A modular implementation of an LSTM module.
    Args:
        input_size (int): size of the input
        hidden_size (int): size of the hidden connections and output.
        num_layers (int): number of LSTMCells
        bias (bool): use bias
        dropout : dropout factor
        bidirectional (bool): Whether or not the LSTM is bidirectional. default: False (unidirectional).
        bidirectional_type (int): 1 or 2, corresponds to type 1 and type 2 as per
            https://github.com/pytorch/pytorch/issues/4930. default: 2
    """
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout=0.5, bidirectional=False,
                 bidirectional_type=2):
        super(LSTM, self).__init__()
        if num_layers < 1:
            raise ValueError("Number of layers has to be at least 1.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bias = bias
        self.bidirectional_type = bidirectional_type

        if bidirectional:
            # Following https://github.com/pytorch/pytorch/issues/4930 -
            if bidirectional_type == 1:
                self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size, bias)] +
                                           [LSTMCell(hidden_size, hidden_size, bias)
                                            for _ in range(1, num_layers)])

                self.cells_reverse = nn.ModuleList([LSTMCell(input_size, hidden_size, bias)] +
                                                   [LSTMCell(hidden_size, hidden_size, bias)
                                                    for _ in range(1, num_layers)])
                self.single_forward = self.__single_bidirectional_type1_forward

            elif bidirectional_type == 2:
                self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size, bias)] +
                                           [LSTMCell(2 * hidden_size, hidden_size, bias)
                                            for _ in range(1, num_layers)])

                self.cells_reverse = nn.ModuleList([LSTMCell(input_size, hidden_size, bias)] +
                                                   [LSTMCell(2 * hidden_size, hidden_size, bias)
                                                    for _ in range(1, num_layers)])
                # Overwrite the current forward.
                self.forward = self.__bidirectional_type2_forward

            else:
                raise ValueError("The only allowed types are [1, 2].")
        else:
            self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size, bias)] +
                                       [LSTMCell(hidden_size, hidden_size, bias)
                                        for _ in range(1, num_layers)])
            self.single_forward = self.__single_unidirectional_forward

        self.dropout = nn.Dropout(dropout)
        self.dropout_factor = dropout

    def forward(self, x, h):
        results = []
        for step in x:
            y, h = self.single_forward(step, h)
            results.append(y)
        return torch.stack(results), h

    def __bidirectional_type2_forward(self, x, h):
        """
        Type 2 bidirectional forward is `sequence first, layers after` approach.
        """
        out = x
        h_h_result = []
        h_c_result = []
        (h_front_all, c_front_all), (h_back_all, c_back_all) = _repackage_bidirectional_input_h(h)
        for i, (cell_front, cell_back) in enumerate(zip(self.cells, self.cells_reverse)):
            h_front, h_back = (h_front_all[i], c_front_all[i]), (h_back_all[i], c_back_all[i])

            # Sequence treatment:
            out_front, h_front = cell_front.sequence_forward(out, h_front)
            out_back, h_back = cell_back.sequence_forward(out.flip([0]), h_back)
            out = torch.cat([out_front, out_back.flip([0])], dim=-1)

            h_h_result += [h_front[0], h_back[0]]
            h_c_result += [h_front[1], h_back[1]]
            if i < self.num_layers-1:
                out = self.dropout(out)
        h = torch.stack(h_h_result, dim=0), torch.stack(h_c_result, dim=0)
        return out, h

    def __single_bidirectional_type1_forward(self, x, h):
        """
        Type 1 bidirectional forward is layer is `layers first, sequence after` approach, similar to regular
        unidirectional LSTM.
        """
        (h_front_all, c_front_all), (h_back_all, c_back_all) = _repackage_bidirectional_input_h(h)
        h_result = []
        out_front, out_back = x, x.flip([0])
        for i, (cell_front, cell_back) in enumerate(zip(self.cells, self.cells_reverse)):
            h_front, h_back = (h_front_all[i], c_front_all[i]), (h_back_all[i], c_back_all[i])
            h_front, c_front = cell_front(out_front, h_front)
            h_back, c_back = cell_back(out_back, h_back)
            out_front, out_back = h_front, h_back
            if i < self.num_layers-1:
                out_front, out_back = self.dropout(out_front), self.dropout(out_back)
            h_current = torch.stack([h_front, h_back]), torch.stack([c_front, c_back])
            h_result.append(h_current)
        h_result = _reformat_hidden_bidirectional(h_result)
        return torch.cat([out_front, out_back], dim=-1), h_result

    def __single_unidirectional_forward(self, x, h):
        h_all, c_all = h
        h_result = []
        out = x
        for i, cell in enumerate(self.cells):
            h = h_all[i], c_all[i]
            out, hid = cell(out, h)
            if i < self.num_layers-1:
                out = self.dropout(out)
            h_result.append((out, hid))
        h_result = _reformat_hidden(h_result)
        return out, h_result

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        n_dir = 2 if self.bidirectional else 1
        return (weight.new_zeros(self.num_layers * n_dir, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers * n_dir, batch_size, self.hidden_size))

    def init_weights(self):
        for cell in self.hidden_cells:
            cell.init_weights()

    def flatten_parameters(self):
        pass

    def to_pytorch_impl(self):
        module = nn.LSTM(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         dropout=self.dropout_factor,
                         bidirectional=self.bidirectional)
        param_gates = ['i', 'h']

        param_types = ['weight']
        if self.bias:
            param_types.append('bias')

        suffixes = ['']
        if self.bidirectional:
            suffixes.append('_reverse')

        for i in range(self.num_layers):
            for ptype, pgate, psuffix in product(param_types, param_gates, suffixes):
                cell = self.cells[i] if psuffix == '' else self.cells_reverse[i]
                lstm_pth_param_name = "%s_%sh_l%d%s" % (ptype, pgate, i, psuffix)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param_tensor = getattr(gate, ptype).clone().detach()

                # same as `module.weight_ih_l0 = nn.Parameter(param_tensor)`:
                setattr(module, lstm_pth_param_name, nn.Parameter(param_tensor))

        module.flatten_parameters()
        return module

    @staticmethod
    def from_pytorch_impl(lstm: nn.LSTM):
        bidirectional = lstm.bidirectional

        module = LSTM(lstm.input_size, lstm.hidden_size, lstm.num_layers, lstm.bias,
                      lstm.dropout, bidirectional=bidirectional)
        param_gates = ['i', 'h']

        param_types = ['weight']
        if lstm.bias:
            param_types.append('bias')

        suffixes = ['']
        if bidirectional:
            suffixes.append('_reverse')

        for i in range(lstm.num_layers):
            for ptype, pgate, psuffix in product(param_types, param_gates, suffixes):
                cell = module.cells[i] if psuffix == '' else module.cells_reverse[i]
                lstm_pth_param_name = "%s_%sh_l%d%s" % (ptype, pgate, i, psuffix)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param_tensor = getattr(lstm, lstm_pth_param_name).clone().detach()  # e.g. `lstm.weight_ih_l0.detach()`
                setattr(gate, ptype, nn.Parameter(param_tensor))

        return module

    def __repr__(self):
        return "%s(%d, %d, num_layers=%d, dropout=%.2f, bidirectional=%s)" % \
               (self.__class__.__name__,
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.dropout_factor,
                self.bidirectional)

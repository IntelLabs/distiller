import torch
import torch.nn as nn
from .eltwise import EltwiseAdd, EltwiseMult
from itertools import product


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Treat f,i,o,c_ex as one single object:
        self.fc_gate_x = nn.Linear(input_size, hidden_size * 4)
        self.fc_gate_h = nn.Linear(hidden_size, hidden_size * 4)
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

    def init_hidden(self, batch_size, device='cuda:0'):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        return h_0, c_0

    def init_weights(self):
        initrange = 0.1
        self.fc_gate_x.weight.data.uniform_(-initrange, initrange)
        self.fc_gate_h.weight.data.uniform_(-initrange, initrange)

    def to_pytorch_impl(self):
        module = nn.LSTMCell(self.input_size, self.hidden_size)
        module.weight_hh, module.weight_ih, module.bias_hh, module.bias_ih = \
            nn.Parameter(self.fc_gate_h.weight.clone().detach()), \
            nn.Parameter(self.fc_gate_x.weight.clone().detach()), \
            nn.Parameter(self.fc_gate_h.bias.clone().detach()), \
            nn.Parameter(self.fc_gate_x.bias.clone().detach())
        return module

    @staticmethod
    def from_pytorch_impl(lstmcell: nn.LSTMCell):
        module = LSTMCell(input_size=lstmcell.input_size, hidden_size=lstmcell.hidden_size)
        module.fc_gate_x.weight = nn.Parameter(lstmcell.weight_ih.clone().detach())
        module.fc_gate_x.bias = nn.Parameter(lstmcell.bias_ih.clone().detach())
        module.fc_gate_h.weight = nn.Parameter(lstmcell.weight_hh.clone().detach())
        module.fc_gate_h.bias = nn.Parameter(lstmcell.bias_hh.clone().detach())

        return module

    def __repr__(self):
        return "%s(%d, %d)" % (self.__class__.__name__, self.input_size, self.hidden_size)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout=0.5):
        super(LSTM, self).__init__()
        if n_layers < 1:
            raise ValueError("Number of layers has to be at least 1.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size)] +
                                   [LSTMCell(hidden_size, hidden_size)
                                    for _ in range(1, n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.dropout_factor = dropout

    def forward(self, x, h):
        results = []
        for step in x:
            y, h = self.single_forward(step, h)
            results.append(y)
        return torch.stack(results), h

    def single_forward(self, x, h):
        h_all, c_all = h
        h_result = []
        out = x
        for i, cell in enumerate(self.cells):
            h = h_all[i], c_all[i]
            out, hid = cell(out, h)
            if i < self.n_layers-1:
                out = self.dropout(out)
            h_result.append((out, hid))
        h_result = self._reformat_hidden(h_result)
        return out, h_result

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.n_layers, batch_size, self.hidden_size),
                weight.new_zeros(self.n_layers, batch_size, self.hidden_size))

    def init_weights(self):
        for cell in self.hidden_cells:
            cell.init_weights()

    def _reformat_hidden(self, h):
        h_all = [t[0] for t in h]
        c_all = [t[1] for t in h]
        return torch.stack(h_all, 0), torch.stack(c_all, 0)

    def flatten_parameters(self):
        pass

    def to_pytorch_impl(self):
        module = nn.LSTM(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.n_layers,
                         dropout=self.dropout_factor)
        for i, cell in enumerate(self.cells):
            for ptype, pgate in product(['weight', 'bias'], ['i', 'h']):
                lstm_pth_param_name = "%s_%sh_l%d" % (ptype, pgate, i)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param = nn.Parameter(getattr(gate, ptype).detach())  # e.g. `cell.fc_gate_x.weight.detach()`

                # same as `module.weight_ih_l0 = nn.Parameter(cell.fc_gate_x.weight.detach())`:
                setattr(module, lstm_pth_param_name, param)

        module.flatten_parameters()
        return module

    @staticmethod
    def from_pytorch_impl(lstm: nn.LSTM):
        module = LSTM(lstm.input_size, lstm.hidden_size, lstm.num_layers, lstm.dropout)

        for i in range(lstm.num_layers):
            for ptype, pgate in product(['weight', 'bias'], ['i', 'h']):
                cell = module.cells[i]
                lstm_pth_param_name = "%s_%sh_l%d" % (ptype, pgate, i)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param = nn.Parameter(
                    getattr(lstm, lstm_pth_param_name).clone().detach())  # e.g. `lstm.weight_ih_l0.detach()`
                setattr(gate, ptype, param)

        return module

    def __repr__(self):
        return "%s(%d, %d, num_layers=%d, dropout=%8.2f)" % \
               (self.__class__.__name__, self.input_size, self.hidden_size, self.n_layers, self.dropout_factor)

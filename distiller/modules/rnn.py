import torch
import torch.nn as nn
from .eltwise import EltwiseAdd, EltwiseMult


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.5):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Treat f,i,o,c_ex as one single object:
        self.fc_gate_x = nn.Linear(input_size, hidden_size * 4)
        self.fc_gate_h = nn.Linear(hidden_size, hidden_size * 4)
        self.eltwiseadd_gate = EltwiseAdd()
        # Split the object into f,i,o,c_ex:
        self.chunk_gate = Chunk(4, dim=1)
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
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.init_weights()

    def forward(self, x, h):
        h_prev, c_prev = h
        fc_gate = self.eltwiseadd_gate(self.fc_gate_x(x), self.fc_gate_h(h_prev))
        f, i, o, c_ex = self.chunk_gate(fc_gate)
        f, i, o, c_ex = self.act_f(f), self.act_i(i), self.act_o(o), self.act_c_ex(c_ex)
        cf, ci = self.eltwisemult_cell_forget(f, c_prev), self.eltwisemult_cell_input(i, c_ex)
        c = self.eltwiseadd_cell(cf, ci)
        h = self.eltwisemult_hidden(o, self.act_h(c))
        if self.dropout:
            h = self.dropout(h)
        return h, (h, c)

    def init_hidden(self, batch_size, device='cuda:0'):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        return h_0, c_0

    def init_weights(self):
        initrange = 0.1
        self.fc_gate_x.weight.data.uniform_(-initrange, initrange)
        self.fc_gate_h.weight.data.uniform_(-initrange, initrange)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout=0.5):
        super(LSTM, self).__init__()
        if n_layers < 1:
            raise ValueError("Number of layers has to be at least 1.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size, dropout=dropout)] +
                                   [LSTMCell(hidden_size, hidden_size, dropout=dropout)
                                    for _ in range(1, n_layers)])
        # Last cell has no dropout:
        self.cells[-1].dropout = None

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
            h_result.append(hid)
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

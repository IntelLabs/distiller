import pytest

import distiller
from distiller.modules import DistillerLSTM, DistillerLSTMCell
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence
from torch.testing import assert_allclose

ATOL = 5e-5
RTOL = 1e-3
BATCH_SIZE = 32
SEQUENCE_SIZE = 35


def test_basic():
    lstmcell = DistillerLSTMCell(3, 5)
    assert lstmcell.fc_gate_x.weight.shape == (5 * 4, 3)
    assert lstmcell.fc_gate_h.weight.shape == (5 * 4, 5)
    assert lstmcell.fc_gate_x.bias.shape == (5 * 4,)
    assert lstmcell.fc_gate_h.bias.shape == (5 * 4,)

    lstm = DistillerLSTM(3, 5, 4, False, False, 0.0, True)
    assert lstm.bidirectional_type == 2
    assert lstm.cells[0].fc_gate_x.weight.shape == (5 * 4, 3)
    assert lstm.cells[1].fc_gate_x.weight.shape == (5 * 4, 5 * 2)


def test_conversion():
    lc_man = DistillerLSTMCell(3, 5)
    lc_pth = lc_man.to_pytorch_impl()
    lc_man1 = DistillerLSTMCell.from_pytorch_impl(lc_pth)

    assert (lc_man.fc_gate_x.weight == lc_man1.fc_gate_x.weight).all()
    assert (lc_man.fc_gate_h.weight == lc_man1.fc_gate_h.weight).all()

    l_man = DistillerLSTM(3, 5, 2)
    l_pth = l_man.to_pytorch_impl()
    l_man1 = DistillerLSTM.from_pytorch_impl(l_pth)

    for i in range(l_man.num_layers):
        assert (l_man1.cells[i].fc_gate_x.weight == l_man.cells[i].fc_gate_x.weight).all()
        assert (l_man1.cells[i].fc_gate_h.weight == l_man.cells[i].fc_gate_h.weight).all()
        assert (l_man1.cells[i].fc_gate_x.bias == l_man.cells[i].fc_gate_x.bias).all()
        assert (l_man1.cells[i].fc_gate_h.bias == l_man.cells[i].fc_gate_h.bias).all()


def assert_output(out_true, out_pred):
    y_t, h_t = out_true
    y_p, h_p = out_pred
    if isinstance(y_t, PackedSequence):
        y_t, lenghts_t = pad_packed_sequence(y_t)
        y_p, lenghts_p = pad_packed_sequence(y_p)
        assert all(lenghts_t == lenghts_p)
    assert_allclose(y_p, y_t, RTOL, ATOL)
    h_h_t, h_c_t = h_t
    h_h_p, h_c_p = h_p
    assert_allclose(h_h_p, h_h_t, RTOL, ATOL)
    assert_allclose(h_c_p, h_c_t, RTOL, ATOL)


@pytest.fixture(name='bidirectional', params=[False, True], ids=['bidirectional_off', 'bidirectional_on'])
def fixture_bidirectional(request):
    return request.param


@pytest.mark.parametrize(
    "input_size, hidden_size, num_layers",
    [
        (1, 1, 2),
        (3, 5, 7),
        (1500, 1500, 5)
    ]
)
def test_forward_lstm(input_size, hidden_size, num_layers, bidirectional):
    # Test conversion from pytorch implementation
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
    lstm_man = DistillerLSTM.from_pytorch_impl(lstm)
    lstm.eval()
    lstm_man.eval()

    h = lstm_man.init_hidden(BATCH_SIZE)
    x = torch.rand(SEQUENCE_SIZE, BATCH_SIZE, input_size)

    out_true = lstm(x, h)
    out_pred = lstm_man(x, h)
    assert_output(out_true, out_pred)
    # Test conversion to pytorch implementation
    lstm_man = DistillerLSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
    lstm = lstm_man.to_pytorch_impl()

    lstm.eval()
    lstm_man.eval()

    h = lstm_man.init_hidden(BATCH_SIZE)
    x = torch.rand(SEQUENCE_SIZE, BATCH_SIZE, input_size)

    out_true = lstm(x, h)
    out_pred = lstm_man(x, h)
    assert_output(out_true, out_pred)


@pytest.mark.parametrize(
    "input_size, hidden_size, num_layers, input_lengths",
    [
        (1, 1, 2, [5, 4, 3]),
        (3, 5, 7, [20, 15, 5]),
        (500, 500, 5, [50, 35, 25])
    ]
)
def test_packed_sequence(input_size, hidden_size, num_layers, input_lengths, bidirectional):
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)
    lstm_man = DistillerLSTM.from_pytorch_impl(lstm)
    lstm.eval()
    lstm_man.eval()

    h = lstm_man.init_hidden(BATCH_SIZE)
    x = pack_sequence([torch.rand(length, input_size) for length in input_lengths])
    out_true = lstm(x)
    out_pred = lstm_man(x)
    assert_output(out_true, out_pred)

import distiller
from torch.testing import assert_allclose
import torch.nn as nn
from distiller.quantization.layer_fusions import FusedLinearBatchNorm
import pytest

ATOL = 5e-5
RTOL = 1e-3
BATCH_SIZE = 32


def test_folding_fc():
    distiller.set_deterministic(seed=1234)
    linear = nn.Linear(512, 1024)
    bn = nn.BatchNorm1d(1024)
    unfolded = nn.Sequential(linear.copy(), bn.copy())
    folded = FusedLinearBatchNorm(linear, bn, quantized=False)




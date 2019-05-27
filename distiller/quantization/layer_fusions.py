import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedQuantBatchNorm(nn.Module):
    """
    Wrapper for simulated fusing of BatchNorm into linear layers.
    Args:
        linear_module: the wrapped linear layer
        bn : batch normalization
    """
    def __init__(self, linear_module, bn, freeze_bn_delay):
        super(FusedQuantBatchNorm, self).__init__()
        if not isinstance(linear_module, (nn.Linear, nn.Conv2d)) \
                and not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            raise ValueError("Only supporting fusing nn.BatchNorm1d/nn.BatchNorm2d into nn.Linear/nn.Conv2d.")

        self.linear_module = linear_module
        self.bn = bn
        self.affine = bn.affine
        self.track_running_stats = bn.track_running_stats
        self.freeze_bn_delay = freeze_bn_delay
        self.freeze_bn = False  # Freezes bn stats

    def forward(self, x):
        pass



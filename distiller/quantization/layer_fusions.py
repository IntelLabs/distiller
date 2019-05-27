import torch
import torch.nn as nn


class FusedQuantBatchNorm(nn.Module):
    def __init__(self, linear_module, bn):
        super(FusedQuantBatchNorm, self).__init__()
        if not isinstance(linear_module, (nn.Linear, nn.Conv2d)) \
                and not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            raise ValueError("Only supporting fusing nn.BatchNorm1d/nn.BatchNorm2d into nn.Linear/nn.Conv2d.")

        self.linear_module = linear_module
        self.bn = bn


    def forward(self, x):
        if self.training:



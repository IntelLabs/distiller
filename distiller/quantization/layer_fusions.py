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
    def __init__(self, linear_module, bn, freeze_bn_delay, frozen=False):
        super(FusedQuantBatchNorm, self).__init__()
        if not isinstance(linear_module, (nn.Linear, nn.Conv2d)) \
                and not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            raise ValueError("Only supporting fusing nn.BatchNorm1d/nn.BatchNorm2d into nn.Linear/nn.Conv2d.")
        if not bn.track_running_stats or not bn.affine:
            raise ValueError("Quantization is only supported for BatchNorm which tracks runnins stats with"
                             "affine weights.")

        self.linear_module = linear_module
        self.bn = bn
        self.freeze_bn_delay = freeze_bn_delay
        self.frozen = frozen  # Indicate whether the BatchNorm stats are frozen

    def forward(self, x):
        pass


    def freeze(self):
        """
        Start using the long term BatchNorm stats.
        We fuse these stats into the linear layer and only use it.
        Now, the corrected weights for the linear layer are:
            sigma = sqrt(bn.variance + bn.eps)
            weights_corrected = bn.weights * linear.weights / sigma
            bias_corrected = bn.weights * (linear.bias - bn.mean) / sigma + bn.bias =
                           = linear.bias * bn.weights / sigma + (bn.bias - bn.weights * bn.mean / sigma)
        """
        inverse_sigma = torch.rsqrt(self.bn.running_var + self.bn.eps)
        self.linear_module.weight.mul_(self.bn.weight * inverse_sigma)
        self.linear_module.bias.mul_(self.bn.weight * inverse_sigma)
        self.linear_module.bias.add_(self.bn.bias - self.bn.weight * self.bn.mean * inverse_sigma)
        self.frozen = True


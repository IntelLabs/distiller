from .range_linear import *
from torch.nn import functional as F

__all__ = ['FusedLinearBatchNorm']

# Number of steps before freezing the batch norm running average and variance
FREEZE_BN_DELAY_DEFAULT = 200000


class FusedLinearBatchNorm(nn.Module):
    def __init__(self, linear_module, bn, quantized=False, num_bits_acts=8, num_bits_accum=32, num_bits_params=8,
                 mode=LinearQuantMode.SYMMETRIC, dequantize=True,
                 freeze_bn_delay=FREEZE_BN_DELAY_DEFAULT, frozen=False):
        """
        Wrapper for simulated fusing of BatchNorm into linear layers.

        Args:
            linear_module (nn.Linear or nn.Conv2d): the wrapped linear layer
            bn (nn.BatchNorm1d or nn.BatchNorm2d): batch normalization
            quantized (bool): whether to quantize the modules.
            num_bits_acts (int): number of bits to quantize activations
            num_bits_accum (int): number of bits to quantize accumulator (or bias)
            num_bits_params (int): number of bits to quantize weights
            mode (str or LinearQuantMode): the mode of quantization, see `distiller.quantization.LinearQuantMode`
            dequantize (bool): whether to dequantized the acts after calculating them
            freeze_bn_delay (int): number of steps before freezing the batchnorm running stats
            frozen (bool): whether to freeze the batchnorm right away
        Note:
            The quantized version was implemented according to https://arxiv.org/pdf/1806.08342.pdf Section 3.2.2.
        """

        super(FusedLinearBatchNorm, self).__init__()
        FusedLinearBatchNorm.verify_module_types(linear_module, bn)

        if not bn.track_running_stats or not bn.affine:
            raise ValueError("Quantization is only supported for BatchNorm which tracks running stats with"
                             "affine weights.")
        self.dequantize = dequantize
        self.mode = verify_quant_mode(mode)
        self.num_bits_acts = num_bits_acts
        self.num_bits_params = num_bits_params
        self.num_bits_accum = num_bits_accum
        self.linear = linear_module
        self.bn = bn
        self.freeze_bn_delay = freeze_bn_delay
        self.frozen = frozen  # Indicate whether the BatchNorm stats are frozen
        self._has_bias = (self.linear.bias is not None)
        self.quantized = quantized
        if isinstance(linear_module, nn.Linear):
            self.linear_forward_fn = self._linear_layer_forward
        else:
            self.linear_forward_fn = self._conv2d_layer_forward

    @staticmethod
    def verify_module_types(linear, bn):
        if not isinstance(linear, (nn.Linear, nn.Conv2d)) \
                and not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            raise TypeError("Only supporting fusing nn.BatchNorm1d/nn.BatchNorm2d into nn.Linear/nn.Conv2d.")
        if isinstance(linear, nn.Linear) and isinstance(bn, nn.BatchNorm2d):
            raise TypeError("nn.Linear layer has to be followed by a nn.BatchNorm1d layer.")
        if isinstance(linear, nn.Conv2d) and isinstance(bn, nn.BatchNorm1d):
            raise TypeError("nn.Con2d layer has to be followed by a nn.BatchNorm2d layer.")

    def forward(self, x):
        """ According to https://arxiv.org/pdf/1806.08342.pdf section 3.2.2."""
        batch_mean = batch_var = None
        if not self.frozen:
            # First update the running mean and variance of the batch norm.
            batch_mean, batch_var = self.batch_stats(self.linear(x))

        bn_inverse_sigma = self.bn_inverse_sigma
        w_corrected = self.linear.weight * self.bn.weight * bn_inverse_sigma
        w_quantized = self.quant_weights(w_corrected)
        if self._has_bias:
            linear_bias, bn_bias = self.quant_bias(self.linear.bias), self.quant_bias(self.bn.bias)
            bias = self.bn.weight * (linear_bias - self.bn.running_mean) * bn_inverse_sigma + bn_bias
        else:
            bias = None
        y = self.linear_forward_fn(x, w_quantized, bias)
        if not self.frozen:
            batch_sigma = torch.sqrt(batch_var + self.bn.eps)
            c = batch_sigma * bn_inverse_sigma
            bias_correction = self.bn.weight * (self.bn.running_mean * bn_inverse_sigma - batch_mean / batch_sigma)
            return y / c + bias_correction
        return y

    def quant_weights(self, w: nn.Parameter):
        if not self.quantized:
            return w
        return self._quant_param(w, self.num_bits_params)

    def quant_bias(self, b: nn.Parameter):
        if not self.quantized:
            return b
        return self._quant_param(b, self.num_bits_accum)

    def _quant_param(self, t: torch.Tensor, num_bits):
        """
        Quantize a parameter locally.
        """
        t_min, t_max = get_tensor_min_max(t)
        t_min, t_max = t_min.item(), t_max.item()
        if self.mode == LinearQuantMode.SYMMETRIC:
            t_max = max(abs(t_min), abs(t_max))
            scale, zero_point = symmetric_linear_quantization_params(num_bits, t_max)
        else:
            signed = self.mode == LinearQuantMode.ASYMMETRIC_SIGNED
            scale, zero_point = asymmetric_linear_quantization_params(num_bits, t_min, t_max,
                                                                      signed=signed)
        return LinearQuantizeSTE.apply(t, scale, zero_point, self.dequantize, False)

    @property
    def bn_inverse_sigma(self):
        """
        Returns 1/sqrt(var + epsilon)
        """
        # .clone().detach() to remove history from these tensors.
        return torch.rsqrt(self.bn.running_var + self.bn.eps).clone().detach()

    def batch_stats(self, x):
        """
        Get the batch mean and variance of x and updates the BatchNorm's running mean and average.
        Args:
            x (torch.Tensor): input batch.
        Returns:
            (mean,variance)
        Note:
            In case of `nn.Linear`, x may be of shape (N, C, L) or (N, L)
            where N is batch size, C is number of channels, L is the features size.
            The batch norm computes the stats over C in the first case or L on the second case.
            The batch normalization layer is
            (`nn.BatchNorm1d`)[https://pytorch.org/docs/stable/nn.html#batchnorm1d]

            In case of `nn.Conv2d`, x is of shape (N, C, H, W)
            where H,W are the image dimensions, and the batch norm computes the stats over C.
            The batch normalization layer is
            (`nn.BatchNorm2d`)[https://pytorch.org/docs/stable/nn.html#batchnorm2d]
        """
        channel_size = self.bn.num_features
        mean = x.transpose(0, 1).contiguous().view(channel_size, -1).mean(1)
        var = x.transpose(0, 1).contiguous().view(channel_size, -1).var(1)
        if self.bn.momentum:
            self.bn.running_mean.mul_(1-self.bn.momentum).add_(self.bn.momentum * mean)
            self.bn.running_var.mul_(1-self.bn.momentum).add_(self.bn.momentum * var)
        else:
            # momentum is None - we compute a cumulative moving average
            # as noted in https://pytorch.org/docs/stable/nn.html#batchnorm2d
            num_batches = self.bn.num_batches_tracked
            if num_batches == 0:
                self.bn.running_mean.data = mean
                self.bn.running_var.data = var
            else:
                momentum = 1/num_batches
                self.bn.running_mean.mul_(1 - momentum).add_(momentum * mean)
                self.bn.running_var.mul_(1 - momentum).add_(momentum * var)
        return mean, var

    def freeze_batchnorm(self):
        self.frozen = True

    def _linear_layer_forward(self, input, w, b):
        return F.linear(input, w, b)

    def _conv2d_layer_forward(self, input, w, b):
        # We replace the weights temporarily to allow for easy forward with the new weights.
        w_old, b_old = self.linear.weight, self.linear.bias
        self.linear.weight, self.linear.bias = w, b
        y = self.linear(input)
        self.linear.weight, self.linear.bias = w_old, b_old
        return y


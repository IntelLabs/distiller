from .range_linear import *

__all__ = ['FusedLinearBatchNorm']

# Number of steps before freezing the batch norm running average and variance
FREEZE_BN_DELAY_DEFAULT = 200000


class FusedLinearBatchNorm(nn.Module):
    """
    Wrapper for simulated fusing of BatchNorm into linear layers.
    Args:
        linear_module: the wrapped linear layer
        bn : batch normalization
    """
    def __init__(self, linear_module, bn, num_bits_acts, num_bits_accum, num_bits_params,
                 mode=LinearQuantMode.SYMMETRIC, ema_decay=0.999, dequantize=True,
                 freeze_bn_delay=FREEZE_BN_DELAY_DEFAULT, frozen=False):

        super(FusedLinearBatchNorm, self).__init__()
        if not isinstance(linear_module, (nn.Linear, nn.Conv2d)) \
                and not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            raise ValueError("Only supporting fusing nn.BatchNorm1d/nn.BatchNorm2d into nn.Linear/nn.Conv2d.")

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
        self.quantizer_inputs = FakeLinearQuantization(num_bits_acts, mode=self.mode, ema_decay=ema_decay)

    def forward(self, x):
        """ According to https://arxiv.org/pdf/1806.08342.pdf section 3.2.2."""
        x_quantized = self.quantizer_inputs(x)
        bn_inverse_sigma = self.bn_inverse_sigma
        w_corrected = self.linear.weight * self.bn.weight * bn_inverse_sigma
        w_quantized = self.quant_weights(w_corrected)
        if self._has_bias:
            linear_bias, bn_bias = self.quant_bias(self.linear.bias), self.quant_bias(self.bn.bias)
            bias = self.bn.weight * (linear_bias - self.bn.running_mean) * bn_inverse_sigma + bn_bias
        else:
            bias = None
        y = self.linear_forward(x_quantized, w_quantized, bias)
        if not self.frozen:
            batch_mean, batch_var = self.batch_stats(self.linear(x))
            batch_sigma = torch.sqrt(batch_var + self.bn.eps)
            c = batch_sigma * bn_inverse_sigma
            bias_correction = self.bn.weight * (self.bn.running_mean * bn_inverse_sigma - batch_mean / batch_sigma)
            return y / c + bias_correction
        return y

    def quant_weights(self, w: nn.Parameter):
        return self._quant_param(w, self.num_bits_params)

    def quant_bias(self, b: nn.Parameter):
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

    def linear_forward(self, input_q, w_q, b_q):
        w_old, b_old = self.linear.weight, self.linear.bias
        # We do a trick to save the original module's attributes but forward with quantized weights -
        # Replace only the weight/biases of the model, forward, and then replace back.
        self.linear.weight = w_q
        self.linear.bias = b_q
        y = self.linear(input_q)
        self.linear.weight = w_old
        self.linear.bias = b_old
        return y

    @property
    def bn_inverse_sigma(self):
        # .clone().detach() to remove history from these tensors.
        return torch.rsqrt(self.bn.running_var + self.bn.eps).clone().detach()

    def batch_stats(self, x):
        """
        Get the batch mean and variance of x.
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
        return mean, var



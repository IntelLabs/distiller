from .range_linear import *
from torch.nn import functional as F

__all__ = ['FusedLinearBatchNorm']

# Number of steps before freezing the batch norm running average and variance
FREEZE_BN_DELAY_DEFAULT = 200000


def _broadcast_correction_factor(c, broadcast_to_shape):
    """
    Returns a view of `c` which is broadcastable with shape `broadcast_to_shape`.
    """
    filler_dims = (1,) * (len(broadcast_to_shape) - len(c.shape) - 1)
    view_dims = (*c.shape, *filler_dims)
    return c.view(view_dims)


class FusedLinearBatchNorm(nn.Module):
    def __init__(self, linear_module, bn, quantized=False, num_bits_acts=8, num_bits_accum=32, num_bits_params=8,
                 mode=LinearQuantMode.SYMMETRIC, dequantize=True,
                 freeze_bn_delay=FREEZE_BN_DELAY_DEFAULT):
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
        Note:
            The quantized version was implemented according to https://arxiv.org/pdf/1806.08342.pdf Section 3.2.2.
        """
        FusedLinearBatchNorm.verify_module_types(linear_module, bn)
        if not bn.track_running_stats or not bn.affine:
            raise ValueError("Quantization is only supported for BatchNorm which tracks running stats with"
                             "affine weights.")
        super(FusedLinearBatchNorm, self).__init__()
        self.dequantize = dequantize
        self.mode = verify_quant_mode(mode)
        self.num_bits_acts = num_bits_acts
        self.num_bits_params = num_bits_params
        self.num_bits_accum = num_bits_accum
        self.linear = linear_module
        self.bn = bn
        self.freeze_bn_delay = freeze_bn_delay
        self._has_bias = (self.linear.bias is not None)
        self.quantized = quantized
        if isinstance(linear_module, nn.Linear):
            self.linear_forward_fn = self._linear_layer_forward
            self.linear_type = "fc"
        else:
            self.linear_forward_fn = self._conv2d_layer_forward
            self.linear_type = "conv2d"

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
        """
        According to https://arxiv.org/pdf/1806.08342.pdf section 3.2.2.
        Note:
            The linear layer bias doesn't get included in the calculation!
            When calculating the batch norm,
            the bias offsets the mean and so when calculating (x - mu) we get the unbiased position
            w.r.t. to the mean.
            i.e. the result of the forward is:
            bn(linear(x)) = ( linear(x) - E(linear(x)) ) * gamma / std(linear(x)) + beta =
                          = ( x*W + B - E(x*W +B) ) * gamma / E((x*W+ B - E(x*W +B))^2) + beta =
                          = (x*W -E(x*W)) * gamma / std(x*W) + beta
        """
        w, b, gamma, beta = self.linear.weight, self.linear.bias, self.bn.weight, self.bn.bias
        batch_mean = batch_var = sigma_batch = None
        if self.training:
            batch_mean, batch_var = self.batch_stats(self.linear_forward_fn(x, w), b)
            sigma_batch = torch.sqrt(batch_var + self.bn.eps)
        with torch.no_grad():
            recip_sigma_moving = torch.rsqrt(self.bn.running_var + self.bn.eps)
        c = sigma_batch * recip_sigma_moving if self.training else None
        w_corrected = w * self.broadcast_correction_weight(gamma * recip_sigma_moving)
        w_quantized = self.quant_weights(w_corrected)
        y = self.linear_forward_fn(x, w_quantized, None)
        if self.training:
            y_corrected = y / self.broadcast_correction(c)
            bias_corrected = beta - gamma * batch_mean / sigma_batch
        else:
            y_corrected = y
            corrected_mean = self.bn.running_mean - (b if b is not None else 0)
            bias_corrected = beta - gamma * corrected_mean * recip_sigma_moving

        return y_corrected + self.broadcast_correction(bias_corrected)

    def broadcast_correction(self, c: torch.Tensor):
        """
        Broadcasts a correction factor to the output for elementwise operations.
        """
        expected_output_dim = 2 if self.linear_type == "fc" else 4
        view_fillers_dim = expected_output_dim - c.dim() - 1
        view_filler = (1,) * view_fillers_dim
        expected_view_shape = c.shape + view_filler
        return c.view(*expected_view_shape)

    def broadcast_correction_weight(self, c: torch.Tensor):
        """
        Broadcasts a correction factor to the weight.
        """
        if c.dim() != 1:
            raise ValueError("Correction factor needs to have a single dimension")
        expected_weight_dim = 2 if self.linear_type == "fc" else 4
        view_fillers_dim = expected_weight_dim - c.dim()
        view_filler = (1,) * view_fillers_dim
        expected_view_shape = c.shape + view_filler
        return c.view(*expected_view_shape)

    def quant_weights(self, w: nn.Parameter):
        if not self.quantized:
            return w
        return self._quant_param(w, self.num_bits_params)

    def quant_bias(self, b: nn.Parameter):
        if b is None or not self.quantized:
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

    def batch_stats(self, x, bias=None):
        """
        Get the batch mean and variance of x and updates the BatchNorm's running mean and average.
        Args:
            x (torch.Tensor): input batch.
            bias (torch.Tensor): the bias that is to be applied to the batch.
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
        n = x.numel() / channel_size
        mean = x.transpose(0, 1).contiguous().view(channel_size, -1).mean(1)
        # BatchNorm currently uses biased variance (without Bessel's correction) as was discussed at
        # https://github.com/pytorch/pytorch/issues/1410
        #
        # also see the source code itself:
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L216
        var = x.transpose(0, 1).contiguous().view(channel_size, -1).var(1, unbiased=False)
        with torch.no_grad():
            if self.bn.momentum:
                biased_mean = mean + (bias if bias is not None else 0)
                self.bn.running_mean.mul_(1-self.bn.momentum).add_(self.bn.momentum * biased_mean)
                # However - running_var is updated using unbiased variance!
                # as seen in the source code:
                # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L223
                unbiased_var = var * (n / (n-1))
                self.bn.running_var.mul_(1-self.bn.momentum).add_(self.bn.momentum * unbiased_var)
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
        self.bn.num_batches_tracked += 1
        return mean, var

    def _linear_layer_forward(self, input, w, b=None):
        return F.linear(input, w, b)

    def _conv2d_layer_forward(self, input, w, b=None):
        # We copy the code from the Conv2d forward, but plug in our weights.
        conv = self.linear  # type: nn.Conv2d
        if conv.padding_mode == 'circular':
            expanded_padding = [(conv.padding[1] + 1) // 2, conv.padding[1] // 2,
                                (conv.padding[0] + 1) // 2, conv.padding[0] // 2]
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            w, b, conv.stride,
                            (0, 0), conv.dilation, conv.groups)
        return F.conv2d(input, w, b, conv.stride,
                        conv.padding, conv.dilation, conv.groups)


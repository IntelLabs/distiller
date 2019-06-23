import torch
import torch.nn as nn
from torch.nn import functional as F

__all__ = ['SimulatedFoldedBatchNorm']

# Number of steps before freezing the batch norm running average and variance
FREEZE_BN_DELAY_DEFAULT = 200000


def _broadcast_correction_factor(c, broadcast_to_shape):
    """
    Returns a view of `c` which is broadcastable with shape `broadcast_to_shape`.
    """
    filler_dims = (1,) * (len(broadcast_to_shape) - len(c.shape) - 1)
    view_dims = (*c.shape, *filler_dims)
    return c.view(view_dims)


class SimulatedFoldedBatchNorm(nn.Module):
    def __init__(self, param_module, bn, freeze_bn_delay=FREEZE_BN_DELAY_DEFAULT, param_quantization_fn=None):
        """
        Wrapper for simulated folding of BatchNorm into convolution / linear layers during training
        Args:
            param_module (nn.Linear or nn.Conv2d): the wrapped parameter layer
            bn (nn.BatchNorm1d or nn.BatchNorm2d): batch normalization
            freeze_bn_delay (int): number of steps before freezing the batchnorm running stats
            param_quantization_fn (function): function to be used for weight/bias quantization
        Note:
            The quantized version was implemented according to https://arxiv.org/pdf/1806.08342.pdf Section 3.2.2.
        """
        SimulatedFoldedBatchNorm.verify_module_types(param_module, bn)
        if not bn.track_running_stats or not bn.affine:
            raise ValueError("Simulated BN folding is only supported for BatchNorm which tracks running stats with"
                             "affine weights.")
        super(SimulatedFoldedBatchNorm, self).__init__()
        self.param_module = param_module
        self.bn = bn
        self.freeze_bn_delay = freeze_bn_delay
        self.frozen = False
        self._has_bias = (self.param_module.bias is not None)
        self.param_quant_fn = param_quantization_fn
        if isinstance(param_module, nn.Linear):
            self.param_forward_fn = self._linear_layer_forward
            self.param_module_type = "fc"
        else:
            self.param_forward_fn = self._conv2d_layer_forward
            self.param_module_type = "conv2d"

    @staticmethod
    def verify_module_types(param_module, bn):
        if not isinstance(param_module, (nn.Linear, nn.Conv2d)) \
                and not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d)):
            raise TypeError("Only supporting fusing nn.BatchNorm1d/nn.BatchNorm2d into nn.Linear/nn.Conv2d.")
        if isinstance(param_module, nn.Linear) and isinstance(bn, nn.BatchNorm2d):
            raise TypeError("nn.Linear layer has to be followed by a nn.BatchNorm1d layer.")
        if isinstance(param_module, nn.Conv2d) and isinstance(bn, nn.BatchNorm1d):
            raise TypeError("nn.Con2d layer has to be followed by a nn.BatchNorm2d layer.")

    def forward(self, x):
        """
        According to https://arxiv.org/pdf/1806.08342.pdf section 3.2.2.
        Note:
            The param layer bias doesn't get included in the calculation!
            When calculating the batch norm,
            the bias offsets the mean and so when calculating (x - mu) we get the unbiased position
            w.r.t. to the mean.
            i.e. the result of the forward is:
            bn(param(x)) = ( param(x) - E(param(x)) ) * gamma / std(param(x)) + beta =
                          = ( x*W + B - E(x*W +B) ) * gamma / sqrt(E((x*W+ B - E(x*W +B))^2)) + beta =
                          = (x*W -E(x*W)) * gamma / std(x*W) + beta
        """
        if not self.frozen:
            w, b, gamma, beta = self.param_module.weight, self.param_module.bias, self.bn.weight, self.bn.bias
            if self.training:
                batch_mean, batch_var = self.batch_stats(self.param_forward_fn(x, w), b)
                recip_sigma_batch = torch.rsqrt(batch_var + self.bn.eps)
                with torch.no_grad():
                    sigma_running = torch.sqrt(self.bn.running_var + self.bn.eps)
                w_corrected = w * self.broadcast_correction_weight(gamma / sigma_running)
                w_quantized = self._quant_param(w_corrected)
                recip_c = self.broadcast_correction(sigma_running * recip_sigma_batch)
                bias_corrected = beta - gamma * batch_mean * recip_sigma_batch
                bias_quantized = self.broadcast_correction(self._quant_param(bias_corrected))
                y = self.param_forward_fn(x, w_quantized, None)
                y.mul_(recip_c).add_(bias_quantized)
            else:
                with torch.no_grad():
                    recip_sigma_running = torch.rsqrt(self.bn.running_var + self.bn.eps)
                w_corrected = w * self.broadcast_correction_weight(gamma * recip_sigma_running)
                w_quantized = self._quant_param(w_corrected)
                corrected_mean = self.bn.running_mean - (b if b is not None else 0)
                bias_corrected = beta - gamma * corrected_mean * recip_sigma_running
                bias_quantized = self._quant_param(bias_corrected)
                y = self.param_forward_fn(x, w_quantized, bias_quantized)
        else:
            w, b = self.param_module.weight, self.param_module.bias
            w_quantized, bias_quantized = self._quant_param(w), self._quant_param(b)
            y = self.param_forward_fn(x, w_quantized, bias_quantized)

        return y

    def broadcast_correction(self, c: torch.Tensor):
        """
        Broadcasts a correction factor to the output for elementwise operations.
        """
        expected_output_dim = 2 if self.param_module_type == "fc" else 4
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
        expected_weight_dim = 2 if self.param_module_type == "fc" else 4
        view_fillers_dim = expected_weight_dim - c.dim()
        view_filler = (1,) * view_fillers_dim
        expected_view_shape = c.shape + view_filler
        return c.view(*expected_view_shape)

    def _quant_param(self, t: torch.Tensor):
        """
        Quantize a parameter locally.
        """
        if t is None or self.param_quant_fn is None:
            return t
        return self.param_quant_fn(t)

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
        self.bn.num_batches_tracked += 1

        # Calculate current batch stats
        batch_mean = x.transpose(0, 1).contiguous().view(channel_size, -1).mean(1)
        # BatchNorm currently uses biased variance (without Bessel's correction) as was discussed at
        # https://github.com/pytorch/pytorch/issues/1410
        #
        # also see the source code itself:
        # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L216
        batch_var = x.transpose(0, 1).contiguous().view(channel_size, -1).var(1, unbiased=False)

        # Update running stats
        with torch.no_grad():
            biased_batch_mean = batch_mean + (bias if bias is not None else 0)
            # However - running_var is updated using unbiased variance!
            # https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/Normalization.cpp#L223
            n = x.numel() / channel_size
            corrected_var = batch_var * (n / (n - 1))
            momentum = self.bn.momentum
            if momentum is None:
                # momentum is None - we compute a cumulative moving average
                # as noted in https://pytorch.org/docs/stable/nn.html#batchnorm2d
                momentum = 1. / float(self.bn.num_batches_tracked)
            self.bn.running_mean.mul_(1 - momentum).add_(momentum * biased_batch_mean)
            self.bn.running_var.mul_(1 - momentum).add_(momentum * corrected_var)

        if self.bn.num_batches_tracked > self.freeze_bn_delay:
            self.freeze()

        return batch_mean, batch_var

    def _linear_layer_forward(self, input, w, b=None):
        return F.linear(input, w, b)

    def _conv2d_layer_forward(self, input, w, b=None):
        # We copy the code from the Conv2d forward, but plug in our weights.
        conv = self.param_module  # type: nn.Conv2d
        if conv.__dict__.get('padding_mode', None) == 'circular':  # This attribute doesn't exist yet in pytorch 1.0.1
            expanded_padding = [(conv.padding[1] + 1) // 2, conv.padding[1] // 2,
                                (conv.padding[0] + 1) // 2, conv.padding[0] // 2]
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            w, b, conv.stride,
                            (0, 0), conv.dilation, conv.groups)
        return F.conv2d(input, w, b, conv.stride,
                        conv.padding, conv.dilation, conv.groups)

    def freeze(self):
        w, b, gamma, beta = self.param_module.weight, self.param_module.bias, self.bn.weight, self.bn.bias
        with torch.no_grad():
            recip_sigma_running = torch.rsqrt(self.bn.running_var + self.bn.eps)
            w.mul_(self.broadcast_correction_weight(gamma * recip_sigma_running))
            corrected_mean = self.bn.running_mean - (b if b is not None else 0)
            bias_corrected = beta - gamma * corrected_mean * recip_sigma_running
            if b is not None:
                b.copy_(bias_corrected)
            else:
                self.param_module.bias = nn.Parameter(bias_corrected)
        self.frozen = True

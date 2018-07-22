# Quantization Algorithms

The following quantization methods are currently implemented in Distiller:

## DoReFa

(As proposed in [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](https://arxiv.org/abs/1606.06160))  
  
In this method, we first define the quantization function \(quantize_k\), which takes a real value \(a_f \in [0, 1]\) and outputs a discrete-valued \(a_q \in \left\{ \frac{0}{2^k-1}, \frac{1}{2^k-1}, ... , \frac{2^k-1}{2^k-1} \right\}\), where \(k\) is the number of bits used for quantization.

\[a_q = quantize_k(a_f) = \frac{1}{2^k-1} round \left( \left(2^k - 1 \right) a_f \right)\]

Activations are clipped to the \([0, 1]\) range and then quantized as follows:

\[x_q = quantize_k(x_f)\]

For weights, we define the following function \(f\), which takes an unbounded real valued input and outputs a real value in \([0, 1]\):

\[f(w) = \frac{tanh(w)}{2 max(|tanh(w)|)} + \frac{1}{2} \]

Now we can use \(quantize_k\) to get quantized weight values, as follows:

\[w_q = 2 quantize_k \left( f(w_f) \right) - 1\]

This method requires training the model with quantization, as discussed [here](quantization.md#training-with-quantization). Use the `DorefaQuantizer` class to transform an existing model to a model suitable for training with quantization using DoReFa.

### Notes:

- Gradients quantization as proposed in the paper is not supported yet.
- The paper defines special handling for binary weights which isn't supported in Distiller yet.

## PACT

(As proposed in [PACT: Parameterized Clipping Activation for Quantized Neural Networks](https://arxiv.org/abs/1805.06085))

This method is similar to DoReFa, but the upper clipping values, \(\alpha\), of the activation functions are learned parameters instead of hard coded to 1. Note that per the paper's recommendation, \(\alpha\) is shared per layer.

This method requires training the model with quantization, as discussed [here](quantization/#training-with-quantization). Use the `PACTQuantizer` class to transform an existing model to a model suitable for training with quantization using PACT.

## WRPN

(As proposed in [WRPN: Wide Reduced-Precision Networks](https://arxiv.org/abs/1709.01134))  

In this method, activations are clipped to \([0, 1]\) and quantized as follows (\(k\) is the number of bits used for quantization):

\[x_q = \frac{1}{2^k-1} round \left( \left(2^k - 1 \right) x_f \right)\]

Weights are clipped to \([-1, 1]\) and quantized as follows:

\[w_q = \frac{1}{2^{k-1}-1} round \left( \left(2^{k-1} - 1 \right)w_f \right)\]

Note that \(k-1\) bits are used to quantize weights, leaving one bit for sign.

This method requires training the model with quantization, as discussed [here](quantization/#training-with-quantization). Use the `WRPNQuantizer` class to transform an existing model to a model suitable for training with quantization using WRPN.

### Notes:

- The paper proposed widening of layers as a means to reduce accuracy loss. This isn't implemented as part of `WRPNQuantizer` at the moment. To experiment with this, modify your model implementation to have wider layers.
- The paper defines special handling for binary weights which isn't supported in Distiller yet.

## Symmetric Linear Quantization

In this method, a float value is quantized by multiplying with a numeric constant (the **scale factor**), hence it is **Linear**. We use a signed integer to represent the quantized range, with no quantization bias (or "offset") used. As a result, the floating-point range considered for quantization is **symmetric** with respect to zero.  
In the current implementation the scale factor is chosen so that the entire range of the floating-point tensor is quantized (we do not attempt to remove outliers).  
Let us denote the original floating-point tensor by \(x_f\), the quantized tensor by \(x_q\), the scale factor by \(q_x\) and the number of bits used for quantization by \(n\). Then, we get:
\[q_x = \frac{2^{n-1}-1}{\max|x|}\]
\[x_q = round(q_x x_f)\]
(The \(round\) operation is round-to-nearest-integer)  
  
Let's see how a **convolution** or **fully-connected (FC)** layer is quantized using this method: (we denote input, output, weights and bias with  \(x, y, w\) and \(b\) respectively)
\[y_f = \sum{x_f w_f} + b_f = \sum{\frac{x_q}{q_x} \frac{w_q}{q_w}} + \frac{b_q}{q_b} = \frac{1}{q_x q_w} \left( \sum { x_q w_q + \frac{q_x q_w}{q_b}b_q } \right)\]
\[y_q = round(q_y y_f) = round\left(\frac{q_y}{q_x q_w} \left( \sum { x_q w_q + \frac{q_x q_w}{q_b}b_q } \right) \right) \]
Note how the bias has to be re-scaled to match the scale of the summation.

### Implementation

We've implemented **convolution** and **FC** using this method.  

- They are implemented by wrapping the existing PyTorch layers with quantization and de-quantization operations. That is - the computation is done on floating-point tensors, but the values themselves are restricted to integer values. The wrapper is implemented in the `RangeLinearQuantParamLayerWrapper` class.  
- All other layers are unaffected and are executed using their original FP32 implementation.  
- To automatically transform an existing model to a quantized model using this method, use the `SymmetricLinearQuantizer` class.
- For weights and bias the scale factor is determined once at quantization setup ("offline"), and for activations it is determined dynamically at runtime ("online").  
- **Important note:** Currently, this method is implemented as **inference only**, with no back-propagation functionality. Hence, it can only be used to quantize a pre-trained FP32 model, with no re-training. As such, using it with \(n < 8\) is likely to lead to severe accuracy degradation for any non-trivial workload.
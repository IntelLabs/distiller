# Quantization Algorithms

## Symmetric Linear Quantization

In this method, a float value is quantized by multiplying with a numeric constant (the **scale factor**), hence it is **Linear**. We use a signed integer to represent the quantized range, with no quantization bias (or "offset") used. As a result, the floating-point range considered for quantization is **symmetric** with respect to zero.  
In the current implementation the scale factor is chosen so that the entire range of the floating-point tensor is quantized (we do not attempt to remove outliers).  
Let us denote the original floating-point tensor by \(x_f\), the quantized tensor by \(x_q\), the scale factor by \(q_x\) and the number of bits used for quantization by \(n\). Then, we get:
\[q_x = \frac{2^{n-1}-1}{\max|x|}\]
\[x_q = round(q_x x_f)\]
(The \(round\) operation is round-to-nearest-integer)  
  
Let's see how a **convolution** or **fully-connected (FC)** layer is quantized using this method: (we denote input, output, weights and bias with  \(x, y, w\) and \(b\) respectively)
\[y_f = \sum{x_f w_f} + b_f = \sum{\frac{x_q}{q_x} \frac{w_q}{q_w}} + \frac{b_q}{q_b} = \frac{1}{q_x q_w} \sum{(x_q w_q + \frac{q_b}{q_x q_w}b_q)}\]
\[y_q = round(q_y y_f) = round(\frac{q_y}{q_x q_w} \sum{(x_q w_q + \frac{q_b}{q_x q_w}b_q)})\]
Note how the bias has to be re-scaled to match the scale of the summation.

### Implementation
We've implemented **convolution** and **FC** using this method.  

- They are implemented by wrapping the existing PyTorch layers with quantization and de-quantization operations. That is - the computation is done on floating-point tensors, but the values themselves are restricted to integer values.  
- All other layers are unaffected and are executed using their original FP32 implementation.  
- For weights and bias the scale factor is determined once at quantization setup ("offline"), and for activations it is determined dynamically at runtime ("online").  
- **Important note:** Currently, this method is implemented as **inference only**, with no back-propagation functionality. Hence, it can only be used to quantize a pre-trained FP32 model, with no re-training. As such, using it with \(n < 8\) is likely to lead to severe accuracy degradation for any non-trivial workload.
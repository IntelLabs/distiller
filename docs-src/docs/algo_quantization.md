# Quantization Algorithms

## Symmetric Linear Quantization

**Linear** - float value quantized by multiplying with scale factor. **Symmetric** - no quantization bias (or "offset") used, so zero in the float domain is mapped to zero in the integer domain.  
In the current implementation the scale factor is chosen so that the entire range of the tensor is quantized. So, we get: (Using \(q\) to denote the scale factor and \(x\) to denote the tensor being quantized)
\[q_x = \frac{2^n-1}{\max|x|}\]
\[x_q = round(q_x\cdot x_f)\]
Where \(n\) is the number of bits used for quantization.  
For weights and bias the scale factor is determined once at quantization setup ("offline"), and for activations it is determined dynamically at runtime ("online").  
Currently, quantized implementations are provided for **convolution** and **fully-connected** layers. These layers are quantized as follows (using \(x, y, w, b\) for input, output, weights, bias respectively):
\[y_f = \sum{x_f\cdot w_f} + b_f = \sum{x_f\cdot w_f} + b_f = \sum{\frac{x_q}{q_x}\cdot \frac{w_q}{q_w}} + \frac{b_q}{q_b} = \frac{1}{q_x q_w} \sum{(x_q w_q + \frac{q_b}{q_x q_w}b_q)}\]
\[y_q = round(q_y\cdot y_f) = round(\frac{q_y}{q_x q_w} \sum{(x_q w_q + \frac{q_b}{q_x q_w}b_q)})\]
Note how the bias has to be re-scaled to match the scale of the summation.  
All other layers are executed in FP32. This is done by adding quantize and de-quantize operations at the beginning and end of the quantized layers.  

!!! note
This method is implemented as **inference only**, with no back-propagation functionality. Hence, it can only be used to quantize a pre-trained FP32 model, and as such, using it with \(n < 8\) is likely to lead to severe accuracy degradation for any non-trivial workload.
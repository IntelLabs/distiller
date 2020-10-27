# Distiller Documentation
## What is Distiller

**[Distiller](https://github.com/IntelLabs/distiller/)** is an open-source Python package for neural network compression research.

Network compression can reduce the footprint of a neural network, increase its inference speed and save energy. Distiller provides a [PyTorch](http://pytorch.org/) environment for prototyping and analyzing compression algorithms, such as sparsity-inducing methods and low precision arithmetic.

Distiller contains:

- A framework for integrating pruning, regularization and quantization algorithms.
- A set of tools for analyzing and evaluating compression performance.
- Example implementations of state-of-the-art compression algorithms.

## Motivation
A sparse tensor is any tensor that contains some zeros, but sparse tensors are usually only interesting if they contain a significant number of zeros.  A sparse neural network performs computations using some sparse tensors (preferably many).  These tensors can be parameters (weights and biases) or activations (feature maps).

Why do we care about sparsity?<br>
Present day neural networks tend to be deep, with millions of weights and activations.  Refer to GoogLeNet or ResNet50, for a couple of examples.
These large models are compute-intensive which means that even with dedicated acceleration hardware, the inference pass (network evaluation) will take time.  You might think that latency is an issue only in certain cases, such as autonomous driving systems, but in fact, whenever we humans interact with our phones and computers, we are sensitive to the latency of the interaction.  We don't like to wait for search results or for an application or web-page to load, and we are especially sensitive in realtime interactions such as speech recognition.  So inference latency is often something we want to minimize.
<br><br>
Large models are also memory-intensive with millions of parameters.  Moving around all of the data required to compute inference results consumes energy, which is a problem on a mobile device as well as in a server environment.  Data center server-racks are limited by their power-envelope and their ToC (total cost of ownership) is correlated to their power consumption and thermal characteristics.  In the mobile device environment, we are obviously always aware of the implications of power consumption on the device battery.
Inference performance in the data center is often measured using a KPI (key performance indicator) which folds latency and power considerations: inferences per second, per Watt (inferences/sec/watt).
<br><br>
The storage and transfer of large neural networks is also a challenge in mobile device environments, because of limitations on application sizes and long application download times.
<br><br>
For these reasons, we wish to compress the network as much as possible, to reduce the amount of bandwidth and compute required.  Inducing sparseness, through regularization or pruning, in neural-network models, is one way to compress the network (quantization is another method).
Sparse neural networks hold the promise of speed, small size, and energy efficiency.  

### Smaller
Sparse NN model representations can be compressed by taking advantage of the fact that the tensor elements are dominated by zeros.  The compression format, if any, is very HW and SW specific, and the optimal format may be different per tensor (an obvious example: largely dense tensors should not be compressed).  The compute hardware needs to support the compressions formats, for representation compression to be meaningful.  Compression representation decisions might interact with algorithms such as the use of tiles for memory accesses.  Data such as a parameter tensor is read/written from/to main system memory compressed, but the computation can be dense or sparse.  In dense compute we use dense operators, so the compressed data eventually needs to be decompressed into its full, dense size.  The best we can do is bring the compressed representation as close as possible to the compute engine.<br>
Sparse compute, on the other hand, operates on the sparse representation which never requires decompression (we therefore distinguish between sparse representation and compressed representation).  This is not a simple matter to implement in HW, and often means lower utilization of the vectorized compute engines.  Therefore, there is a third class of representations, which take advantage of specific hardware characteristics.  For example, for a vectorized compute engine we can remove an entire zero-weights vector and skip its computation (this uses structured pruning or regularization).

### Faster
Many of the layers in modern neural-networks are bandwidth-bound, which means that the execution latency is dominated by the available bandwidth. In essence, the hardware spends more time bringing data close to the compute engines, than actually performing the computations.  Fully-connected layers, RNNs and LSTMs are some examples of bandwidth-dominated operations.<br>
Reducing the bandwidth required by these layers, will immediately speed them up.<br>
Some pruning algorithms prune entire kernels, filters and even layers from the network without adversely impacting the final accuracy.  Depending on the hardware implementation, these methods can be leveraged to skip computations, thus reducing latency and power.

### More energy efficient
Because we pay two orders-of-magnitude more energy to access off-chip memory (e.g. DDR) compared to on-chip memory (e.g. SRAM or cache), many hardware designs employ a multi-layered cache hierarchy.  Fitting the parameters and activations of a network in these on-chip caches can make a big difference on the required bandwidth, the total inference latency, and off course reduce power consumption.<br>
And of course, if we used a sparse or compressed representation, then we are reducing the data throughput and therefore the energy consumption.

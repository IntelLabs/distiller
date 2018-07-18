# Early Exit Inference
While Deep Neural Networks benefit from a large number of layers, it's often the case that many datapoints in classification tasks can be classified accurately with much less work. There have been several studies recently regarding the idea of exiting before the normal endpoint of the neural network. Panda et al in [Conditional Deep Learning for Energy-Efficient and Enhanced Pattern Recognition](#panda) points out that a lot of data points can be classified easily and require less processing than some more difficult points and they view this in terms of power savings. Surat et al in [BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks](#branchynet) look at a selective approach to exit placement and criteria for exiting early.

## Why Does Early Exit Work?
Early Exit is a strategy with a straightforward and easy to understand concept Figure #fig(boundaries) shows a simple example in a 2-D feature space. While deep networks can representative more complex and expressive boundaries between classes (assuming we’re confident of avoiding over-fitting the data), it’s also clear that much of the data can be properly classified with even the simplest of classification boundaries.

![Figure !fig(boundaries): Simple and more expressive classification boundaries](/docs-src/docs/imgs/decision_boundary.png)

Data points far from the boundary can be considered "easy to classify" and achieve a high degree of confidence quicker than do data points close to the boundary. In fact, we can think of the area between the outer straight lines as being the region that is "difficult to classify" and require the full expressiveness of the neural network to accurately classify it.

## Example code for Early Exit
Both CIFAR10 and Imagenet code comes directly from publically available examples from Pytorch. The only edits are the exits that are inserted in a methodology similar to BranchyNet work.

Deeper networks can benefit from multiple exits. Our examples illustrate both a single and a pair of early exits for CIFAR10 and Imagenet, respectively.

### Heuristics
The insertion of the exits are ad-hoc, but there are some heuristic principals guiding their placement and parameters. The earlier exits are placed, the more agressive the exit as it essentially prunes the rest of the network at a very early stage, thus saving a lot of work. However, a diminishing percentage of data will be directed through the exit if we are to preserve accuracy.

### Early Exit Parameters
There are two parameters that are required to enable early exit. Leave them undefined if you are not enabling Early Exit:

1. **--earlyexit** defines the
thresholds for each of the early exits. The output cross entropy must be less than these thresholds to take this exit, otherwise the data continues along the regular path.

2. **--lossweights** provide the weights for the linear combination of losses during training to compute a signle, overall loss.

### CIFAR10
In the case of CIFAR10, we have inserted a single exit after the first full layer grouping. The layers on the exit path itself includes a convolutional layer and a fully connected layer. If you move the exit, be sure to match the proper sizes for inputs and outputs to the exit layers.

### Imagenet
The imagenet subdirectory is taken directly from the torchvision source code. This supports training and inference of the imagenet dataset via several well known deep architectures. ResNet-50 is the architecture of interest in this study, however the exit is defined in the generic resnet code and could be used with other size resnets. There are two exits inserted in this example. Again, exit layers must have their sizes match properly.

## References
<div id="panda"></div> **Priyadarshini Panda, Abhronil Sengupta, Kaushik Roy**.
    [*Conditional Deep Learning for Energy-Efficient and Enhanced Pattern Recognition*](https://arxiv.org/abs/1509.08971v6), arXiv:1509.08971v6, 2017.

<div id="branchynet"></div> **Surat Teerapittayanon, Bradley McDanel, H. T. Kung**.
    [*BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks*](http://arxiv.org/abs/1709.01686), arXiv:1709.01686, 2017.

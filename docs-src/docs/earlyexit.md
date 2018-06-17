# Early Exit Inference
While Deep Neural Networks benefit from a large number of layers, it's often the case that many datapoints in classification tasks can be classified accurately with much less work. There have been several studies recently regarding the idea of exiting before the normal endpoint of the neural network. Panda et al in [Conditional Deep Learning for Energy-Efficient and Enhanced Pattern Recognition](#panda) points out that a lot of data points can be classified easily and require less processing than some more difficult points and they view this in terms of power savings. Surat et al in [BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks](#branchynet) look at a selective approach to exit placement and criteria for exiting early.

## Why Does Early Exit Work?
Early Exit is a strategy with a straightforward and easy to understand concept Figure #fig(boundaries) shows a simple example in a 2-D feature space. While a deep network can representative a more complex and expressive boundary between classes (assuming we’re confident of avoiding over-fitting the data), it’s also clear that much of the data can be properly classified with even the simplest of classification boundaries.

![Figure !fig(boundaries): Simple and more expressive classification boundaries](docs-src/docs/imgs/decision_boundary.png)

Data points far from the boundary can be considered "easy to classify" and achieve a high degree of confidence quicker than do data points close to the boundary. In fact, we can think of the area between the outer straight lines as being the region that is "difficult to classify" and require the full expressiveness of the neural network to accurately classify it.

## Example code for Early Exit
The CIFAR10 image set example is based on code from Yerlan Idelbayev. The imagenet example code based on pytorch/torchvision example for resnet-50 and imagenet. Exits are inserted in a methodology similar to BranchyNet work.

### CIFAR10
The cifar10 subdirectory is taken from Yerlan Idelbayev's Github repo available at https://github.com/akamaster/pytorch_resnet_cifar10. The code has been modified to include a single early exit. 

### imagenet
The imagenet subdirectory is taken directly from the torchvision source code. This supports training and inference of the imagenet dataset via several well known deep architectures. ResNet-50 is the architecture of interest in this study.

### Running an interactive session inside kubernetes
You can work interactively within a container provisioned by kubernetes using the [interactive pod](docker/interactive_pod.yaml). This pod has all of the same volumes mounted as the training pod but does not run any code. The pod can be created using `kubectl create -f docker/interactive_pod.yaml`. Take note of the name of the pod you just created. Once the pod is created, you can access a bash terminal within it using `kubectl exec -it <name of created pod> -- bash`. Remember to set your default namespace accordingly.

## References
<div id="panda"></div> **Priyadarshini Panda, Abhronil Sengupta, Kaushik Roy**.
    [*Conditional Deep Learning for Energy-Efficient and Enhanced Pattern Recognition*](https://arxiv.org/abs/1509.08971v6), arXiv:1509.08971v6, 2017.

<div id="branchynet"></div> **Surat Teerapittayanon, Bradley McDanel, H. T. Kung**.
    [*BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks*](http://arxiv.org/abs/1709.01686), arXiv:1709.01686, 2017.
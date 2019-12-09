# Image Classifiers Compression

This is Distiller's main example application for compressing image classification models.

- [Image Classifiers Compression](#image-classifiers-compression)
  - [Usage](#usage)
  - [Compression Methods](#compression-methods)
    - [Sparsity - Pruning and regularization](#sparsity---pruning-and-regularization)
    - [Quantization](#quantization)
    - [Knowledge Distillation](#knowledge-distillation)
    - [Early-exit](#early-exit)
  - [Models and Datasets Supported](#models-and-datasets-supported)
  - [Re-usable Image Classification Code](#re-usable-image-classification-code)

## Usage

Please see the [docs](https://nervanasystems.github.io/distiller/usage.html) for usage details. In addition, run `compress_classifier.py -h` to show the extensive list of command-line options available.

## Compression Methods

The following compression methods are implemented for image classifiers: (follow the links for more details on each method and experiment results)

### Sparsity - Pruning and regularization

A non-exhaustive list of the methods implemented:

* [AGP](https://github.com/NervanaSystems/distiller/tree/master/examples/agp-pruning)
* [DropFilter](https://github.com/NervanaSystems/distiller/tree/master/examples/drop_filter)
* [Lottery-Ticket Hypothesis](https://github.com/NervanaSystems/distiller/tree/master/examples/lottery_ticket)
* [Network Surgery](https://github.com/NervanaSystems/distiller/tree/master/examples/network_surgery)
* [Network Trimming](https://github.com/NervanaSystems/distiller/tree/master/examples/network_trimming)
* [Hybrids](https://github.com/NervanaSystems/distiller/tree/master/examples/hybrid): These are examples where multiple pruning strategies are combined.

### Quantization

* [Post-training quantization](https://github.com/NervanaSystems/distiller/blob/update_readmes/examples/quantization/post_train_quant/command_line.md) based on the TensorFlow quantization scheme (originally GEMMLOWP) with additional capabilities.
* [Quantization-aware training](https://github.com/NervanaSystems/distiller/tree/master/examples/quantization/quant_aware_train): TensorFlow scheme, DoReFa, PACT

### Knowledge Distillation

See details in the [docs](https://nervanasystems.github.io/distiller/schedule.html#knowledge-distillation), and these YAML schedules training ResNet on CIFAR-10 with knowledge distillation: [FP32](https://github.com/NervanaSystems/distiller/blob/update_readmes/examples/quantization/fp32_baselines/preact_resnet_cifar_base_fp32.yaml) ; [DoReFa](https://github.com/NervanaSystems/distiller/blob/update_readmes/examples/quantization/quant_aware_train/preact_resnet_cifar_dorefa.yaml).

### Early-exit

See details in the [docs](https://nervanasystems.github.io/distiller/algo_earlyexit.html).

## Models and Datasets Supported

The sample app integrates with [TorchVision](https://pytorch.org/docs/master/torchvision/models.html#classification) and [Cadene's pre-trained models](https://github.com/Cadene/pretrained-models.pytorch). Barring specific issues, any model from these two repositories can be specified from the command line and used.

We've implemented additional models, which can be found [here](https://github.com/NervanaSystems/distiller/tree/master/distiller/models).

The application supports ImageNet, CIFAR-10 and MNIST.

## Re-usable Image Classification Code

We borrow the main flow code from PyTorch's ImageNet classification training sample application ([see here](https://github.com/pytorch/examples/tree/master/imagenet)). Much of the flow was refactored into a class called `ClassifierCompressor`, which can be re-used to build different scripts that perform image classifiers compression. Its implementation can be found in [`distiller/apputils/image_classifier.py`](https://github.com/NervanaSystems/distiller/blob/update_readmes/distiller/apputils/image_classifier.py).  

The [AMC auto-compression](https://github.com/NervanaSystems/distiller/tree/master/examples/auto_compression/amc) sample is another application that uses this building block.

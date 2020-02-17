# Image Classifiers Compression

This is Distiller's main example application for compressing image classification models.

- [Image Classifiers Compression](#image-classifiers-compression)
  - [Usage](#usage)
  - [Compression Methods](#compression-methods)
    - [Sparsity - Pruning and Regularization](#sparsity---pruning-and-regularization)
    - [Quantization](#quantization)
    - [Knowledge Distillation](#knowledge-distillation)
  - [Models Supported](#models-supported)
  - [Datasets Supported](#datasets-supported)
  - [Re-usable Image Classification Code](#re-usable-image-classification-code)

## Usage

Please see the [docs](https://nervanasystems.github.io/distiller/usage.html) for usage details. In addition, run `compress_classifier.py -h` to show the extensive list of command-line options available.

## Compression Methods

**Follow the links for more details on each method and experiment results.**

### Sparsity - Pruning and Regularization

A non-exhaustive list of the methods implemented:

- [AGP](https://github.com/NervanaSystems/distiller/tree/master/examples/agp-pruning)
- [DropFilter](https://github.com/NervanaSystems/distiller/tree/master/examples/drop_filter)
- [Lottery-Ticket Hypothesis](https://github.com/NervanaSystems/distiller/tree/master/examples/lottery_ticket)
- [Network Surgery](https://github.com/NervanaSystems/distiller/tree/master/examples/network_surgery)
- [Network Trimming](https://github.com/NervanaSystems/distiller/tree/master/examples/network_trimming)
- [Hybrids](https://github.com/NervanaSystems/distiller/tree/master/examples/hybrid): These are examples where multiple pruning strategies are combined.

### Quantization

- [Post-training quantization](https://github.com/NervanaSystems/distiller/tree/master/examples/quantization/post_train_quant/command_line.md) based on the TensorFlow quantization scheme (originally GEMMLOWP) with additional capabilities.
  - Optimizing post-training quantization parameters with the [LAPQ](https://arxiv.org/abs/1911.07190) method - see [example YAML](https://github.com/NervanaSystems/distiller/blob/master/examples/quantization/post_train_quant/resnet18_imagenet_post_train_lapq.yaml) file for details.
- [Quantization-aware training](https://github.com/NervanaSystems/distiller/tree/master/examples/quantization/quant_aware_train): TensorFlow scheme, DoReFa, PACT

### Knowledge Distillation

See details in the [docs](https://nervanasystems.github.io/distiller/schedule.html#knowledge-distillation), and these YAML schedules training ResNet on CIFAR-10 with knowledge distillation: [FP32](https://github.com/NervanaSystems/distiller/tree/master/examples/quantization/fp32_baselines/preact_resnet_cifar_base_fp32.yaml) ; [DoReFa](https://github.com/NervanaSystems/distiller/tree/master/examples/quantization/quant_aware_train/preact_resnet_cifar_dorefa.yaml).

## Models Supported

The sample app integrates with [TorchVision](https://pytorch.org/docs/master/torchvision/models.html#classification) and [Cadene's pre-trained models](https://github.com/Cadene/pretrained-models.pytorch). Barring specific issues, any model from these two repositories can be specified from the command line and used.

We've implemented additional models, which can be found [here](https://github.com/NervanaSystems/distiller/tree/master/distiller/models).

## Datasets Supported

The application supports ImageNet, CIFAR-10 and MNIST.

The `compress_classifier.py` application will download the CIFAR-10 and MNIST datasets automatically the first time you try to use them (thanks to TorchVision).  The example invocations used  throughout Distiller's documentation assume that you have downloaded the images to directory `distiller/../data.cifar10`, but you can place the images anywhere you want (you tell `compress_classifier.py` where the dataset is located - or where you want the application to download the dataset to - using a command-line parameter).

ImageNet needs to be [downloaded](http://image-net.org/download) manually, due to copyright issues.  Facebook has created a [set of scripts](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) to help download and extract the dataset.

Again, the Distiller documentation assumes the following directory structure for the datasets, but this is just a suggestion:
```
distiller
  examples
    classifier_compression
data.imagenet/
    train/
    val/
data.cifar10/
    cifar-10-batches-py/
        batches.meta
        data_batch_1
        data_batch_2
        data_batch_3
        data_batch_4
        data_batch_5
        readme.html
        test_batch
data.mnist/
    MNIST/
        processed/
        raw/
```

## Re-usable Image Classification Code

We borrow the main flow code from PyTorch's ImageNet classification training sample application ([see here](https://github.com/pytorch/examples/tree/master/imagenet)). Much of the flow was refactored into a class called `ClassifierCompressor`, which can be re-used to build different scripts that perform image classifiers compression. Its implementation can be found in [`distiller/apputils/image_classifier.py`](https://github.com/NervanaSystems/distiller/tree/master/distiller/apputils/image_classifier.py).  

The [AMC auto-compression](https://github.com/NervanaSystems/distiller/tree/master/examples/auto_compression/amc) sample is another application that uses this building block.

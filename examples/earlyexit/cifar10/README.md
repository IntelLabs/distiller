*Note that this section of the Early-Exit Repository started with the CIFAR10/Resnet code from 
Yerlan Idelbayev. It has been modified to illustrate Early Exit.*

What follows below is the README.md file from Yerlan's repository. Note however, that small modifications were made to the code to include an early exit and collect statistcs from these runs,

# Proper ResNet Implementation for CIFAR10/CIFAR100 in pytorch
[Torchvision model zoo](https://github.com/pytorch/vision/tree/master/torchvision/models) provides number of implementations of various state-of-the-art architectures, however, most of them are defined and implemented for ImageNet.
Usually it is very straightforward to use them on other datasets, but sometimes this models needs manual setup.

Unfortunately, none of the pytorch repositories with ResNets on CIFAR10 provides an implementation as described in  [original paper](https://arxiv.org/abs/1512.03385). If you just use torchvision's models on CIFAR10 you'll get the model **that differs in number of layers and parameters**. That is unacceptable if you want to directly compare ResNets on CIFAR10.
The purpose of this repo is to provide a valid pytorch implementation of ResNet-s for CIFAR10. Following models are provided:

| Name      | # layers | # params| Test err(paper) | Test err(this impl.)|
|-----------|---------:|--------:|:-----------------:|:---------------------:|
|[ResNet20](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20.th)   |    20    | 0.27M   | 8.75%| **8.27%**|
|[ResNet32](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32.th)  |    32    | 0.46M   | 7.51%| **7.37%**|
|[ResNet44](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet44.th)   |    44    | 0.66M   | 7.17%| **6.90%**|
|[ResNet56](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56.th)   |    56    | 0.85M   | 6.97%| **6.61%**|
|[ResNet110](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110.th)  |   110    |  1.7M   | 6.43%| **6.32%**|
|[ResNet1202](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202.th) |  1202    | 19.4M   | 7.93%| **6.18%**|

And their implementation matches description in original paper, with comparable or better test error.

## How to run?
```bash
git clone https://github.com/akamaster/pytorch_resnet_cifar10
cd pytorch_resnet_cifar10
chmod +x run.sh && ./run.sh
```

## Details of training
This implementation follows paper in straightforward manner with some caveats. **Firstly**, original paper uses 45k/5k train/validation split to train data, and selects best performing model based on performance on validation set. This implementation does not do any validation testing, so if you need to compare your results on ResNet head-to-head to orginal paper's keep this in mind. **Secondly**, if you want to train ResNet1202 keep in mind that you need 16GB memory on GPU.

## Pretrained models for download
1. [ResNet20, 8.27% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet20.th)
2. [ResNet32, 7.37% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet32.th)
3. [ResNet44, 6.90% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet44.th)
4. [ResNet56, 6.61% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet56.th)
5. [ResNet110, 6.32% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet110.th)
6. [ResNet1202, 6.18% err](https://github.com/akamaster/pytorch_resnet_cifar10/raw/master/pretrained_models/resnet1202.th)

If you find this implementation is useful and used it in your production/academic work please cite/mention this page and author Yerlan Idelbayev.

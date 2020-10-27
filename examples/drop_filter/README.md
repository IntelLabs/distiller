## DropFilter
DropFilter - a regularization method similar to Dropout, which drops entire convolutional
filters, instead of mere neurons.
However, unlike the original intent of DropFilter - to act as a regularizer and reduce the generalization error
of the network, here we employ higher rates of filter-dropping (rates are increased over time by following an AGP
schedule) in order to make the network more robust to filter-pruning.  We test this robustness using sensitivity
analysis.


A relevant quote from [3]:
> To train slimmable neural networks, we begin with a naive approach, where we directly train a
shared neural network with different width configurations. The training framework is similar to the
one of our final approach, as shown in Algorithm 1. The training is stable, however, the network
obtains extremely low top-1 testing accuracy around 0:1% on 1000-class ImageNet classification.
Error curves of the naive approach are shown in Figure 2. We conjecture the major problem in
the naive approach is that: for a single channel in a layer, different numbers of input channels in
previous layer result in different means and variances of the aggregated feature, which are then
rolling averaged to a shared batch normalization layer. The inconsistency leads to inaccurate batch
normalization statistics in a layer-by-layer propagating manner. Note that these batch normalization
statistics (moving averaged means and variances) are only used during testing, in training the means
and variances of the current mini-batch are used.

### Examples

Dropping filters requires finer control over the scheduling process since we want to drop different sets
of filters every `n` training iterations/steps (i.e. mini-batches), whereas usually we make such decisions
at the epoch boundary. 

1. [plain20_cifar_dropfilter_training.yaml](https://github.com/IntelLabs/distiller/blob/master/examples/drop_filter/plain20_cifar_dropfilter_training.yaml)

    In this example we train a Plain20 model with increasing levels of dropped filters, 
    starting at 5% drop and going as high as 50% drop.  Our aim is to make the network more robust to filter-pruning<br>
    * The network is trained from scratch.
    * We spend a few epochs just training, to start from weights that are somewhat trained.
    * We use AGP to control the schedule of the slow increase in the percentage of dropped filters.
    * To choose which filters to drop we use a Bernoulli probability function.

    | Model | Drop Rate  | Top1  | Baseline Top1
    | --- | :---: |    ---: |  ---: |
    | Plain-20 | 5-50%| 89.61 | 90.18
    
2. [plain20_cifar_dropfilter_training_regularization.yaml](https://github.com/IntelLabs/distiller/blob/master/examples/drop_filter/plain20_cifar_dropfilter_training_regularization.yaml)
    
   In this example we use DropFilter for regularization.  
   * The network is trained from scratch.
   * To choose which filters to drop we use a Bernoulli probability function.
   
    | Model | Drop Rate  | Top1  | Baseline Top1
    | --- | :---: |    ---: |  ---: |
    | Plain-20 | 10%| 90.88 | 90.18
    
3. [resnet20_cifar_randomlevel_training.yaml](https://github.com/IntelLabs/distiller/blob/master/examples/drop_filter/resnet20_cifar_randomlevel_training.yaml)
    
    In this example we randomly choose a percentage of filters to prune (level), and then use L1-norm ranking to 
    choose which filters to drop.

    | Model | Drop Rate  | Top1  | Baseline Top1
    | --- | :---: |    ---: |  ---: |
    | ResNet-20 | 10-20%| 90.80 | 90.78

### References:
[1] Zhengsu Chen Jianwei Niu Qi Tian.
    DropFilter: Dropout for Convolutions.    
    https://arxiv.org/abs/1810.09849
     
[2] Hengyue Pan, Hui Jiang, Xin Niu, Yong Dou.
    DropFilter: A Novel Regularization Method for Learning Convolutional Neural Networks
    <br>https://arxiv.org/abs/1811.06783
     
[3] Jiahui Yu, Linjie Yang, Ning Xu, Jianchao Yang, Thomas Huang. 
Slimmable Neural Networks.  In ICLR 2019, arXiv:1812.08928
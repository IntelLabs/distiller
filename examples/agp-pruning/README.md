## Automated Gradual Pruner (AGP) Pruning Examples

### Introduction
In [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878),
authors Michael Zhu and Suyog Gupta provide an algorithm to schedule iterative level pruning.

> We introduce a new automated gradual pruning algorithm in which the sparsity is increased from an initial sparsity value (usually 0) to a ﬁnal sparsity value over a span of n pruning steps.
The intuition behind this sparsity function in equation (1)  is to prune the network rapidly in the initial phase when the redundant connections are
abundant and gradually reduce the number of weights being pruned each time as there are fewer and fewer weights remaining in the network.

The authors describe AGP:
>
- Our automated gradual pruning algorithm prunes the smallest magnitude weights to achieve a preset level of network sparsity.
-  Doesn't require much hyper-parameter tuning
- Shown to perform well across different models
- Does not make any assumptions about the structure of the network or its constituent layers, and is therefore more generally applicable.

### Distiller 
* The original AGP paper described the application of AGP for fine-grained pruning, and in Distiller we also implemented AGP for structured-pruning.
* We also provide examples of applying AGP for pruning language models. The results and 
methodology are discussed at length in the [documentation](https://intellabs.github.io/distiller/tutorial-lang_model.html)

### Examples

The tables below provide the results of the experimental pruning schedules that
appear in this directory.  Each example YAML schedule-file contains the command-line
used to execute the experiment, and further details.

#### Element-wise sparsity
| Model | Granularity | Sparsity (%) | Top1  | Baseline Top1
| --- |  :--- |  ---: |  ---: |  ---: |
| AlexNet | Fine | 88.3 | 56.528 | 56.55
| MobileNet v1 (width=1)| Fine | 51.6 | 68.8 | 68.9
| ResNeXt-101-32x4d| Fine | 75.0 | 78.66 | 78.19
| ResNet-18 | Fine | 59.9 | 69.87 | 69.76 
| ResNet-50 | Fine | 26 .0 | 76.54 | 76.15
| ResNet-50 | Fine | 80.0 | 75.99 | 76.15
| ResNet-50 | Fine | 84.6 | 75.66 | 76.15

#### Block sparsity
| Model | Granularity | Sparsity | Top1  | Baseline Top1
| --- |  :--- |  ---: |  ---: |  ---: |
| ResNet-50 | 1x1x8 | 36.7 | 76.36 | 76.15

#### Filter pruning with thinning

Our objective here is to minimize compute but performing thinning.  Therefore,
sparsity is often at 0%, but the number of parameters is reduced as
filters are removed.

In this table we seek to see a <b>lower</b> value for `Parameters Kept (%)` and, more importantely, 
`Compute Kept (%)`.

| Model | Granularity | Sparsity (%) | Parameters Kept (%) | Compute Kept (%)| Top1 | Baseline Top1
| --- |  :--- |  ---: |  ---: |  ---: | ---: |  ---: |
| ResNet-50 | Filters| 0.0 | 43.37 | 44.56 | 74.47 | 76.15
| ResNet-50 (2) | Filters| 0.0 | 49.69 | 49.82 | 74.78 | 76.15
| ResNet-50 (3) | Filters| 0.0 | 67.95 | 67.33 | 75.75 | 76.15
| ResNet-50 (w/ FC) | Filters| 11.6 | 42.74 | 44.56 | 74.56 | 76.15
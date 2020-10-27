## Hybrid-Pruning Schedules

The examples in this directory show hybrid pruning schedules in which we combine several different pruning strategies.

1. [alexnet.schedule_agp_2Dreg.yaml](https://github.com/IntelLabs/distiller/blob/master/examples/hybrid/alexnet.schedule_agp_2Dreg.yaml)
<br>
This example presents a pruning-schedule that performs element-wise (fine grain) pruning, 
with 2D group (kernel) regularization.  The regularization "pushes" 2D kernels towards zero, while
the pruning attends to individual weights coefficients.  The pruning schedule is driven by AGP.

2. [alexnet.schedule_sensitivity_2D-reg.yaml](https://github.com/IntelLabs/distiller/blob/master/examples/hybrid/alexnet.schedule_sensitivity_2D-reg.yaml)
<br>
This example also presents a pruning-schedule that performs element-wise (fine grain) pruning, 
with 2D group (kernel) regularization.  However, the pruner is a `Distiller.pruning.SensitivityPruner` which is
driven by the tensors' [sensitivity](https://intellabs.github.io/distiller/algo_pruning.html#sensitivity-pruner), instead of AGP.


|Experiment| Model | Sparsity  | Top1  | Baseline Top1
| :---: | --- | :---: |    ---: |  ---: |
|1| Alexnet | 88.31| 56.40 | 56.55
|2| Alexnet | 88.31| 56.24 | 56.55
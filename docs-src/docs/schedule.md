# Compression scheduler
In iterative pruning, we create some kind of pruning regimen that specifies how to prune, and what to prune at every stage of the pruning and training stages. This motivated the design of ```CompressionScheduler```: it needed to be part of the training loop, and to be able to make and implement pruning, regularization and (later) quantization decisions.  We wanted to be able to change the particulars of the compression schedule, w/o touching the code, and settled on using YAML as a container for this specification.  We found that when we make many experiments on the same code base, it is easier to maintain all of these experiments if we decouple the differences from the code-base.  Therefore, we added to the scheduler support for learning-rate decay scheduling because, again, we wanted the freedom to change the LR-decay policy without changing code.  

## High level overview
Let's briefly discuss the main mechanisms and abstractions: A schedule specification is composed of a list of sections defining instances of Pruners, Regularizers, LR-scheduler and Policies.

  -  Pruners and Regularizers are very similar: they implement either a Pruning algorithm or a Regularization algorithm.  
  - An LR-scheduler specifies the LR-decay algorithm.  

These define the **what** part of the schedule.  

The Policies define the **when** part of the schedule: at which epoch to start applying the Pruner/Regularizer/LR-decay, the epoch to end, and how often to invoke the policy (frequency of application).  A policy also defines the instance of Pruner/Regularizer/LR-decay it is managing.
<br>
The CompressionScheduler is configured from a YAML file or from a dictionary, but you can also manually create Policies, Pruners and Regularizers from code.

## Syntax through example
We'll use ```alexnet.schedule_agp.yaml``` to explain some of the YAML syntax for configuring Sensitivity Pruning of Alexnet.
```
version: 1
pruners:
  my_pruner:
    class: 'SensitivityPruner'
    sensitivities:
      'features.module.0.weight': 0.25
      'features.module.3.weight': 0.35
      'features.module.6.weight': 0.40
      'features.module.8.weight': 0.45
      'features.module.10.weight': 0.55
      'classifier.1.weight': 0.875
      'classifier.4.weight': 0.875
      'classifier.6.weight': 0.625

lr_schedulers:
   pruning_lr:
     class: ExponentialLR
     gamma: 0.9

policies:
  - pruner:
      instance_name : 'my_pruner'
    starting_epoch: 0
    ending_epoch: 38
    frequency: 2

  - lr_scheduler:
      instance_name: pruning_lr
    starting_epoch: 24
    ending_epoch: 200
    frequency: 1
```

There is only one version of the YAML syntax, and the version number is not verified at the moment.  However, to be future-proof it is probably better to let the YAML parser know that you are using version-1 syntax, in case there is ever a version 2.
```
version: 1
```
In the ```pruners``` section, we define the instances of pruners we want the scheduler to instantiate and use.<br>
We define a single pruner instance, named ```my_pruner``` of algorithm ```SensitivityPruner```.  We will refer to this instance in the ```Policies``` section.<br>
Then we list the sensitivity multipliers, \\(s\\), of each of the weight tensors.<br>
You may list as many Pruners as you want in this section, as long as each has a unique name.  You can several types of pruners in one schedule.

```
pruners:
  my_pruner:
    class: 'SensitivityPruner'
    sensitivities:
      'features.module.0.weight': 0.25
      'features.module.3.weight': 0.35
      'features.module.6.weight': 0.40
      'features.module.8.weight': 0.45
      'features.module.10.weight': 0.55
      'classifier.1.weight': 0.875
      'classifier.4.weight': 0.875
      'classifier.6.weight': 0.6
```

Next, we want to specify the learning-rate decay scheduling in the ```lr_schedulers``` section.  We assign a name to this instance: ```pruning_lr```.  As in the ```pruners``` section, you may use any name, as long as all LR-schedulers have a unique name.  At the moment, only one instance of LR-scheduler is allowed.  You can use any LR-scheduler class that ```torch.optim.lr_scheduler``` supports and pass their arguments.  The keyword arguments (kwargs) are passed directly to the constructor of the subclasses of [_LRScheduler](http://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html), so that as new LR-schedulers are added to ```torch.optim.lr_scheduler```, they can be used without changing the application code.

```
lr_schedulers:
   pruning_lr:
     class: ExponentialLR
     gamma: 0.9
```   

Finally, we define the ```policies``` section which defines the actual scheduling.  A ```Policy``` manages an instance of a ```Pruner```, ```Regularizer```, or ```LRSchedule```, by naming the instance.  In the example below, a ```PruningPolicy``` uses the pruner instance named ```my_pruner```: it activates it at a frequency of 2 epochs (i.e. every other epoch), starting at epoch 0, and ending at epoch 38.  
```
policies:
  - pruner:
      instance_name : 'my_pruner'
    starting_epoch: 0
    ending_epoch: 38
    frequency: 2

  - lr_scheduler:
      instance_name: pruning_lr
    starting_epoch: 24
    ending_epoch: 200
    frequency: 1
```
This is *iterative pruning*:

1. Train Connectivity

2. Prune Connections

3. Retrain Weights

4. Goto 2

It is described  in [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626):
> "Our method prunes redundant connections using a three-step method. First, we train the network to learn which connections are important. Next, we prune the unimportant connections. Finally, we retrain the network to fine tune the weights of the remaining connections...After an initial training phase, we remove all connections whose weight is lower than a threshold. This pruning converts a dense, fully-connected layer to a sparse layer. This first phase learns the topology of the networks — learning which connections are important and removing the unimportant connections. We then retrain the sparse network so the remaining connections can compensate for the connections that have been removed. The phases of pruning and retraining may be repeated iteratively to further reduce network complexity."

## Regularization
You can also define and schedule regularization.

### L1 regularization
Format (this is an informal specification, not a valid [ABNF](https://en.wikipedia.org/wiki/Augmented_Backus%E2%80%93Naur_form) specification):
```
regularizers:
  <REGULARIZER_NAME_STR>:
    class: L1Regularizer
    reg_regims:
      <PYTORCH_PARAM_NAME_STR>: <STRENGTH_FLOAT>
      ...
      <PYTORCH_PARAM_NAME_STR>: <STRENGTH_FLOAT>
    threshold_criteria: [Mean_Abs | Max]
```

For example:
```
version: 1

regularizers:
  my_L1_reg:
    class: L1Regularizer
    reg_regims:
      'module.layer3.1.conv1.weight': 0.000002
      'module.layer3.1.conv2.weight': 0.000002
      'module.layer3.1.conv3.weight': 0.000002
      'module.layer3.2.conv1.weight': 0.000002
    threshold_criteria: Mean_Abs

policies:
  - regularizer:
      instance_name: my_L1_reg
    starting_epoch: 0
    ending_epoch: 60
    frequency: 1
```

### Group regularization
Format (informal specification):
```
Format:
  regularizers:
    <REGULARIZER_NAME_STR>:
      class: L1Regularizer
      reg_regims:
        <PYTORCH_PARAM_NAME_STR>: [<STRENGTH_FLOAT>, <'2D' | '3D' | '4D' | 'Channels' | 'Cols' | 'Rows'>]
        <PYTORCH_PARAM_NAME_STR>: [<STRENGTH_FLOAT>, <'2D' | '3D' | '4D' | 'Channels' | 'Cols' | 'Rows'>]
      threshold_criteria: [Mean_Abs | Max]
```

For example:
```
version: 1

regularizers:
  my_filter_regularizer:
    class: GroupLassoRegularizer
    reg_regims:
      'module.layer3.1.conv1.weight': [0.00005, '3D']
      'module.layer3.1.conv2.weight': [0.00005, '3D']
      'module.layer3.1.conv3.weight': [0.00005, '3D']
      'module.layer3.2.conv1.weight': [0.00005, '3D']
    threshold_criteria: Mean_Abs

policies:
  - regularizer:
      instance_name: my_filter_regularizer
    starting_epoch: 0
    ending_epoch: 60
    frequency: 1
```

## Mixing it up
You can mix pruning and regularization.

```
version: 1
pruners:
  my_pruner:
    class: 'SensitivityPruner'
    sensitivities:
      'features.module.0.weight': 0.25
      'features.module.3.weight': 0.35
      'features.module.6.weight': 0.40
      'features.module.8.weight': 0.45
      'features.module.10.weight': 0.55
      'classifier.1.weight': 0.875
      'classifier.4.weight': 0.875
      'classifier.6.weight': 0.625

regularizers:
  2d_groups_regularizer:
    class: GroupLassoRegularizer
    reg_regims:
      'features.module.0.weight': [0.000012, '2D']
      'features.module.3.weight': [0.000012, '2D']
      'features.module.6.weight': [0.000012, '2D']
      'features.module.8.weight': [0.000012, '2D']
      'features.module.10.weight': [0.000012, '2D']


lr_schedulers:
  # Learning rate decay scheduler
   pruning_lr:
     class: ExponentialLR
     gamma: 0.9

policies:
  - pruner:
      instance_name : 'my_pruner'
    starting_epoch: 0
    ending_epoch: 38
    frequency: 2

  - regularizer:
      instance_name: '2d_groups_regularizer'
    starting_epoch: 0
    ending_epoch: 38
    frequency: 1

  - lr_scheduler:
      instance_name: pruning_lr
    starting_epoch: 24
    ending_epoch: 200
    frequency: 1

```

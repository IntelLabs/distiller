# Compression scheduler
In iterative pruning, we create some kind of pruning regimen that specifies how to prune, and what to prune at every stage of the pruning and training stages. This motivated the design of ```CompressionScheduler```: it needed to be part of the training loop, and to be able to make and implement pruning, regularization and quantization decisions.  We wanted to be able to change the particulars of the compression schedule, w/o touching the code, and settled on using YAML as a container for this specification.  We found that when we make many experiments on the same code base, it is easier to maintain all of these experiments if we decouple the differences from the code-base.  Therefore, we added to the scheduler support for learning-rate decay scheduling because, again, we wanted the freedom to change the LR-decay policy without changing code.  

## High level overview
Let's briefly discuss the main mechanisms and abstractions: A schedule specification is composed of a list of sections defining instances of Pruners, Regularizers, Quantizers, LR-scheduler and Policies.

  - Pruners, Regularizers and Quantizers are very similar: They implement either a Pruning/Regularization/Quantization algorithm, respectively. 
  - An LR-scheduler specifies the LR-decay algorithm.  

These define the **what** part of the schedule.  

The Policies define the **when** part of the schedule: at which epoch to start applying the Pruner/Regularizer/Quantizer/LR-decay, the epoch to end, and how often to invoke the policy (frequency of application).  A policy also defines the instance of Pruner/Regularizer/Quantizer/LR-decay it is managing.  
The `CompressionScheduler` is configured from a YAML file or from a dictionary, but you can also manually create Policies, Pruners, Regularizers and Quantizers from code.

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
In the ```pruners``` section, we define the instances of pruners we want the scheduler to instantiate and use.  
We define a single pruner instance, named ```my_pruner```, of algorithm ```SensitivityPruner```.  We will refer to this instance in the ```Policies``` section.  
Then we list the sensitivity multipliers, \\(s\\), of each of the weight tensors.  
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

Next, we want to specify the learning-rate decay scheduling in the ```lr_schedulers``` section.  We assign a name to this instance: ```pruning_lr```.  As in the ```pruners``` section, you may use any name, as long as all LR-schedulers have a unique name.  At the moment, only one instance of LR-scheduler is allowed.  The LR-scheduler must be a subclass of PyTorch's [\_LRScheduler](http://pytorch.org/docs/master/_modules/torch/optim/lr_scheduler.html).  You can use any of the schedulers defined in ```torch.optim.lr_scheduler``` (see [here](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)).  In addition, we've implemented some additional schedulers in Distiller (see [here](https://github.com/NervanaSystems/distiller/blob/master/distiller/learning_rate.py)). The keyword arguments (kwargs) are passed directly to the LR-scheduler's constructor, so that as new LR-schedulers are added to ```torch.optim.lr_scheduler```, they can be used without changing the application code.

```
lr_schedulers:
   pruning_lr:
     class: ExponentialLR
     gamma: 0.9
```   

Finally, we define the ```policies``` section which defines the actual scheduling.  A ```Policy``` manages an instance of a ```Pruner```, ```Regularizer```, `Quantizer`, or ```LRScheduler```, by naming the instance.  In the example below, a ```PruningPolicy``` uses the pruner instance named ```my_pruner```: it activates it at a frequency of 2 epochs (i.e. every other epoch), starting at epoch 0, and ending at epoch 38.  
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

## Quantization-Aware Training

Similarly to pruners and regularizers, specifying a quantizer in the scheduler YAML follows the constructor arguments of the `Quantizer` class (see details [here](design.md#quantization)). **Note** that only a single quantizer instance may be defined per YAML.

Let's see an example:

```
quantizers:
  dorefa_quantizer:
    class: DorefaQuantizer
    bits_activations: 8
    bits_weights: 4
    bits_overrides:
      conv1:
        wts: null
        acts: null
      relu1:
        wts: null
        acts: null
      final_relu:
        wts: null
        acts: null
      fc:
        wts: null
        acts: null
```

- The specific quantization method we're instantiating here is `DorefaQuantizer`.
- Then we define the default bit-widths for activations and weights, in this case 8 and 4-bits, respectively. 
- Then, we define the `bits_overrides` mapping. In the example above, we choose not to quantize the first and last layer of the model. In the case of `DorefaQuantizer`, the weights are quantized as part of the convolution / FC layers, but the activations are quantized in separate layers, which replace the ReLU layers in the original model (remember - even though we replaced the ReLU modules with our own quantization modules, the name of the modules isn't changed). So, in all, we need to reference the first layer with parameters `conv1`, the first activation layer `relu1`, the last activation layer `final_relu` and the last layer with parameters `fc`.
- Specifying `null` means "do not quantize".
- Note that for quantizers, we reference names of modules, not names of parameters as we do for pruners and regularizers.

### Defining overrides for **groups of layers** using regular expressions

Suppose we have a sub-module in our model named `block1`, which contains multiple convolution layers which we would like to quantize to, say, 2-bits. The convolution layers are named `conv1`, `conv2` and so on. In that case we would define the following:

```
bits_overrides:
  'block1\.conv*':
    wts: 2
    acts: null
```

- **RegEx Note**: Remember that the dot (`.`) is a meta-character (i.e. a reserved character) in regular expressions. So, to match the actual dot characters which separate sub-modules in PyTorch module names, we need to escape it: `\.`
    
**Overlapping patterns** are also possible, which allows to define some override for a groups of layers and also "single-out" specific layers for different overrides. For example, let's take the last example and configure a different override for `block1.conv1`:

```
bits_overrides:
  'block1\.conv1':
    wts: 4
    acts: null
  'block1\.conv*':
    wts: 2
    acts: null
```

- **Important Note**: The patterns are evaluated eagerly - first match wins. So, to properly quantize a model using "broad" patterns and more "specific" patterns as just shown, make sure the specific pattern is listed **before** the broad one.


The `QuantizationPolicy`, which controls the quantization procedure during training, is actually quite simplistic. All it does is call the `prepare_model()` function of the `Quantizer` when it's initialized, followed by the first call to `quantize_params()`. Then, at the end of each epoch, after the float copy of the weights has been updated, it calls the `quantize_params()` function again.

```
policies:
  - quantizer:
      instance_name: dorefa_quantizer
      starting_epoch: 0
      ending_epoch: 200
      frequency: 1
```

**Important Note**: As mentioned [here](design.md#quantization-aware-training), since the quantizer modifies the model's parameters (assuming training with quantization in the loop is used), the call to `prepare_model()` must be performed before an optimizer is called. Therefore, currently, the starting epoch for a quantization policy must be 0, otherwise the quantization process will not work as expected. If one wishes to do a "warm-startup" (or "boot-strapping"), training for a few epochs with full precision and only then starting to quantize, the only way to do this right now is to execute a separate run to generate the boot-strapped weights, and execute a second which will resume the checkpoint with the boot-strapped weights.

## Post-Training Quantization

Post-training quantization differs from the other techniques described here. Since it is not executed during training, it does not require any Policies nor a Scheduler. Currently, the only method implemented for pots-training quantization is [range-based linear quantization](algo_quantization.md#range-based-linear-quantization). Quantizing a model using this method, requires adding 2 lines of code:

```python
quantizer = distiller.quantization.PostTrainLinearQuantizer(model, <quantizer arguments>)
quantizer.prepare_model()
# Execute evaluation on model as usual
```

See the documentation for `PostTrainLinearQuantizer` in [range_linear.py](https://github.com/NervanaSystems/distiller/blob/master/distiller/quantization/range_linear.py) for details on the available arguments.  
In addition to directly instantiating the quantizer with arguments, it can also be configured from a YAML file. The syntax for the YAML file is exactly the same as seen in the quantization-aware training section above. Not surprisingly, the `class` defined must be `PostTrainLinearQuantizer`, and any other components or policies defined in the YAML file are ignored. We'll see how to create the quantizer in this manner below.

If more configurability is needed, a helper function can be used that will add a set of command-line arguments to configure the quantizer:

```python
parser = argparse.ArgumentParser()
distiller.quantization.add_post_train_quant_args(parser)
args = parser.parse_args()
```

These are the available command line arguments:

```
Arguments controlling quantization at evaluation time("post-training quantization"):
  --quantize-eval, --qe
                        Apply linear quantization to model before evaluation.
                        Applicable only if --evaluate is also set
  --qe-calibration NUM_CALIBRATION_BATCHES
                        Run the model in evaluation mode for the specified
                        number of batches and collect statistics. Ignores all
                        other 'qe--*' arguments
  --qe-mode QE_MODE, --qem QE_MODE
                        Linear quantization mode. Choices: sym | asym_s |
                        asym_u
  --qe-bits-acts NUM_BITS, --qeba NUM_BITS
                        Number of bits for quantization of activations
  --qe-bits-wts NUM_BITS, --qebw NUM_BITS
                        Number of bits for quantization of weights
  --qe-bits-accum NUM_BITS
                        Number of bits for quantization of the accumulator
  --qe-clip-acts, --qeca
                        Enable clipping of activations using min/max values
                        averaging over batch
  --qe-no-clip-layers LAYER_NAME [LAYER_NAME ...], --qencl LAYER_NAME [LAYER_NAME ...]
                        List of layer names for which not to clip activations.
                        Applicable only if --qe-clip-acts is also set
  --qe-per-channel, --qepc
                        Enable per-channel quantization of weights (per output
                        channel)
  --qe-stats-file PATH  Path to YAML file with calibration stats. If not
                        given, dynamic quantization will be run (Note that not
                        all layer types are supported for dynamic
                        quantization)
  --qe-config-file PATH
                        Path to YAML file containing configuration for
                        PostTrainLinearQuantizer (if present, all other --qe*
                        arguments are ignored)
```

(Note that `--quantize-eval` and `--qe-calibration` are mutually exclusive.)

When using these command line arguments, the quantizer can be invoked as follows:

```python
if args.quantize_eval:
    if args.qe_config_file:
        quantizer = distiller.config_component_from_file_by_class(model, args.qe_config_file,
                                                                  'PostTrainLinearQuantizer')
    else:
        quantizer = quantization.PostTrainLinearQuantizer(model, args.qe_bits_acts, args.qe_bits_wts,
                                                          args.qe_bits_accum, None, args.qe_mode, args.qe_clip_acts,
                                                          args.qe_no_clip_layers, args.qe_per_channel,
                                                          args.qe_stats_file)
    quantizer.prepare_model()
    # Execute evaluation on model as usual
```

Note that the command-line arguments don't expose the `bits_overrides` parameter of the quantizer, which allows fine-grained control over how each layer is quantized. To utilize this functionality, configure with a YAML file.

To see integration of these command line arguments in use, see the [image classification example](https://github.com/NervanaSystems/distiller/blob/master/examples/classifier_compression/compress_classifier.py). For examples invocations of post-training quantization see [here](https://github.com/NervanaSystems/distiller/blob/master/examples/quantization/post_training_quant).

### Collecting Statistics for Quantization

To collect generate statistics that can be used for static quantization of activations, do the following (shown here assuming the command line argument `--qe-calibration` shown above is used, which specifies the number of batches to use for statistics generation):

```python
if args.qe_calibration:
    distiller.utils.assign_layer_fq_names(model)
    msglogger.info("Generating quantization calibration stats based on {0} users".format(args.qe_calibration))
    collector = distiller.data_loggers.QuantCalibrationStatsCollector(model)
    with collector_context(collector):
        # Here call your model evaluation function, making sure to execute the
        # number of batches specified by the qe_calibration argument
    yaml_path = 'some/dir/quantization_stats.yaml'
    collector.save(yaml_path)
```

The genreated YAML stats file can then be provided using the ``--qe-stats-file` argument. An example of a generated stats file can be found [here](https://github.com/NervanaSystems/distiller/blob/master/examples/quantization/post_training_quant/stats/resnet18_quant_stats.yaml).

## Knowledge Distillation

Knowledge distillation (see [here](knowledge_distillation.md)) is also implemented as a `Policy`, which should be added to the scheduler. However, with the current implementation, it cannot be defined within the YAML file like the rest of the policies described above.

To make the integration of this method into applications a bit easier, a helper function can be used that will add a set of command-line arguments related to knowledge distillation:

```
import argparse
import distiller

parser = argparse.ArgumentParser()
distiller.knowledge_distillation.add_distillation_args(parser)
```

(The `add_distillation_args` function accepts some optional arguments, see its implementation at `distiller/knowledge_distillation.py` for details)

These are the command line arguments exposed by this function:

```
Knowledge Distillation Training Arguments:
  --kd-teacher ARCH     Model architecture for teacher model
  --kd-pretrained       Use pre-trained model for teacher
  --kd-resume PATH      Path to checkpoint from which to load teacher weights
  --kd-temperature TEMP, --kd-temp TEMP
                        Knowledge distillation softmax temperature
  --kd-distill-wt WEIGHT, --kd-dw WEIGHT
                        Weight for distillation loss (student vs. teacher soft
                        targets)
  --kd-student-wt WEIGHT, --kd-sw WEIGHT
                        Weight for student vs. labels loss
  --kd-teacher-wt WEIGHT, --kd-tw WEIGHT
                        Weight for teacher vs. labels loss
  --kd-start-epoch EPOCH_NUM
                        Epoch from which to enable distillation

```

Once arguments have been parsed, some initialization code is required, similar to the following:

```
# Assuming:
# "args" variable holds command line arguments
# "model" variable holds the model we're going to train, that is - the student model
# "compression_scheduler" variable holds a CompressionScheduler instance

args.kd_policy = None
if args.kd_teacher:
    # Create teacher model - replace this with your model creation code
    teacher = create_model(args.kd_pretrained, args.dataset, args.kd_teacher, device_ids=args.gpus)
    if args.kd_resume:
        teacher, _, _ = apputils.load_checkpoint(teacher, chkpt_file=args.kd_resume)

    # Create policy and add to scheduler
    dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
    args.kd_policy = distiller.KnowledgeDistillationPolicy(model, teacher, args.kd_temp, dlw)
    compression_scheduler.add_policy(args.kd_policy, starting_epoch=args.kd_start_epoch, ending_epoch=args.epochs,
                                     frequency=1)
```

Finally, during the training loop, we need to perform forward propagation through the teacher model as well. The `KnowledgeDistillationPolicy` class keeps a reference to both the student and teacher models, and exposes a `forward` function that performs forward propagation on both of them. Since this is not one of the standard policy callbacks, we need to call this function manually from our training loop, as follows:

```
if args.kd_policy is None:
    # Revert to a "normal" forward-prop call if no knowledge distillation policy is present
    output = model(input_var)
else:
    output = args.kd_policy.forward(input_var)
```

To see this integration in action, take a look at the image classification sample at `examples/classifier_compression/compress_classifier.py`.

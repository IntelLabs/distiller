# Post-Training Quantization Command Line Examples

Following are a few examples of invoking post-training quantization on ResNet-50, using Distiller's image classification sample.  

### Command Line vs. YAML Configuration

Post-training quantization can either be configured straight from the command-line, or using a YAML configuration file. Using a YAML file allows fine-grained control over the quantization parameters of each layer, whereas command-line configuration is "monolithic". To apply a YAML configuration file, use the `--qe-config-file` argument.   

**All the examples shown below are using command-line configuration.**  

**For an example of how to use a YAML config file, please see `resnet18_imagenet_post_train.yaml` in this directory ([link to view in GitHub repo](https://github.com/NervanaSystems/distiller/blob/master/examples/quantization/post_train_quant/resnet18_imagenet_post_train.yaml)). It shows how to override the configuration of specific layers in order to obtain better accuracy.**  

## Available Command Line Arguments

| Long Form                | Short     | Description                                                                           | Default |
|--------------------------|-----------|---------------------------------------------------------------------------------------|---------|
| `--quantize-eval`        | `--qe`    | Apply linear quantization to model before evaluation                                  | Off     |
| `--qe-mode`              | `--qem`   | Default linear quantization mode (for weights and activations). Choices: "sym", "asym_u", "asym_s"      | "sym"   |
| `--qe-mode-acts`         | `--qema`  | Linear quantization mode for activations. **Overrides `--qe-mode`**. Choices: "sym", "asym_u", "asym_s" | None    |
| `--qe-mode-wts`          | `--qemw`  | Linear quantization mode for weights. **Overrides `--qe-mode`**. Choices: "sym", "asym_u", "asym_s"     | None    |
| `--qe-bits-acts`         | `--qeba`  | # of bits for quantization of activations. Use 0 to not quantize activations          | 8       |
| `--qe-bits-wts`          | `--qebw`  | # of bits for quantization of weights. Use 0 to not quantize weights                  | 8       |
| `--qe-bits-accum`        | N/A       | # of bits for quantization of the accumulator                                         | 32      |
| `--qe-clip-acts`         | `--qeca`  | Set activations clipping mode. Choices: "none", "avg", "n_std", "gauss", "laplace"    | "none"  |
| `--qe-clip-n-stds`       | N/A       | When qe-clip-acts is set to 'n_std', this is the number of standard deviations to use | None    |
| `--qe-no-clip-layers`    | `--qencl` | List of layer names (space-separated) for which not to clip activations               | ''      |
| `--qe-no-quant-layers`   | `--qenql` | List of layer names (space-separated) for which not to skip quantization              | ''      |
| `--qe-per-channel`       | `--qepc`  | Enable per-channel quantization of weights (per output channel)                       | Off     |
| `--qe-scale-approx-bits` | `--qesab` | Enables scale factor approximation using integer multiply + bit shift, using this number of bits the integer multiplier | None |
| `--qe-stats-file`        | N/A       | Use stats file for static quantization of activations. See details below              | None    |
| `--qe-dynamic`           | N/A       | Perform dynamic quantization. See details below                                       | None    |
| `--qe-config-file`       | N/A       | Path to YAML config file. See section above. (ignores all other --qe* arguments)      | None    |
| `--qe-convert-pytorch`   | `--qept`  | Convert the model to PyTorch native post-train quantization modules. See [tutorial](https://github.com/NervanaSystems/distiller/blob/master/jupyter/post_train_quant_convert_pytorch.ipynb) for more details | Off     |
| `--qe-pytorch-backend`   | N/A       | When --qe-convert-pytorch is set, specifies the PyTorch quantization backend to use. Choices: "fbgemm", "qnnpack"   | Off     |
| `--qe-lapq`              | N/A       | Optimize post-training quantization parameters using [LAPQ](https://arxiv.org/abs/1911.07190) method. Beyond the scope of this document. See [example YAML](https://github.com/NervanaSystems/distiller/blob/master/examples/quantization/post_train_quant/resnet18_imagenet_post_train_lapq.yaml) file for details   | Off     |

### Notes

1. These arguments can be added to any `argparse.ArgumentParser` by calling `distiller.quantization.add_post_train_quant_args()` and passing an existing parser. This is provided as a convenience only. If you are writing a script and adding these arguments, it is up to you to implement the actual functionality implied by them.
2. The `--qe-convert-pytorch` works in two settings:
    * `--quantize-eval` is also set, in which case an FP32 model is first quantized using Distiller's post-training quantization flow, and then converted to a PyTorch native quantization model.
    * `--quantize-eval` is not set, but a previously post-train quantized model is loaded via `--resume`. In this case, the loaded model is converted to PyTorch native quantization.

## "Net-Aware" Quantization

The term "net-aware" quantization, coined in [this](https://arxiv.org/abs/1811.09886) paper from Facebook (section 3.2.2), means we can achieve better quantization by considering sequences of operations instead of just quantizing each operation independently. This isn't exactly layer fusion - in Distiller we modify activation stats prior to setting quantization parameters, in to make sure that when a module is followed by certain activation functions, only the relevant ranges are quantized. We do this for:

* **ReLU** - Clip all negative values
* **Tanh / Sigmoid** - Clip according to the (approximated) saturation values for these functions. We use [-4, 4] for tanh and [-6, 6] for sigmoid.

## Static vs. Dynamic Quantization of Activations

Distiller supports both "static" and "dynamic" post-training quantization of **activations**.

### Static Quantization

Pre-calculated tensor statistics are used to calculate the quantization parameters. A preliminary step of collecting these statistics is required. This step is commonly refered to as the **calibration step**.

#### Generating stats

To generate stats, use the `--qe-calibration <VAL>` command line argument. `VAL` should be a numeric value in the range \[0 .. 1\], indicating how much of the test dataset should be used to collect statistics. For example, passing 0.05 will use 5% of the test set. Stats are saved in a YAML file name `acts_quantization_stats.yaml` in the run directory.  
* In the image classification sample, if both `--qe-calibration` and `--quantize-eval` are passed, calibration will be followed by model quantization in the same run. If only the calibration argument is passed, then the script will exit after the calibration step.
* **NOTE:** The image classification sample performs static quantization by default. That means that if a stats file isn't passed (see next section), then a calibration step will be executed prior to quantization, using 5% of the test set (equivalent to using `--qe-calibration 0.05`).

#### Using previously generated stats

In most cases, there is no need to re-run calibration each time we quantize a model. A previously generated stats file can be passed via `--qe-stats-file <path_to_yaml_stats_file>`. This will skip calibration step.

### Dynamic Quantization

Quantization parameters are re-calculated for each batch.
  
**Support for this mode is limited**. It isn't as fully featured as static quantization, and the accuracy results obtained when using it are likely not as representative of real-world results. Specifically:

* Only convolution, FC (aka Linear) and embedding layers are supported at this time. Non-supported layers are kept in FP32, and a warning is displayed.
* "Net-aware" quantization, described above, isn't supported in dynamic mode. 

## Sample Invocations

To execute the command lines below, go into the Distiller image classification sample directory:

```bash
cd <distiller_root>/examples/classifier_compression
```

All the paths used are relative to this directory.

All the examples below are using **static quantization** of activations. As discussed above, to avoid running a calibration step each time, we'll use a pre-generated stats file located at:

```
<dilstiller_root>/examples/quantization/post_train_quant/stats/resnet50_quant_stats.yaml
```

The command line used to generate this file is:

```
python compress_classifier.py -a resnet50 -p 10 -j 22 <path_to_imagenet_dataset> --pretrained --qe-calibration 0.05
```

Note that the `qe-calibration` argument expects a numeric value in the range \[0 .. 1\], which indicates how much of the test dataset should be used to collect statistics. In this case we use 0.05, meaning 5% of the test set is used. 

This table summarizes the settings and results for each run. The command lines for all runs follow in the next table.

|    | Mode       | # Bits Acts | # Bits Weights | Per-Channel | Clip Acts             | Top-1 Accuracy |
|----|------------|-------------|----------------|-------------|-----------------------|----------------|
| 1  | FP32       | 32          | 32             | N/A         |                       | 76.130%        |
| 2  | Symmetric  | 8           | 8              | No          | none                  | 74.904%        |
| 3  | Symmetric  | 8           | 8              | Yes         | none                  | 75.154%        |
| 4  | Symmetric  | 8           | 8              | Yes         | avg                   | 72.268% (See Note 1 below) |
| 5  | Symmetric  | 8           | 8              | Yes         | avg (exc. last layer) | 75.824%        |
| 6  | Asymmetric | 8           | 8              | No          | none                  | 75.292%        |
| 7  | Asymmetric | 8           | 8              | Yes         | avg (exc. last layer) | 75.986%        |
| 8  | Symmetric  | 6           | 6              | No          | none                  | 36.124% (See Note 2 below) |
| 9  | Asymmetric | 6           | 6              | No          | none                  | 62.230%        |
| 10 | Asymmetric | 6           | 6              | Yes         | avg (exc. last layer) | 74.494%        |

(Note that it's possible to define symmetric/asymmetric mode separately for weights and activations using `--qe-mode-wts` and `--qe-mode-acts`, respectively. For brevity and simplicity here we use a monolithic setting via the `--qe-mode` flag)

Command lines:

|    | Command Line |
|----|--------------|
| 1  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate`
| 2  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-mode sym --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`
| 3  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-mode sym --qe-per-channel --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`
| 4  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-mode sym --qe-per-channel --qe-clip-acts avg --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`
| 5  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-mode sym --qe-per-channel --qe-clip-acts avg --qe-no-clip-layers fc --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`
| 6  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-mode asym_u --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`
| 7  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-mode asym_u --qe-per-channel --qe-clip-acts avg --qe-no-clip-layers fc --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`
| 8  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-bits-acts 6 --qe-bits-wts 6 --qe-mode sym --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`
| 9  | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-bits-acts 6 --qe-bits-wts 6 --qe-mode asym_u --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`
| 10 | `python compress_classifier.py -a resnet50 --pretrained <path_to_imagenet_dataset> --evaluate --quantize-eval --qe-bits-acts 6 --qe-bits-wts 6 --qe-mode asym_u --qe-per-channel --qe-clip-acts avg --qe-no-clip-layers fc --qe-stats-file ../quantization/post_train_quant/stats/resnet50_quant_stats.yaml`

### Note 1: Accuracy Loss When Clipping Activations

Notice the degradation in accuracy in run (4) - ~2.6% compared to per-channel without clipping. Let's recall that the output of the final layer of the model holds the "score" of each class (which, since we're using softmax, can be interpreted as the un-normalized log probability of each class). So if we clip the outputs of this layer, we're in fact "cutting-off" the highest (and lowest) scores. If the highest scores for some sample are close enough, this can result in a wrong classification of that sample.  
We can provide Distiller with a list of layers for which not to clip activations. In this case we just want to skip the last layer, which in the case of the ResNet-50 model is called `fc`. This is what we do in run (5), and we regain most of the accuracy back.

### Note 2: Under 8-bits

Runs (8) - (10) are examples of trying post-training quantization below 8-bits. Notice how with the most basic settings we get a massive accuracy loss of ~53%. Even with asymmetric quantization and all other optimizations enabled, we still get a non-trivial degradation of just under 2% vs. FP32. In many cases, quantizing with less than 8-bits requires quantization-aware training. However, if we allow some layers to remain in 8-bit, we can regain some of the accuracy. We can do this by using a YAML configuration file and specifying overrides. As mentioned at the top of this document, check out the `resnet18_imagenet_post_train.yaml` file located in this directory for an example of how to do this.

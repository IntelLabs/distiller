# Post-Training Quantization Examples

Following are a few examples of invoking post-training quantization on ResNet-50, using Distiller's image classification sample. Note that for post-training quantization we don't use a YAML schedule file, instead we specify command line arguments. The available command line arguments are:

| Long Form             | Short     | Description                                                              | Default |
|-----------------------|-----------|--------------------------------------------------------------------------|---------|
| `--quantize-eval`     | `--qe`    | Apply linear quantization to model before evaluation                     | Off     |
| `--qe-mode`           | `--qem`   | Linear quantization mode. Choices: "sym", "asym_u", "asym_s"             | "sym"   |
| `--qe-bits-acts`      | `--qeba`  | # of bits for quantization of activations                                | 8       |
| `--qe-bits-wts`       | `--qebw`  | # of bits for quantization of weights                                    | 8       |
| `--qe-bits-accum`     | N/A       | # of bits for quantization of the accumulator                            | 32      |
| `--qe-clip-acts`      | `--qeca`  | Enable clipping of activations using min/max values averaging over batch | Off     |
| `--qe-no-clip-layers` | `--qencl` | List of layer names (space-separated) for which not to clip activations  | ''      |
| `qe-per-channel`      | `--qepc`  | Enable per-channel quantization of weights (per output channel)          | Off     |

This table summarizes the settings and results for each run. The command lines for all runs follow in the next table.

|   | Mode       | # Bits Acts | # Bits Weights | Per-Channel | Clip Acts             | Top-1 Accuracy |
|---|------------|-------------|----------------|-------------|-----------------------|----------------|
| 1 | FP32       | 32          | 32             | N/A         |                       | 76.13%         |
| 2 | Symmetric  | 8           | 8              | No          | No                    | 75.42%         |
| 3 | Symmetric  | 8           | 8              | Yes         | No                    | 75.66%         |
| 4 | Symmetric  | 8           | 8              | Yes         | Yes                   | 72.54% (See Note 1 below) |
| 5 | Symmetric  | 8           | 8              | Yes         | Yes (exc. last layer) | 75.94%         |
| 6 | Asymmetric | 8           | 8              | No          | No                    | 75.90%         |
| 7 | Symmetric  | 6           | 6              | No          | No                    | 48.46% (See Note 2 below) |
| 8 | Asymmetric | 6           | 6              | No          | No                    | 63.31%         |
| 9 | Asymmetric | 6           | 6              | Yes         | Yes (exc. last layer) | 73.08%         |

Command lines:

|   | Command Line |
|---|--------------|
| 1 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate`
| 2 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate --quantize-eval`
| 3 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate --quantize-eval --qe-per-channel`
| 4 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate --quantize-eval --qe-per-channel --qe-clip-acts`
| 5 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate --quantize-eval --qe-per-channel --qe-clip-acts --qe-no-clip-layers fc`
| 6 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate --quantize-eval --qe-mode asym_u`
| 7 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate --quantize-eval --qe-bits-acts 6 --qe-bits-wts 6`
| 8 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate --quantize-eval --qe-bits-acts 6 --qe-bits-wts 6 --qe-mode asym_u`
| 9 | `python compress_classifier.py -a resnet50 --pretrained ~/datasets/imagenet --evaluate --quantize-eval --qe-bits-acts 6 --qe-bits-wts 6 --qe-mode asym_u --qe-per-channel --qe-clip-acts --qe-no-clip-layers fc`

## Note 1: Accuracy Loss When Clipping Activations

Notice the degradation in accuracy in run (4) - ~3% compared to per-channel without clipping. Let's recall that the output of the final layer of the model holds the "score" of each class (which, since we're using softmax, can be interpreted as the un-normalized log probability of each class). So if we clip the outputs of this layer, we're in fact "cutting-off" the highest (and lowest) scores. If the highest scores for some sample are close enough, this can result in a wrong classification of that sample.  
We can provide Distiller with a list of layers for which not to clip activations. In this case we just want to skip the last layer, which in the case of the ResNet-50 model is called `fc`. This is what we do in run (5), and we regain most of the accuracy back.

## Note 2: Under 8-bits

Runs (7) - (9) are examples of trying post-training quantization below 8-bits. Notice how with the most basic settings we get a massive accuracy loss of almost 28%. Even with asymmetric quantization and all other optimizations enabled, we still get a non-trivial degradation of just over 3% vs. FP32. Quantizing with less than 8-bits, in most cases, required quantization-aware training.
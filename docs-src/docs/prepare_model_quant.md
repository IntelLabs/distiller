# Preparing a Model for Quantization

## Background

*Note: If you just want a run-down of the required modifications to make sure a model is properly quantized in Distiller, you can skip this part and head right to the next section.*

Distiller provides an automatic mechanism to convert a "vanilla" FP32 PyTorch model to a quantized counterpart (for [quantization-aware training](https://intellabs.github.io/distiller/schedule.html#quantization-aware-training) and [post-training quantization](https://intellabs.github.io/distiller/schedule.html#post-training-quantization)). This mechanism works at the PyTorch "Module" level. By "Module" we refer to any sub-class of the `torch.nn.Module` [class](https://pytorch.org/docs/stable/nn.html#module). The Distiller [Quantizer](https://intellabs.github.io/distiller/design.html#quantization) can detect modules, and replace them with other modules.

However, it is not a requirement in PyTorch that all operations be defined as modules. Operations are often executed via direct overloaded tensor operator (`+`, `-`, etc.) and functions under the `torch` namespace (e.g. `torch.cat()`). There is also the `torch.nn.functional` namespace, which provides functional equivalents to modules provided in `torch.nn`. When an operation does not maintain any state, even if it has a dedicated `nn.Module`, it'll often be invoked via its functional counterpart. For example - calling `nn.functional.relu()` instead of creating an instance of `nn.ReLU` and invoking that. Such non-module operations are called directly from the module's `forward` function. There are ways to **discover** these operations up-front, which are [used in Distiller](https://github.com/IntelLabs/distiller/blob/master/distiller/summary_graph.py) for different purposes. Even so, we cannot **replace** these operations without resorting to rather "dirty" Python tricks, which we would rather not do for numerous reasons.

In addition, there might be cases where the same module instance is re-used multiple times in the `forward` function. This is also a problem for Distiller. There are several flows that will not work as expected if each call to an operation is not "tied" to a dedicated module instance. For example:

* When collecting statistics, each invocation of a re-used it will overwrite the statistics collected for the previous invocation. We end up with statistics missing for all invocations except the last one.
* ["Net-aware" quantization](https://github.com/IntelLabs/distiller/blob/master/examples/quantization/post_train_quant/command_line.md#net-aware-quantization) relies on a 1:1 mapping from each operation executed in the model to a module which invoked it. With re-used modules, this mapping is not 1:1 anymore.

Hence, to make sure all supported operations in a model are properly quantized by Distiller, it might be necessary to modify the model code before passing it to the quantizer. Note that the exact set of supported operations might vary between the different [available quantizers](https://intellabs.github.io/distiller/algo_quantization.html).

## Model Preparation To-Do List

The steps required to prepare a model for quantization can be summarized as follows:

1. Replace direct tensor operations with modules
2. Replace re-used modules with dedicated instances
3. Replace `torch.nn.functional` calls with equivalent modules
4. Special cases - replace modules that aren't quantize-able with quantize-able variants

In the next section we'll see an example of the items 1-3 in this list.

As for "special cases", at the moment the only such case is LSTM. See the section after the example for details.

## Model Preparation Example

We'll using the following simple module as an example. This module is loosely based on the ResNet implementation in [torchvision](https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), with some changes that don't make much sense and are meant to demonstrate the different modifications that might be required.

```python
import torch.nn as nn
import torch.nn.functional as F

class BasicModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(BasicModule, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # (1) Overloaded tensor addition operation
        # Alternatively, could be called via a tensor function: skip_1.add_(identity)
        out += identity
        # (2) Relu module re-used
        out = self.relu(out)

        # (3) Using operation from 'torch' namespace
        out = torch.cat([identity, out], dim=1)
        # (4) Using function from torch.nn.functional
        out = F.sigmoid(out)

        return out
```

### Replace direct tensor operations with modules

The addition (1) and concatenation (3) operations in the `forward` function are examples of direct tensor operations. These operations do not have equivalent modules defined in `torch.nn.Module`. Hence, if we want to quantize these operations, we must implement modules that will call them. In Distiller we've implemented a few simple wrapper modules for common operations. These are defined in the `distiller.modules` namespace. Specifically, the addition operation should be replaced with the `EltWiseAdd` module, and the concatenation operation with the `Concat` module. Check out the code [here](https://github.com/IntelLabs/distiller/tree/master/distiller/modules) to see the available modules.

### Replace re-used modules with dedicated instances

The relu operation above is called via a module, but the same instance is used for both calls (2). We need to create a second instance of `nn.ReLU` in `__init__` and use that for the second call during `forward`.

### Replace `torch.nn.functional` calls with equivalent modules

The sigmoid (4) operation is invoked using the functional interface. Luckily, operations in `torch.nn.functional` have equivalent modules, so se can just use those. In this case we need to create an instance of `torch.nn.Sigmoid`.

### Putting it all together

After making all of the changes detailed above, we end up with:

```python
import torch.nn as nn
import torch.nn.functional as F
import distiller.modules

class BasicModule(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(BasicModule, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Fixes start here
        # (1) Replace '+=' with an inplace module
        self.add = distiller.modules.EltWiseAdd(inplace=True)
        # (2) Separate instance for each relu call
        self.relu2 = nn.ReLU()
        # (3) Dedicated module instead of tensor op
        self.concat = distiller.modules.Concat(dim=1)
        # (4) Dedicated module instead of functional call
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.add(out, identity)
        out = self.relu(out)
        out = self.concat(identity, out)
        out = self.sigmoid(out)

        return out
```

## Special Case: LSTM (a "compound" module)

### Background

LSTMs present a special case. An LSTM block is comprised of building blocks, such as fully-connected layers and sigmoid/tanh non-linearities, all of which have dedicated modules in `torch.nn`. However, the LSTM implementation provided in PyTorch does not use these building blocks. For optimization purposes, all of the internal operations are implemented at the C++ level. The only part of the model exposed at the Python level are the parameters of the fully-connected layers. Hence, all we can do with the PyTorch LSTM module is to quantize the inputs/outputs of the entire block, and to quantize the FC layers parameters. We cannot quantize the internal stages of the block at all. In addition to just quantizing the internal stages, we'd also like the option to control the quantization parameters of each of the internal stage separately.

### What to do

Distiller provides a "modular" implementation of LSTM, comprised entirely of operations defined at the Python level. We provide an implementation of `DistillerLSTM` and `DistillerLSTMCell`, paralleling `LSTM` and `LSTMCell` provided by PyTorch. See the implementation [here](https://github.com/IntelLabs/distiller/blob/master/distiller/modules/rnn.py).

A function to convert all LSTM instances in the model to the Distiller variant is also provided:

```python
model = distiller.modules.convert_model_to_distiller_lstm(model)
```

To see an example of this conversion, and of mixed-precision quantization within an LSTM block, check out our tutorial on word-language model quantization [here](https://github.com/IntelLabs/distiller/blob/master/examples/word_language_model/quantize_lstm.ipynb).

# Distiller Installation

## Cloning Distiller
The installation of distiller starts with cloning the Distiller code repository from github.<br>
The rest of the documentation that follows, assumes that you have cloned your repository to a directory called ```distiller```. <br>

## Using a virtualenv
We recommend using a [Python virtual environment](https://docs.python.org/3/library/venv.html#venv-def), but that of course, is up to you.
There's nothing special about using Distiller in a virtualenv, but we provide some instructions, for completeness.<br>
Start by making sure you have virtualenv installed.  Python pip and virtualenv installation instructions can be found [here](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).
<br>
Before creating the environment, make sure you are located in directory ```distiller```.  After creating the environment, you should see a directory called ```distiller/env```.
<br>
Creating the environment:
```
$ python3 -m virtualenv env
```
This creates a subdirectory named ```env``` where the python virtual environment is stored, and configures the current shell to use it as the default python environment.

## Using venv
If you prefer to use ```venv``` , then begin by installing it:
```
$ sudo apt-get install python3-venv
```
Creating the environment:
```
$ python3 -m venv env
```
As with virtualenv, this creates a directory called ```distiller/env```.<br>
<br><br>
The environment activation and deactivation commands for ```venv``` and ```virtualenv``` are the same.<br>
**!NOTE: Make sure to activate the environment, before proceeding with the installation of the dependency packages:<br>**
```
$ source env/bin/activate
```

## Distiller setup 
Install the Python packages Distiller is dependent on using ```pip3 install```.  PyTorch is included in this list and will currently download PyTorch version 3.1 for CUDA 8.0.
```
$ pip3 install -r requirements.txt
```

## Setting up the example code
Distiller comes with a sample application for compressing image classification DNNs, ```compress_classifier.py``` located at ```distiller/examples/classifier_compression```, which uses both [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) and [ImageNet](http://www.image-net.org/) image datasets.<br>

The ```compress_classifier.py``` application will download the CIFAR10 automatically the first time you try to use it (thanks to TorchVision).  The example invocation used  throughout Distiller's documentation assume that you have downloaded the images to directory ```distiller/../data.cifar10```, but you can place the images anywhere you want (you tell ```compress_classifier.py``` where the dataset is located, using a command-line parameter).

ImageNet needs to be [downloaded](http://image-net.org/download-images) manually, due to copyright issues and such.  Download the Imagenet-12 dataset (~1.2 images from 1000 classes).  After downloading the dataset you want to move the validation images to labeled subfolders which you can do with PyTorch's [Soumith's script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).  You can find some more information [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset).

Again, the Distiller documentation assumes the following directory structure for the datasets, but this is just a suggestion:
```
distiller
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
```

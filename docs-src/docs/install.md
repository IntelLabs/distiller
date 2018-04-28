# Distiller Installation

These instructions will help get Distiller up and running on your local machine.

You may also want to refer to these resources:

* [Dataset installation](https://github.com/NervanaSystems/distiller#set-up-the-classification-datasets) instructions.
* [Jupyter installation](https://nervanasystems.github.io/distiller/jupyter/index.html#installation) instructions.

Notes:
- Distiller has only been tested on Ubuntu 16.04 LTS, and with Python 3.5.
- If you are not using a GPU, you might need to make small adjustments to the code.

## Clone Distiller
Clone the Distiller code repository from github:
```
$ git clone https://github.com/NervanaSystems/distiller.git
```
The rest of the documentation that follows, assumes that you have cloned your repository to a directory called ```distiller```. <br>

## Create a Python virtual environment
We recommend using a [Python virtual environment](https://docs.python.org/3/library/venv.html#venv-def), but that of course, is up to you.
There's nothing special about using Distiller in a virtual environment, but we provide some instructions, for completeness.<br>
Before creating the virtual environment, make sure you are located in directory ```distiller```.  After creating the environment, you should see a directory called ```distiller/env```.
<br>
### Using virtualenv
If you don't have virtualenv installed, you can find the installation instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/).

To create the environment, execute:
```
$ python3 -m virtualenv env
```
This creates a subdirectory named ```env``` where the python virtual environment is stored, and configures the current shell to use it as the default python environment.

### Using venv
If you prefer to use ```venv```, then begin by installing it:
```
$ sudo apt-get install python3-venv
```
Then create the environment:
```
$ python3 -m venv env
```
As with virtualenv, this creates a directory called ```distiller/env```.<br>

### Activate the environment
The environment activation and deactivation commands for ```venv``` and ```virtualenv``` are the same.<br>
**!NOTE: Make sure to activate the environment, before proceeding with the installation of the dependency packages:<br>**
```
$ source env/bin/activate
```

## Install dependencies
Finally, install Distiller's dependency packages using ```pip3```:
```
$ pip3 install -r requirements.txt
```
PyTorch is included in the ```requirements.txt``` file, and will currently download PyTorch version 3.1 for CUDA 8.0.  This is the setup we've used for testing Distiller.

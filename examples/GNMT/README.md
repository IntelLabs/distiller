# 1. Problem

This problem uses recurrent neural network to do language translation.

## Requirements
* [Python 3.6](https://www.python.org)
* [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive)
* [PyTorch 0.4.0](https://pytorch.org)
* [sacrebleu](https://pypi.org/project/sacrebleu/)

### Recommended setup
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
* [pytorch/pytorch:0.4_cuda9_cudnn7 container](https://hub.docker.com/r/pytorch/pytorch/tags/)

# 2. Directions
### Steps to configure machine

Common steps for all rnn-translation tests
To setup the environment on Ubuntu 16.04 (16 CPUs, one P100, 100 GB disk), you can use these commands. This may vary on a different operating system or graphics card.


    # Install docker
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo apt-key fingerprint 0EBFCD88
    sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
    sudo apt update
    # sudo apt install docker-ce -y
    sudo apt install docker-ce=18.03.0~ce-0~ubuntu -y --allow-downgrades

    # Install nvidia-docker2
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt install nvidia-docker2 -y


    sudo tee /etc/docker/daemon.json <<EOF
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    EOF
    sudo pkill -SIGHUP dockerd

    sudo apt install -y bridge-utils
    sudo service docker stop
    sleep 1;
    sudo iptables -t nat -F
    sleep 1;
    sudo ifconfig docker0 down
    sleep 1;
    sudo brctl delbr docker0
    sleep 1;
    sudo service docker start

### Steps to download and verify data
Download the data using the following command:

    bash download_dataset.sh

Verify data with:

    bash verify_dataset.sh

### Steps specific to the pytorch version to run and time

    sudo docker build . --rm -t gnmt:latest
    SEED=1
    NOW=`date "+%F-%T"`
    sudo nvidia-docker run -it --rm --ipc=host \
      -v $(pwd)/../data:/data \
      gnmt:latest "./run_and_time.sh" $SEED |tee benchmark-$NOW.log

### one can control which GPUs are used with the NV_GPU variable
    sudo NV_GPU=0 nvidia-docker run -it --rm --ipc=host \ 
      -v $(pwd)/../data:/data \ 
      gnmt:latest "./run_and_time.sh" $SEED |tee benchmark-$NOW.log

# 3. Dataset/Environment
### Publication/Attribution
We use [WMT16 English-German](http://www.statmt.org/wmt16/translation-task.html)
for training.

### Data preprocessing
Script uses [subword-nmt](https://github.com/rsennrich/subword-nmt) package to
segment text into subword units (BPE), by default it builds shared vocabulary of
32,000 tokens.
Preprocessing removes all pairs of sentences that can't be decoded by latin-1
encoder.

### Training and test data separation
Training uses WMT16 English-German dataset, validation is on concatenation of
newstest2015 and newstest2016, BLEU evaluation is done on newstest2014.

### Training data order
By default training script does bucketing by sequence length. Before each epoch
dataset is randomly shuffled and split into chunks of 80 batches each. Within
each chunk it's sorted by (src + tgt) sequence length and then batches are
reshuffled within each chunk.

# 4. Model
### Publication/Attribution

Implemented model is similar to the one from [Google's Neural Machine
Translation System: Bridging the Gap between Human and Machine
Translation](https://arxiv.org/abs/1609.08144) paper.

Most important difference is in the attention mechanism. This repository
implements `gnmt_v2` attention: output from first LSTM layer of decoder goes
into attention, then re-weighted context is concatenated with inputs to all
subsequent LSTM layers in decoder at current timestep.

The same attention mechanism is also implemented in default
GNMT-like models from [tensorflow/nmt](https://github.com/tensorflow/nmt) and
[NVIDIA/OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq).

### Structure

* general:
  * encoder and decoder are using shared embeddings
  * data-parallel multi-gpu training
  * dynamic loss scaling with backoff for Tensor Cores (mixed precision) training
  * trained with label smoothing loss (smoothing factor 0.1)
* encoder:
  * 4-layer LSTM, hidden size 1024, first layer is bidirectional, the rest are
    undirectional
  * with residual connections starting from 3rd layer
  * uses standard LSTM layer (accelerated by cudnn)
* decoder:
  * 4-layer unidirectional LSTM with hidden size 1024 and fully-connected
    classifier
  * with residual connections starting from 3rd layer
  * uses standard LSTM layer (accelerated by cudnn)
* attention:
  * normalized Bahdanau attention
  * model uses `gnmt_v2` attention mechanism
  * output from first LSTM layer of decoder goes into attention,
  then re-weighted context is concatenated with the input to all subsequent
  LSTM layers in decoder at the current timestep
* inference:
  * beam search with default beam size 5
  * with coverage penalty and length normalization
  * BLEU computed by [sacrebleu](https://pypi.org/project/sacrebleu/)

### Loss function
Cross entropy loss with label smoothing (smoothing factor = 0.1), padding is not
considered part of the loss.

### Optimizer
Adam optimizer with learning rate 5e-4.

# 5. Quality

### Quality metric
BLEU score on newstest2014 dataset.
BLEU scores reported by [sacrebleu](https://pypi.org/project/sacrebleu/) package

### Quality target
Uncased BLEU score of 21.80.

### Evaluation frequency
Evaluation of BLEU score is done after every epoch.

### Evaluation thoroughness
Evaluation uses all of `newstest2014.en`.

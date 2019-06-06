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
### Steps to download and verify data
Download the data using the following command:

    bash download_dataset.sh

Verify data with:

    bash verify_dataset.sh


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

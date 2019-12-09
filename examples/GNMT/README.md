# Google's Neural Machine Translation
In this example we quantize [MLPerf's implementation of GNMT](https://github.com/mlperf/training/tree/master/rnn_translator/pytorch)
and show different configurations of quantization to achieve the highest accuracy using **post-training quantization**.

Note that this folder contains only code required to run evaluation. All training code was removed. A link to a pre-trained model is provided below.

## Summary of Post-Training Quantization Results

| Precision | Mode       | Per-Channel | Clip Activations                                              | Bleu Score |
|-----------|------------|-------------|---------------------------------------------------------------|------------|
| FP32      | N/A        | N/A         | N/A                                                           | 22.16      |
| INT8      | Symmetric  | No          | No                                                            | 18.05      |
| INT8      | Asymmetric | No          | No                                                            | 18.52      |
| INT8      | Asymmetric | Yes         | AVG in all layers                                             | 9.63       |
| INT8      | Asymmetric | Yes         | AVG in all layers except attention block                      | 16.94      |
| INT8      | Asymmetric | Yes         | AVG in all layers except attention block and final classifier | 21.49      |

For details on how the model is being quantized, see [below](#what-is-quantized).

## Running the Example

This example is implemented as Jupyter notebook.

### Install Requirements

    pip install -r requirements

(This will install [sacrebleu](https://pypi.org/project/sacrebleu/))

### Get the Dataset

Download the data using the following command:

    bash download_dataset.sh

Verify data with:

    bash verify_dataset.sh

### Download the Pre-trained Model

    wget https://zenodo.org/record/2581623/files/model_best.pth

### Run the Example

    jupyter notebook

And start the `quantize_gnmt.ipynb` notebook.

## Summary of Quantization Results

### What is Quantized

The following operations / modules are fully quantized:

* Linear (fully-connected)
* Embedding
* Element-wise addition
* Element-wise multiplication
* MatMul / Batch MatMul
* Concat

The following operations do not have a quantized implementation. The operations run in FP32, with quantized + de-quantize applied at the op boundary (input and output):

* Softmax
* Tanh
* Sigmoid
* Division by norm in the attention block. That is, in pseudo code:
  ```python
  quant_dequant(x)
  y = x / norm(x)
  quant_dequant(y)
  ```

## Dataset / Environment

### Publication / Attribution

We use [WMT16 English-German](http://www.statmt.org/wmt16/translation-task.html) for training.

### Data preprocessing

Script uses [subword-nmt](https://github.com/rsennrich/subword-nmt) package to segment text into subword units (BPE), by default it builds shared vocabulary of 32,000 tokens. Preprocessing removes all pairs of sentences that can't be decoded by latin-1 encoder.

## Model

### Publication / Attribution

Implemented model is similar to the one from [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144) paper.

Most important difference is in the attention mechanism. This repository implements `gnmt_v2` attention: output from first LSTM layer of decoder goes into attention, then re-weighted context is concatenated with inputs to all subsequent LSTM layers in decoder at current timestep.

The same attention mechanism is also implemented in default GNMT-like models from [tensorflow/nmt](https://github.com/tensorflow/nmt) and [NVIDIA/OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq).

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

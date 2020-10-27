# Word-level language modeling RNN

This example trains a multi-layer RNN (Elman, GRU, or LSTM) on a language modeling task.
It is based on the PyTorch example found [here](https://github.com/pytorch/examples/tree/master/word_language_model). Note that we're using an earlier version, that doesn't include the Transformer implementation.

- [Word-level language modeling RNN](#word-level-language-modeling-rnn)
  - [Running the example](#running-the-example)
  - [Compression: Pruning](#compression-pruning)
  - [Compression: Post-Training Quantization](#compression-post-training-quantization)

## Running the example

By default, the training script uses the Wikitext-2 dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --epochs 6        # Train a LSTM on Wikitext-2 with CUDA, reaching perplexity of 117.61
python main.py --cuda --epochs 6 --tied # Train a tied LSTM on Wikitext-2 with CUDA, reaching perplexity of 110.44
python main.py --cuda --tied            # Train a tied LSTM on Wikitext-2 with CUDA for 40 epochs, reaching perplexity of 87.17
python generate.py                      # Generate samples from the trained LSTM model.
```

The model uses the `nn.RNN` module (and its sister modules `nn.GRU` and `nn.LSTM`)
which will automatically use the cuDNN backend if run on CUDA with cuDNN installed.

During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --log-interval N   report interval
  --save SAVE        path to save the final model
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied
```

Perplexities on PTB are equal or better than
[Recurrent Neural Network Regularization (Zaremba et al. 2014)](https://arxiv.org/pdf/1409.2329.pdf)
and are similar to [Using the Output Embedding to Improve Language Models (Press & Wolf 2016](https://arxiv.org/abs/1608.05859) and [Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling (Inan et al. 2016)](https://arxiv.org/pdf/1611.01462.pdf), though both of these papers have improved perplexities by using a form of recurrent dropout [(variational dropout)](http://papers.nips.cc/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks).

## Compression: Pruning

[**Tutorial: Using Distiller to prune a PyTorch language model**](https://intellabs.github.io/distiller/tutorial-lang_model.html)

We modified the `main.py` script to allow pruning via Distiller's scheduling mechanism. The tutorial linked above provides a step-by-step description of how these modifications were done. It then shows how to use [AGP (automated gradual pruning)](https://arxiv.org/abs/1710.01878) to prune the model to various levels.

The following table summarizes the pruning results obtained in the tutorial. The parameters used here are:  
`--emsize 650 --nhid 1500 --dropout 0.65 --tied --wd 1e-6`

| Sparsity      | # Non-zero parameters | Validation ppl | Test ppl |
|---------------|:---------------------:|:--------------:|:--------:|
| Baseline - 0% | 85,917,000            | 87.49          | 83.85    |
| 70%           | 25,487,550            | 90.67          | 85.96    |
| 70%           | 25,487,550            | 90.59          | 85.84    |
| 70%           | 25,487,550            | 87.40          | 82.93    |
| **80.4%**     | **16,847,550**        | **89.31**      | **83.64**|
| 90%           | 8,591,700             | 90.70          | 85.67    |
| 95%           | 4,295,850             | 98.42          | 92.79    |

We can see that we are able to maintain the original perplexity using only ~20% of the parameters.

## Compression: Post-Training Quantization

[**Tutorial: Post-Training Quantization of a Language Model using Distiller** (Jupyter Notebook)](https://github.com/IntelLabs/distiller/blob/master/examples/word_language_model/quantize_lstm.ipynb)

(Note that post-training quantization is NOT implemented in the `main.py` script - it is only shown in the notebook tutorial)

The tutorial covers the following:

* Converting the model to use Distiller's modular LSTM implementation, which allows flexible quantization of internal LSTM operations.
* Collecting activation statistics prior to quantization
* Creating a `PostTrainLinearQuantizer` and preparing the model for quantization
* "Net-aware quantization" capability of `PostTrainLinearQuantizer`
* Progressively tweaking the quantization settings in order to improve accuracy

The following table summarizes the post-training quantization experiments shown in the tutorial:

| Precision       | INT8: Mode | INT8: Per-channel | INT8: Clipping | FP16 Modules                       | Test ppl |
|-----------------|------------|-------------------|----------------|------------------------------------|----------|
| FP32            | N/A        | N/A               | N/A            | N/A                                | 86.87    |
| Full FP16       | N/A        | N/A               | N/A            | Entire Model                       | 86.80    |
| Full INT8       | Symmetric  | No                | No             | None                               | 104.2    |
| Full INT8       | Asymmetric | Yes               | No             | None                               | 100.45   |
| Full INT8       | Asymmetric | Yes               | Averaging      | None                               | 88.85    |
| Mixed INT8/FP16 | Asymmetric | Yes               | No             | Encoder, decoder, Eltwise add/mult | 86.77    |
| Mixed INT8/FP16 | Asymmetric | Yes               | Averaging      | Encoder, decoder                   | 88.96    |

For more details see the tutorial itself.

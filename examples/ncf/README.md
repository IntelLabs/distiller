# NCF - Neural Collaborative Filtering

The NCF implementation provided here is based on the implementation found in the MLPerf Training GitHub repository.
This sample is not based on the latest implementation in MLPerf, it is based on an earlier revision which uses the ml-20m dataset. The latest code uses a much larger dataset. We plan to move to the latest version in the near future.  
You can fine the revision this sample is based on [here](https://github.com/mlperf/training/tree/fe17e837ed12974d15c86d5173fe8f2c188434d5/recommendation/pytorch).

We've made several modifications to the code:
* Removed all MLPerf specific code including logging
* In `ncf.py`:
  * Added calls to Distiller compression APIs
  * Added progress indication in training and evaluation flows
* In `neumf.py`:
  * Added option to split final the FC layer (the `split_final` parameter). See [below](#side-note-splitting-the-final-fc-layer).
  * Replaced all functional calls with modules so they can be detected by Distiller, as per this [guide](https://intellabs.github.io/distiller/prepare_model_quant.html) in the Distiller docs.
* In `dataset.py`:
  * Speed up data loading - On first data will is loaded from CSVs and then pickled. On subsequent runs the pickle is loaded. This is much faster than the original implementation, but still very slow.
  * Added progress indication during data load process

The sample command lines provided [below](#running-the-sample) focus on **post-training quantization**. We did integrate the capability to run quantization-aware training into `ncf.py`. We'll add examples for this at a later time.

## Problem

This task benchmarks recommendation with implicit feedback on the [MovieLens 20 Million (ml-20m) dataset](https://grouplens.org/datasets/movielens/20m/) with a [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569) model.
The model trains on binary information about whether or not a user interacted with a specific item.

## Summary of Post-Training Quantization Results

| Precision | Mode       | Per-Channel | Split Final Layer | HR@10 |
|-----------|------------|-------------|-------------------|-------|
| FP32      | N/A        | N/A         | N/A               | 63.55 |
| INT8      | Asymmetric | Yes         | No                | 49.54 |
| INT8      | Asymmetric | Yes         | Yes               | 62.78 |

Details on how to run the experiments, including what we mean by "split final layer" are [below](#running-the-sample).

## Setup

* Install `unzip` and `curl`

  ```bash
  sudo apt-get install unzip curl
  ```

* Make sure the latest Distiller requirements are installed

  ```bash
  # Relative to this sample directory
  cd <distiller-repo-root>
  pip install -e .
  ```

* Obtain the ml-20m dataset

  ```bash
  cd <distiller-repo-root>/examples/ncf
  
  # Creates ml-20.zip
  source download_dataset.sh
  
  # Confirms the MD5 checksum of ml-20.zip
  source verify_dataset.sh
  
  # Extracts the dataset into a sub-directory named 'ml-20m'
  # During the last step the script might appear to hang,
  # This is normal, it finishes after a few minutes
  source extract_dataset.sh
  ```

## Running the Sample

### Train a Base FP32 Model

We train a model with the following parameters:

* MLP Side
  * Embedding size per user / item: 128
  * FC layer sizes: 256x256 --> 256x128 --> 128x64
* MF (matrix factorization) Side
  * Embedding size per user / item: 64
* Therefore, the final FC layer size is: 128x1

Adam optimizer is used, with an initial learning rate of 0.0005. Batch size is 2048. Convergence is obtained after 7 epochs.

```bash
python ncf.py ml-20m -l 0.0005 -b 2048 --layers 256 256 128 64 -f 64 --seed 1 --processes 10 -o run/neumf/base_fp32
...
Epoch 0 Loss 0.1179 (0.1469): 100%|█████████████████████████████| 48491/48491 [07:04<00:00, 114.23it/s]
Epoch 0 evaluation
Epoch 0: HR@10 = 0.5738, NDCG@10 = 0.3367, AvgTrainLoss = 0.1469, train_time = 424.52, val_time = 47.04
...
Epoch 6 Loss 0.0914 (0.0943): 100%|█████████████████████████████| 48491/48491 [06:47<00:00, 118.90it/s]
Epoch 6 evaluation
Epoch 6: HR@10 = 0.6355, NDCG@10 = 0.3820, AvgTrainLoss = 0.0943, train_time = 407.84, val_time = 62.99
```

The hit-rate of the base model is 63.55.

### Side-Note: Splitting the Final FC Layer

As mentioned above, we added an option to split the final FC layer of the model (the `split_final` parameter in `NeuMF.__init__`).

The reasoning behind this is that the input to the final FC layer in NCF is a concatenation of the outputs of the MLP and MF "branches". These outputs have very different dynamic ranges.  
In the model we just trained, the MLP branch output range is [0 .. 203] while the MF branch output range is [-6.3 .. 7.4]. When doing quantized concatenation, we have to accommodate the larger range, which leads to a large quantization error for the data that came from the MF branch. When quantizing to 8-bits, the MF branch will cover only 10 bins out of the 256 bins, which means just over 3-bits.  
The mitigation we use is to split the final FC layer as follows:

```
  Before Split:            After Split:
  -------------            ------------
  MF_OUT  MLP_OUT          MF_OUT  MLP_OUT
    \        /               |        |
     \      /      --->    MF_FC   MLP_FC
      CONCAT                 \        /
        |                     \      /
     FINAL_FC                  \    /
                                ADD
```
After splitting, the two inputs to the add operation have ranges [-283 .. 40] from the MLP side and [-54 .. 47] from the MF side. While the problem isn't completely solved, it's much better than before. Now the MF covers 126 bins, which is almost 7-bits.

Note that in FP32 the 2 modes are functionally identical. The split final option is for evaluation only, and we take care to convert the model trained without splitting into a split model when loading the checkpoint. 

### Collect Quantization Stats for Post-Training Quantization

We generated stats for both the non-split and split case. These are the `quantization_stats_no_split.yaml` and `quantization_stats_split.yaml` files in the example folder.

For reference, the command lines used to generate these are:

```bash
python ncf.py ml-20m -b 2048 --layers 256 256 128 64 -f 64 --seed 1 --load run/neumf/base_fp32/best.pth.tar --qe-calibration 0.1
python ncf.py ml-20m -b 2048 --layers 256 256 128 64 -f 64 --seed 1 --load run/neumf/base_fp32/best.pth.tar --qe-calibration 0.1 --split-final
```
Note that `--qe-calibration 0.1` means that we use 10% of the test dataset for the stats collection.

### Post-Training Quantization Experiments

We'll use the following settings for quantization:

* 8-bits for weights and activations: `--qeba 8 --qebw 8`
* Asymmetric: `--qem asym_u`
* Per-channel: `--qepc`

Let's see the difference splitting the final FC layer makes in terms of overall accuracy:

```bash
ncf.py ml-20m -b 2048 --layers 256 256 128 64 -f 64 --seed 1 --load run/neumf/base_fp32/best.pth.tar --evaluate --quantize-eval --qeba 8 --qebw 8 --qem asym_u --qepc --qe-stats-file quantization_stats_no_split.yaml
...
Initial HR@10 = 0.4954, NDCG@10 = 0.2802, val_time = 521.11
```

```bash
ncf.py ml-20m -b 2048 --layers 256 256 128 64 -f 64 --seed 1 --load run/neumf/base_fp32/best.pth.tar --evaluate --quantize-eval --qeba 8 --qebw 8 --qem asym_u --qepc --split-final --qe-stats-file quantization_stats_split.yaml
...
HR@10 = 0.6278, NDCG@10 = 0.3760, val_time = 601.87
```

We can see that without splitting, we get ~14% degradation in hit-rate. With splitting we gain almost all of the accuracy back, with about 0.8% degradation.

## Dataset / Environment

### Publication / Attribution

Harper, F. M. & Konstan, J. A. (2015), 'The MovieLens Datasets: History and Context', ACM Trans. Interact. Intell. Syst. 5(4), 19:1--19:19.

### Data preprocessing

1. Unzip
2. Remove users with less than 20 reviews
3. Create training and test data separation described below

### Training and test data separation

Positive training examples are all but the last item each user rated.
Negative training examples are randomly selected from the unrated items for each user.

The last item each user rated is used as a positive example in the test set.
A fixed set of 999 unrated items are also selected to calculate hit rate at 10 for predicting the test item.

### Training data order

Data is traversed randomly with 4 negative examples selected on average for every positive example.

## Model

### Publication/Attribution

Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua (2017). [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569). In Proceedings of WWW '17, Perth, Australia, April 03-07, 2017.

The author's original code is available at [hexiangnan/neural_collaborative_filtering](https://github.com/hexiangnan/neural_collaborative_filtering).

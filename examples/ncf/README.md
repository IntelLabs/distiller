# NCF - Neural Collaborative Filtering

The NCF implementation provided here is based on the implementation found in the MLPerf Training GitHub repository, specifically on the last revision of the code before the switch to the extended dataset. See [here](https://github.com/mlperf/training/tree/fe17e837ed12974d15c86d5173fe8f2c188434d5/recommendation/pytorch).

We've made several modifications to the code:
* Removed all MLPerf specific code including logging
* In `ncf.py`:
  * Added calls to Distiller compression APIs
  * Added progress indication in training and evaluation flows
* In `neumf.py`:
  * Added option to split final FC layer
  * Replaced all functional calls with modules so they can be detected by Distiller
* In `dataset.py`:
  * Speed up data loading - On first data will is loaded from CSVs and then pickled. On subsequent runs the pickle is loaded. This is much faster than the original implementation, but still very slow.
  * Added progress indication during data load process
* Removed some irrelevant content from this README

## Problem

This task benchmarks recommendation with implicit feedback on the [MovieLens 20 Million (ml-20m) dataset](https://grouplens.org/datasets/movielens/20m/) with a [Neural Collaborative Filtering](http://dl.acm.org/citation.cfm?id=3052569) model.
The model trains on binary information about whether or not a user interacted with a specific item.

## Setup

### Steps to configure machine

1. Install `unzip` and `curl`

```bash
sudo apt-get install unzip curl
```

2. Install required python packages

```bash
pip install -r requirements.txt
```

3. Download and verify data

```bash
# Creates ml-20.zip
source ../download_dataset.sh
# Confirms the MD5 checksum of ml-20.zip
source ../verify_dataset.sh
```

## Running the Sample

### TODO: Add some Distiller specific example command line

## Dataset/Environment

### Publication/Attribution

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

## Quality

### Quality metric

Hit rate at 10 (HR@10) with 999 negative items.

### Evaluation frequency

After every epoch through the training data.

### Evaluation thoroughness

Every users last item rated, i.e. all held out positive examples.

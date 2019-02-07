#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Helper code for data loading.

This code will help with the image classification datasets: ImageNet and CIFAR10

"""
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

DATASETS_NAMES = ['imagenet', 'cifar10']


def load_data(dataset, data_dir, batch_size, workers, validation_split=0.1, deterministic=False, shuffle_test=False,
              effective_train_size=1., effective_valid_size=1., effective_test_size=1.):
    """Load a dataset.

    Args:
        dataset: a string with the name of the dataset to load (cifar10/imagenet)
        data_dir: the directory where the datset resides
        batch_size: the batch size
        workers: the number of worker threads to use for loading the data
        validation_split: portion of training dataset to set aside for validation
        deterministic: set to True if you want the data loading process to be deterministic.
          Note that deterministic data loading suffers from poor performance.
        shuffle_test: set to True if test set should be shuffled by the data loader
        effective_train/valid/test_size: portion of the datasets to actually load. For the training and validation
          sets, this is applied AFTER the split to those sets according to the validation_split parameter
    """
    if dataset not in DATASETS_NAMES:
        raise ValueError('load_data does not support dataset %s" % dataset')
    datasets_fn = cifar10_get_datasets if dataset == 'cifar10' else imagenet_get_datasets
    return get_data_loaders(datasets_fn, data_dir, batch_size, workers, validation_split=validation_split,
                            deterministic=deterministic, shuffle_test=shuffle_test,
                            effective_train_size=effective_train_size, effective_valid_size=effective_valid_size,
                            effective_test_size=effective_test_size)


def cifar10_get_datasets(data_dir):
    """Load the CIFAR10 dataset.

    The original training dataset is split into training and validation sets (code is
    inspired by https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb).
    By default we use a 90:10 (45K:5K) training:validation split.

    The output of torchvision datasets are PIL Image images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1]
    https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py

    Data augmentation: 4 pixels are padded on each side, and a 32x32 crop is randomly sampled
    from the padded image or its horizontal flip.
    This is similar to [1] and some other work that use CIFAR10.

    [1] C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu. Deeply Supervised Nets.
    arXiv:1409.5185, 2014
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def imagenet_get_datasets(data_dir):
    """
    Load the ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset


def __image_size(dataset):
    # un-squeeze is used here to add the batch dimension (value=1), which is missing
    return dataset[0][0].unsqueeze(0).size()


def __deterministic_worker_init_fn(worker_id, seed=0):
    import random
    import numpy
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)


def __split_list(l, ratio):
    split_idx = int(np.floor(ratio * len(l)))
    return l[:split_idx], l[split_idx:]


def get_data_loaders(datasets_fn, data_dir, batch_size, num_workers, validation_split=0.1, deterministic=False,
                     shuffle_test=False, effective_train_size=1., effective_valid_size=1., effective_test_size=1.):
    train_dataset, test_dataset = datasets_fn(data_dir)

    num_train = len(train_dataset)
    indices = list(range(num_train))

    # TODO: Switch to torch.utils.data.datasets.random_split()

    # We shuffle indices here in case the data is arranged by class, in which case we'd would get mutually
    # exclusive datasets if we didn't shuffle
    np.random.shuffle(indices)

    valid_indices, train_indices = __split_list(indices, validation_split)
    effective_train_indices, _ = __split_list(train_indices, effective_train_size)

    train_sampler = SubsetRandomSampler(effective_train_indices)

    worker_init_fn = __deterministic_worker_init_fn if deterministic else None

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    valid_loader = None
    if valid_indices:
        effective_valid_indices, _ = __split_list(valid_indices, effective_valid_size)
        valid_sampler = SubsetRandomSampler(effective_valid_indices)
        valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

    test_indices = list(range(len(test_dataset)))
    if shuffle_test:
        np.random.shuffle(test_indices)
    effective_test_indices, _ = __split_list(test_indices, effective_test_size)
    test_dataset = torch.utils.data.dataset.Subset(test_dataset, effective_test_indices)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=num_workers, pin_memory=True)

    input_shape = __image_size(train_dataset)

    # If validation split was 0 we use the test set as the validation set
    return train_loader, valid_loader or test_loader, test_loader, input_shape

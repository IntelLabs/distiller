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
from torch.utils.data.sampler import Sampler
import numpy as np
import distiller


DATASETS_NAMES = ['imagenet', 'cifar10', 'mnist']


def classification_dataset_str_from_arch(arch):
    if 'cifar' in arch:
        dataset = 'cifar10' 
    elif 'mnist' in arch:
        dataset = 'mnist' 
    else:
        dataset = 'imagenet'
    return dataset


def classification_num_classes(dataset):
    return {'cifar10': 10,
            'mnist': 10,
            'imagenet': 1000}.get(dataset, None)


def classification_get_input_shape(dataset):
    if dataset == 'imagenet':
        return 1, 3, 224, 224
    elif dataset == 'cifar10':
        return 1, 3, 32, 32
    elif dataset == 'mnist':
        return 1, 1, 28, 28
    else:
        raise ValueError("dataset %s is not supported" % dataset)


def __dataset_factory(dataset):
    return {'cifar10': cifar10_get_datasets,
            'mnist': mnist_get_datasets,
            'imagenet': imagenet_get_datasets}.get(dataset, None)


def load_data(dataset, data_dir, batch_size, workers, validation_split=0.1, deterministic=False,
              effective_train_size=1., effective_valid_size=1., effective_test_size=1.,
              fixed_subset=False, sequential=False, test_only=False):
    """Load a dataset.

    Args:
        dataset: a string with the name of the dataset to load (cifar10/imagenet)
        data_dir: the directory where the dataset resides
        batch_size: the batch size
        workers: the number of worker threads to use for loading the data
        validation_split: portion of training dataset to set aside for validation
        deterministic: set to True if you want the data loading process to be deterministic.
          Note that deterministic data loading suffers from poor performance.
        effective_train/valid/test_size: portion of the datasets to load on each epoch.
          The subset is chosen randomly each time. For the training and validation sets,
          this is applied AFTER the split to those sets according to the validation_split parameter
        fixed_subset: set to True to keep the same subset of data throughout the run
          (the size of the subset is still determined according to the effective_train/valid/test
          size args)
    """
    if dataset not in DATASETS_NAMES:
        raise ValueError('load_data does not support dataset %s" % dataset')
    datasets_fn = __dataset_factory(dataset)
    return get_data_loaders(datasets_fn, data_dir, batch_size, workers, 
                            validation_split=validation_split,
                            deterministic=deterministic, 
                            effective_train_size=effective_train_size,
                            effective_valid_size=effective_valid_size, 
                            effective_test_size=effective_test_size,
                            fixed_subset=fixed_subset,
                            sequential=sequential,
                            test_only=test_only)


def mnist_get_datasets(data_dir, load_train=True, load_test=True):
    """Load the MNIST dataset."""
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(root=data_dir, train=True,
                                       download=True, transform=train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_dataset = datasets.MNIST(root=data_dir, train=False,
                                      transform=test_transform)

    return train_dataset, test_dataset


def cifar10_get_datasets(data_dir, load_train=True, load_test=True):
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
    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                         download=True, transform=train_transform)

    test_dataset = None
    if load_test:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=test_transform)

    return train_dataset, test_dataset


def imagenet_get_datasets(data_dir, load_train=True, load_test=True):
    """
    Load the ImageNet dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = None
    if load_train:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_dataset = None
    if load_test:
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


class SwitchingSubsetRandomSampler(Sampler):
    """Samples a random subset of elements from a data source, without replacement.

    The subset of elements is re-chosen randomly each time the sampler is enumerated

    Args:
        data_source (Dataset): dataset to sample from
        subset_size (float): value in (0..1], representing the portion of dataset to sample at each enumeration.
    """
    def __init__(self, data_source, effective_size):
        self.data_source = data_source
        self.subset_length = _get_subset_length(data_source, effective_size)

    def __iter__(self):
        # Randomizing in the same way as in torch.utils.data.sampler.SubsetRandomSampler to maintain
        # reproducibility with the previous data loaders implementation
        indices = torch.randperm(len(self.data_source))
        subset_indices = indices[:self.subset_length]
        return (self.data_source[i] for i in subset_indices)

    def __len__(self):
        return self.subset_length


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """Sequentially samples a subset of the dataset, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def _get_subset_length(data_source, effective_size):
    if effective_size <= 0 or effective_size > 1:
        raise ValueError('effective_size must be in (0..1]')
    return int(np.floor(len(data_source) * effective_size))


def _get_sampler(data_source, effective_size, fixed_subset=False, sequential=False):
    if fixed_subset:
        subset_length = _get_subset_length(data_source, effective_size)
        indices = np.random.permutation(len(data_source))
        subset_indices = indices[:subset_length]
        if sequential:
            return SubsetSequentialSampler(subset_indices)
        else:
            return torch.utils.data.SubsetRandomSampler(subset_indices)
    return SwitchingSubsetRandomSampler(data_source, effective_size)


def get_data_loaders(datasets_fn, data_dir, batch_size, num_workers, validation_split=0.1, deterministic=False,
                     effective_train_size=1., effective_valid_size=1., effective_test_size=1., fixed_subset=False,
                     sequential=False, test_only=False):
    train_dataset, test_dataset = datasets_fn(data_dir, load_train=not test_only, load_test=True)

    worker_init_fn = None
    if deterministic:
        distiller.set_deterministic()
        worker_init_fn = __deterministic_worker_init_fn

    test_indices = list(range(len(test_dataset)))
    test_sampler = _get_sampler(test_indices, effective_test_size, fixed_subset, sequential)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, sampler=test_sampler,
                                              num_workers=num_workers, pin_memory=True)

    input_shape = __image_size(test_dataset)

    if test_only:
        return None, None, test_loader, input_shape

    num_train = len(train_dataset)
    indices = list(range(num_train))

    # TODO: Switch to torch.utils.data.datasets.random_split()

    # We shuffle indices here in case the data is arranged by class, in which case we'd would get mutually
    # exclusive datasets if we didn't shuffle
    np.random.shuffle(indices)

    valid_indices, train_indices = __split_list(indices, validation_split)

    train_sampler = _get_sampler(train_indices, effective_train_size, fixed_subset, sequential)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True,
                                               worker_init_fn=worker_init_fn)

    valid_loader = None
    if valid_indices:
        valid_sampler = _get_sampler(valid_indices, effective_valid_size, fixed_subset, sequential)
        valid_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, sampler=valid_sampler,
                                                   num_workers=num_workers, pin_memory=True,
                                                   worker_init_fn=worker_init_fn)

    # If validation split was 0 we use the test set as the validation set
    return train_loader, valid_loader or test_loader, test_loader, input_shape

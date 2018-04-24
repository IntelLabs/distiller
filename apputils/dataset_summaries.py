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

def dataset_summary(data_loader):
    """Create a histogram of class membership distribution within a dataset.

    It is important to examine our training, validation, and test
    datasets, to make sure that they are balanced.
    """
    print("Analyzing dataset:")
    hist = {}
    for idx, (input, label_batch) in enumerate(data_loader):
        for label in label_batch:
            hist[label] = hist.get(label, 0) + 1
        if idx%50 == 0:
            print("idx: %d" % idx)
            
    nclasses = len(hist)
    from statistics import mean
    print('Dataset contains {} items'.format(len(data_loader.sampler)))
    print('Found {} classes'.format(nclasses))
    for data_class, size in hist.iteritems():
        print('\tClass {} = {}'.format(data_class, size))

    print('mean: ', mean(list(hist.values())))

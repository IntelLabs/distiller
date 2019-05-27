#
# Copyright (c) 2019 Intel Corporation
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

import distiller
import numpy as np
import logging
msglogger = logging.getLogger()

def dataset_summary(data_loader):
    """Create a histogram of class membership distribution within a dataset.

    It is important to examine our training, validation, and test
    datasets, to make sure that they are balanced.
    """
    msglogger.info("Analyzing dataset:")
    print_frequency = 50
    for batch, (input, label_batch) in enumerate(data_loader):
        try:
            all_labels = np.append(all_labels, distiller.to_np(label_batch))
        except NameError:
            all_labels = distiller.to_np(label_batch)
        if (batch+1) % print_frequency == 0:
            # progress indicator
            print("batch: %d" % batch)
            
    hist = np.histogram(all_labels, bins=np.arange(1000+1))
    nclasses = len(hist[0])
    for data_class, size in enumerate(hist[0]):
        msglogger.info("\tClass {} = {}".format(data_class, size))
    msglogger.info("Dataset contains {} items".format(len(data_loader.sampler)))
    msglogger.info("Found {} classes".format(nclasses))
    msglogger.info("Average: {} samples per class".format(np.mean(hist[0])))

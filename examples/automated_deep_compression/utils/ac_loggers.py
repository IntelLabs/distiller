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

import csv


__all__ = ["AMCStatsLogger", "FineTuneStatsLogger"]


class _CSVLogger(object):
    def __init__(self, fname, headers):
        """Create the CSV file and write the column names"""
        with open(fname, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        self.fname = fname

    def add_record(self, fields):
        # We close the file each time to flush on every write, and protect against data-loss on crashes
        with open(self.fname, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)


class AMCStatsLogger(_CSVLogger):
    def __init__(self, fname):
        headers = ['episode', 'top1', 'reward', 'total_macs', 'normalized_macs', 'normalized_nnz',
                   'ckpt_name', 'action_history', 'agent_action_history', 'performance']
        super().__init__(fname, headers)


class FineTuneStatsLogger(_CSVLogger):
    def __init__(self, fname):
        headers = ['episode', 'ft_top1_list']
        super().__init__(fname, headers)

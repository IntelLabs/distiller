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

"""Performance trackers used to track the best performing epochs when training.
"""
import operator
import distiller


__all__ = ["SparsityAccuracyTracker"]


class AccuracyPerformanceTracker(object):
    """Base class for performance trackers using Top1 and Top5 accuracy metrics"""
    def __init__(self, num_best_scores):
        self.perf_scores_history = []
        self.max_len = num_best_scores

    def reset(self):
        self.perf_scores_history = []

    def step(self, model, top1, top5, epoch):
        """Update the list of top training scores achieved so far"""
        raise NotImplementedError

    def best_scores(self, how_many=1):
        """Returns `how_many` best scores experienced so far"""
        if how_many < 1:
            how_many = self.max_len
        how_many = min(how_many, self.max_len)
        return self.perf_scores_history[:how_many]


class SparsityAccuracyTracker(AccuracyPerformanceTracker):
    """A performance tracker which prioritizes non-zero parameters.

    Sort the performance history using the count of non-zero parameters
    as main sort key, then sort by top1, top5 and and finally epoch number.
    """
    def step(self, model, top1, top5, epoch):
        model_sparsity, _, params_nnz_cnt = distiller.model_params_stats(model)
        self.perf_scores_history.append(distiller.MutableNamedTuple({
            'params_nnz_cnt': -params_nnz_cnt,
            'sparsity': model_sparsity,
            'top1': top1,
            'top5': top5,
            'epoch': epoch}))
        # Keep perf_scores_history sorted from best to worst
        self.perf_scores_history.sort(
            key=operator.attrgetter('params_nnz_cnt', 'top1', 'top5', 'epoch'),
            reverse=True)

    def log_best_scores(self, msglogger, how_many=-1):
        if how_many < 1:
            how_many = self.max_len
        how_many = min(how_many, self.max_len)
        for score in self.perf_scores_history[:self.max_len]:
            msglogger.info('==> Best [Top1: %.3f   Top5: %.3f   Sparsity:%.2f   NNZ-Params: %d on epoch: %d]',
                           score.top1, score.top5, score.sparsity, -score.params_nnz_cnt, score.epoch)

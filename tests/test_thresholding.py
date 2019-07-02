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
import torch
import pytest
import distiller


def get_test_tensor():
    return torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0],
                         [10., 11., 12.]])


def test_row_thresholding():
    for dtype in [torch.float16, torch.float32]:
        p = get_test_tensor().type(dtype).cuda()
        group_th = distiller.GroupThresholdMixin()
        mask = group_th.group_threshold_mask(p, 'Rows', 7, 'Max')
        assert torch.eq(mask, torch.tensor([[ 0.,  0.,  0.],
                                            [ 0.,  0.,  0.],
                                            [ 1.,  1.,  1.],
                                            [ 1.,  1.,  1.]], device=mask.device).type(dtype)).all()
    return mask


def test_col_thresholding():
    for dtype in [torch.float16, torch.float32]:
        p = get_test_tensor().type(dtype).cuda()
        group_th = distiller.GroupThresholdMixin()
        mask = group_th.group_threshold_mask(p, 'Cols', 11, 'Max')
        assert torch.eq(mask, torch.tensor([[ 0.,  0.,  1.],
                                            [ 0.,  0.,  1.],
                                            [ 0.,  0.,  1.],
                                            [ 0.,  0.,  1.]], device=mask.device).type(dtype)).all()
    return mask

if __name__ == '__main__':
    test_row_thresholding()
    m = test_col_thresholding()
    print(m)

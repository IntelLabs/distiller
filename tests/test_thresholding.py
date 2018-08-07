import torch
import os
import sys
import pytest
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import distiller


def get_test_tensor():
    return torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0],
                         [10., 11., 12.]])


def test_row_thresholding():
    p = get_test_tensor().cuda()
    group_th = distiller.GroupThresholdMixin()
    mask = group_th.group_threshold_mask(p, 'Rows', 7, 'Max')
    assert torch.eq(mask, torch.tensor([[ 0.,  0.,  0.],
                                        [ 0.,  0.,  0.],
                                        [ 1.,  1.,  1.],
                                        [ 1.,  1.,  1.]], device=mask.device)).all()
    return mask


def test_col_thresholding():
    p = get_test_tensor().cuda()
    group_th = distiller.GroupThresholdMixin()
    mask = group_th.group_threshold_mask(p, 'Cols', 11, 'Max')
    assert torch.eq(mask, torch.tensor([[ 0.,  0.,  1.],
                                        [ 0.,  0.,  1.],
                                        [ 0.,  0.,  1.],
                                        [ 0.,  0.,  1.]], device=mask.device)).all()
    return mask

if __name__ == '__main__':
    m = test_col_thresholding()
    print(m)

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
import numpy as np
import distiller
import common


def get_test_2d_tensor():
    return torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0],
                         [10., 11., 12.]])


def get_test_4d_tensor():
    # Channel normalized L1 norms:
    # 0.8362   torch.norm(a[:,0,:,:], p=1) / 18
    # 0.7625   torch.norm(a[:,1,:,:], p=1) / 18
    # 0.6832   torch.norm(a[:,2,:,:], p=1) / 18

    # Channel L2 norms: tensor([4.3593, 3.6394, 3.9037])
    #   a.transpose(0,1).contiguous().view(3,-1).norm(2, dim=1)

    return torch.tensor(
       [
        # Filter L2 = 4.5039   torch.norm(a[0,:,:,:], p=2)
        [[[-1.2982,  0.7574,  0.7962],  # Kernel L1 = 6.5997   torch.norm(a[0,0,:,:], p=1)
          [-0.6695,  1.5907,  0.2659],
          [ 0.1423,  0.3165, -0.7629]],

         [[-0.5480, -1.2718,  0.8286],  # Kernel L1 = 7.7756   torch.norm(a[0,1,:,:], p=1)
          [-0.6427,  0.3814, -0.7988],
          [ 1.0346,  1.3023, -0.9674]],

         [[-0.7951,  1.8784, -0.5654],  # Kernel L1 = 5.8073   torch.norm(a[0,2,:,:], p=1)
          [ 0.0456, -0.2849, -0.3332],
          [-0.2367,  0.7467,  0.9212]]],

        # Filter L2 = 5.2156   torch.norm(a[1,:,:,:], p=2)
        [[[ 1.3672,  0.2993, -0.0619],  # Kernel L1 = 8.4522   torch.norm(a[1,0,:,:], p=1)
          [ 1.8156,  0.7599,  0.1815],
          [ 0.4136,  1.8316,  1.7214]],

         [[ 0.5125, -1.5329,  0.9257],  # Kernel L1 = 5.9498   torch.norm(a[1,1,:,:], p=1)
          [ 0.9200,  0.4376,  0.5743],
          [-0.0097,  0.9473, -0.0899]],

         [[ 0.2372,  2.4369, -0.3410],  # Kernel L1 = 6.4908  torch.norm(a[1,2,:,:], p=1)
          [-1.0595,  0.8056, -0.0357],
          [-1.0105, -0.1451, -0.4194]]]])


def test_norm_names():
    assert str(distiller.norms.l1_norm) == "L1"
    assert str(distiller.norms.l2_norm) == "L2"
    assert str(distiller.norms.max_norm) == "Max"


def test_threshold_mask():
    # Create a 4-D tensor of 1s
    a = torch.ones(3, 64, 32, 32)
    # Change one element
    a[1, 4, 17, 31] = 0.2
    # Create and apply a mask
    mask = distiller.threshold_mask(a, threshold=0.3)
    assert np.sum(distiller.to_np(mask)) == (distiller.volume(a) - 1)
    assert mask[1, 4, 17, 31] == 0
    assert common.almost_equal(distiller.sparsity(mask), 1/distiller.volume(a))


def test_kernel_thresholding():
    p = get_test_4d_tensor().cuda()
    mask, map = distiller.group_threshold_mask(p, '2D', 6, 'L1')

    # Test the binary map: 1s indicate 2D-kernels that have an L1 above 6
    assert map.shape == torch.Size([6])
    assert torch.eq(map, torch.tensor([1., 1., 0.,
                                       1., 0., 1.], device=map.device)).all()
    # Test the full mask
    expected_mask = torch.tensor(
       [[[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]],


        [[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]]], device=mask.device)
    assert torch.eq(mask, expected_mask).all()
    return mask


def test_filter_thresholding():
    p = get_test_4d_tensor().cuda()
    mask, map = distiller.group_threshold_mask(p, '3D', 4.7, 'L2')

    # Test the binary map: 1s indicate 3D-filters that have an L2 above 4.7
    assert map.shape == torch.Size([2])
    assert torch.eq(map, torch.tensor([0., 1.], device=map.device)).all()
    # Test the full mask
    expected_mask = torch.tensor(
       [[[[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]],


        [[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]]], device=mask.device)
    assert torch.eq(mask, expected_mask).all()
    return mask


def test_channel_thresholding_1():
    p = get_test_4d_tensor().cuda()
    mask, map = distiller.group_threshold_mask(p, 'Channels', 3.7, 'L2')

    # Test the binary map: 1s indicate 3D-channels that have a length-normalized-L2 above 1.3
    assert map.shape == torch.Size([3])
    assert torch.eq(map, torch.tensor([1., 0., 1.], device=map.device)).all()
    # Test the full mask
    expected_mask = torch.tensor(
       [[[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]],


        [[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]]]], device=mask.device)
    assert torch.eq(mask, expected_mask).all()
    return mask


def test_channel_thresholding_2():
    p = get_test_4d_tensor().cuda()
    mask, map = distiller.group_threshold_mask(p, 'Channels', 0.7, 'Mean_L1')

    # Test the binary map: 1s indicate 3D-channels that have a length-normalized-L2 above 1.3
    assert map.shape == torch.Size([3])
    assert torch.eq(map, torch.tensor([1., 1., 0.], device=map.device)).all()
    # Test the full mask
    expected_mask = torch.tensor(
       [[[[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[1., 1., 1.],
          [1., 1., 1.],
          [1., 1., 1.]],

         [[0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]]],

        [[[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]],

          [[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]],

          [[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]]]], device=mask.device)
    assert torch.eq(mask, expected_mask).all()
    return mask


def test_row_thresholding():
    p = get_test_2d_tensor().cuda()
    mask, map = distiller.group_threshold_mask(p, 'Rows', 7, 'Max')

    assert torch.eq(map, torch.tensor([ 0.,  0.,  1., 1.], device=mask.device)).all()
    assert torch.eq(mask, torch.tensor([[ 0.,  0.,  0.],
                                        [ 0.,  0.,  0.],
                                        [ 1.,  1.,  1.],
                                        [ 1.,  1.,  1.]], device=mask.device)).all()
    return mask


def test_col_thresholding():
    p = get_test_2d_tensor().cuda()
    mask, map = distiller.group_threshold_mask(p, 'Cols', 11, 'Max')
    assert torch.eq(mask, torch.tensor([[ 0.,  0.,  1.],
                                        [ 0.,  0.,  1.],
                                        [ 0.,  0.,  1.],
                                        [ 0.,  0.,  1.]], device=mask.device)).all()
    return mask


if __name__ == '__main__':
    m = test_channel_thresholding_2()
    #print(m)

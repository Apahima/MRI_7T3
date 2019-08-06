if __name__ == '__main__':
    from os.path import dirname, join as pjoin
    import scipy.io as sio

    import unittest
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    from fastMRI.data import transforms as Ttorch



    test = unittest.TestCase()
    plt.rcParams.update({'font.size': 12})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    data_dir = pjoin(dirname(sio.__file__), 'data')
    mat_fname = pjoin(data_dir, 'fat-water@3T-3echo')

    print(data_dir)
    print(mat_fname)

    mat_contents = sio.loadmat(mat_fname)
    B0_low = 0.3

    sorted(mat_contents.keys())

    print('N_cov shape', (mat_contents['n_cov'].shape))
    print('k_rad shape', (mat_contents['k_high'].shape))
    print('T shape', (mat_contents['TE'].shape))

    k_high_T = Ttorch.to_tensor(mat_contents['k_high'])

    print('k_high Tensor shape:', k_high_T.shape)

    inParam = {'B_high': 3, 'B_low': B0_low, 'tissue': 'liver', 'sequence': 'GradientEcho', 'theta': 3, 'BW_high': 62.5,
               'BW_low': 62.5, 'TR_high': torch.tensor(9), 'TR_low': 4.6, 'n_cov': mat_contents['n_cov']}

    # inParam['TE_high'] = 2.6
    # inParam['TE_low'] = 2.6
    inParam['BW_low'] = inParam['BW_high'] * B0_low / 3
    inParam['TR_low'] = mat_contents['TE'][-1] * 1000 * 3 / B0_low + 1 / inParam['BW_low'] / 2 + 5.608

    k_low = torch.zeros_like(k_high_T)
    print('k_low Tensor with two channels size:', k_low.shape)

    import Function.lowfieldgen as Klow
    import Function.gridkb
    import Function.elementwise_mul_abs_complex

    manipol = Klow.lowfieldgen()  # This is a clsass so it must be loaded with ()
    gridkb = Function.gridkb.gridkb  # This is function so it not neccessery to load it with ()
    elementwise_mul_abs_complex = Function.elementwise_mul_abs_complex.elementwise_mul_abs_complex

    for i in range(0, 3):
        inParam['k_high'] = k_high_T[:, :, :, i, :, :]
        inParam['TE_high'] = mat_contents['TE'][i]
        inParam['TE_low'] = mat_contents['TE'][i] * 1000 * 3 / B0_low
        k_low[:, :, :, i, :, :] = manipol.klow(inParam)

import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io
import os

from Function import transforms as Ttorch

'''Comment'''


test = unittest.TestCase()
plt.rcParams.update({'font.size': 12})
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

import Function.lowfieldgen as Klow
import Function.gridkb
import Function.elementwise_mul_abs_complex
from torch import nn

class lowfieldsim():

    def __init__(self):
        # Loading raw data to low field class
        self.manipol = Klow.lowfieldgen()  # This is a clsass so it must be loaded with ()
        self.elementwise_mul_abs_complex = Function.elementwise_mul_abs_complex.elementwise_mul_abs_complex


        self.mat_contents = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'Data','fat-water@3T-3echo.mat')) #Load directly from the working directory
        self.B0_low = 0.3

        sorted(self.mat_contents.keys())

        self.k_high_T = Ttorch.to_tensor(self.mat_contents['k_high'])
        # print('k_high Tensor shape:', self.k_high_T.shape)

        #Setting parameters
        self.inParam = {'B_high': 3, 'B_low': self.B0_low, 'tissue': 'liver', 'sequence': 'GradientEcho', 'theta': 3,
                   'BW_high': 62.5,
                   'BW_low': 62.5, 'TR_high': 9.0, 'TR_low': 4.6, 'n_cov': self.mat_contents['n_cov']}

        # inParam['TE_high'] = 2.6
        # inParam['TE_low'] = 2.6
        self.inParam['BW_low'] = self.inParam['BW_high'] * self.B0_low / 3
        self.inParam['TR_low'] = self.mat_contents['TE'][0][-1] * 1000 * 3 / self.B0_low + 1 / self.inParam['BW_low'] / 2 + 5.608


    def lowfieldsim(self, k_space_high_field):


        k_low = torch.zeros_like(k_space_high_field)
        # print('k_low Tensor with two channels size:', k_low.shape)


        for i in range(0, 1):
            self.inParam['k_high'] = k_space_high_field[:, :, :, i, :, :]
            self.inParam['TE_high'] = self.mat_contents['TE'][0][i] * 1000
            self.inParam['TE_low'] = self.mat_contents['TE'][0][i] * 1000 * 3 / self.B0_low

            #Generate lowfield k-space from existing high field k-space
            k_low[:, :, :, i, :, :] = self.manipol.klow(self.inParam)

        size_k_low = torch.tensor(1.0*k_low.size(-5)) # Scaling factor using the image size

        k_space_high_field = k_space_high_field.permute(2, 3, 4, 0, 1, 5)  # permute to orgenize it to feed it thru the ifft2 function. two fft dim. need to be -2, -3 position
        # print('k_high_T', k_space_high_field.shape)

        k_low = k_low.permute(2, 3, 4, 0, 1, 5)  # permute to orgenize it to feed it thru the ifft2 function. two fft dim. need to be -2, -3 position
        # print('k_low',k_low.shape)

        recon_high = torch.sqrt(size_k_low * size_k_low) * Ttorch.ifft2_MRI_7T3(k_space_high_field) # Pass thru fft2 and scaling
        recon_low = torch.sqrt(size_k_low * size_k_low) * Ttorch.ifft2_MRI_7T3(k_low)

        # print('recon_high',recon_high.shape)
        recon_high = recon_high.permute(3, 4, 0, 1, 2, 5) # Re-orgenize to [image_w, inage_h, z_dim=1, # pictures, #coils, complex (2)]
        # print('recon_high',recon_high.shape)
        #
        # print('recon_low',recon_low.shape)
        recon_low = recon_low.permute(3, 4, 0, 1, 2, 5)  # Re-orgenize to [image_w, inage_h, z_dim=1, # pictures, #coils, complex (2)]
        # print('recon_low',recon_low.shape)

        recon_high = recon_high[:, :, :, 0, :, :] #Choose one of the three images
        # print('recon_high', recon_high.shape)

        recon_low = recon_low[:, :, :, 0, :, :] #Choose one of the three images
        # print('recon_low',recon_low.shape)

        # Transform the k-space to real image with
        high_res_real = self.elementwise_mul_abs_complex(recon_high).squeeze(-2)
        # calculate abs(recon_high **2), where the **2 is element-wise multiplication (a+ib)**2.
        # self.elementwise_mul_abs_complex output is [image_2, image_h, z_dim, #coils] then I'm doing squeeze to reduce the z_dim.

        low_res_real =self.elementwise_mul_abs_complex(recon_low).squeeze(-2)   # calculate abs(recon_high **2), where the **2 is element-wise multiplication (a+ib)**2
        ###

        img_high_combined = torch.sum(high_res_real, 2) #sum over the #coils
        # print('img_high_combined',img_high_combined.shape)

        img_low_combined = torch.sum(low_res_real, 2) #sum over the #coils
        # print('img_low_combined', img_low_combined.shape)

        # plt.figure(figsize=(15, 15))
        #
        # # Accquired
        # plt.subplot(121),
        # plt.imshow(np.sqrt((img_high_combined)), cmap='gray')
        # plt.title('Accquired @ 3T')
        #
        # # low field
        # plt.subplot(122),
        # plt.imshow(np.sqrt((img_low_combined)), cmap='gray');
        # plt.title("Simulated @ %s T" % self.inParam['B_low'])

        return low_res_real, high_res_real,img_low_combined ,img_high_combined

if __name__ == '__main__':
    mat_contents = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'Data','fat-water@3T-3echo.mat')) #Load directly from the working directory
    B0_low = 0.3

    sorted(mat_contents.keys())

    k_high_T = Ttorch.to_tensor(mat_contents['k_high'])

    simulator_build = lowfieldsim()
    simulator_build.lowfieldsim(k_high_T)
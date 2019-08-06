import torch.nn as nn
import torch
import numpy as np
from fastMRI.data import transforms as T


class lowfieldgen(nn.Module):
    """
    """
    def __init__(self):
        # T1 & T2 Correction Table REF: Principles of MRI.D.G.Nishimura
        self.B0_set = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3]

        self.tissue_type = {'muscle', 'kidney', 'white matter', 'gray matter', 'liver', 'fat', 'other'}

        self.T1_muscle = [0.28, 0.37, 0.45, 0.50, 0.55, 0.73, 0.87, 1.42]
        self.T1_kidney = [0.34, 0.39, 0.44, 0.47, 0.50, 0.58, 0.65, 1.19]
        self.T1_wm = [0.31, 0.39, 0.45, 0.49, 0.53, 0.68, 0.78, 1.08]
        self.T1_gm = [0.40, 0.49, 0.55, 0.61, 0.65, 0.82, 0.92, 1.82]
        self.T1_liver = [0.18, 0.23, 0.27, 0.30, 0.32, 0.43, 0.49, 0.81]
        self.T1_fat = [0.17, 0.18, 0.20, 0.21, 0.22, 0.24, 0.26, 0.30]

        self.T2_muscle = 0.047
        self.T2_kidney = 0.058
        self.T2_wm = 0.092
        self.T2_gm = 0.1
        self.T2_liver = 0.043
        self.T2_fat = 0.085

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def klow(self, inParam):
        # print('Test')
        # print(type(inParam))
        # print('Shape k_high_T', inParam['k_high'].shape)

        (Nkx, Nky, Nkz, Ncoil, Complex) = tuple(inParam['k_high'].shape)
        Nt = 1
        if inParam['n_cov'].size == 1:
            n_cov = inParam['n_cov'] * torch.eye(Ncoil)
        else:
            n_cov = inParam['n_cov']



        def unismember(A, B):
            if A in B:
                return 0
            # print('Len',len(A))
            # print([np.sum(a != B) for a in A])
            return 1
        def ismember(A, B):
            if A in B:
                return 1

            return 0
        def myany(a,b):
            # print(self.B0_set == inParam['B_high'])
            return self.B0_set == inParam['B_high']

        # print(unismember('T1_high', inParam))


        if unismember('T1_high', inParam):
            if myany(self.B0_set, inParam['B_high']) or unismember(inParam['tissue'], self.tissue_type):
                #I use unismemner for (self.B0_set , inParam['B_high']) the original code use ~any, need to check the difference
                print('specify T1 value at high field strength')
            else:
                ind = 7
                def f(x, ind):
                #Define the switch case in Matlab code line 94 function lowfieldgen
                    return {'muscle': self.T1_muscle[ind], 'kidney': self.T1_kidney[ind], 'white matter': self.T1_wm[ind], 'gray matter': self.T1_gm[ind], 'liver': self.T1_liver[ind], 'fat': self.T1_fat[ind]}.get(x, 'specify T1 value at high field strength')
                T1_high = f(inParam['tissue'], ind)
        else:
            T1_high = inParam['T1_high']

        # print('T1_high',T1_high)


        if unismember('T1_low', inParam):
            if myany(self.B0_set , inParam['B_low']) or unismember(inParam['tissue'], self.tissue_type):
                #I use unismemner for (self.B0_set , inParam['B_high']) the original code use ~any, need to check the difference
                print('specify T1 value at low field strength')
            else:
                ind = 2
                def f(x, ind):
                #Define the switch case in Matlab code line 94 function lowfieldgen
                    return {'muscle': self.T1_muscle[ind], 'kidney': self.T1_kidney[ind], 'white matter': self.T1_wm[ind], 'gray matter': self.T1_gm[ind], 'liver': self.T1_liver[ind], 'fat': self.T1_fat[ind]}.get(x, 'specify T1 value at low field strength')
                T1_low = f(inParam['tissue'], ind)
        else:
            T1_low = inParam['T1_low']

        # print('T1_low',T1_low)


        if unismember('T2', inParam):
            if unismember(inParam['tissue'], self.tissue_type):
                #I use unismemner for (self.B0_set , inParam['B_high']) the original code use ~any, need to check the difference
                print('specify T2 value')
            else:
                ind = 2
                def f(x):
                #Define the switch case in Matlab code line 94 function lowfieldgen
                    return {'muscle': self.T2_muscle, 'kidney': self.T2_kidney, 'white matter': self.T2_wm, 'gray matter': self.T2_gm, 'liver': self.T2_liver, 'fat': self.T2_fat}.get(x, 'specify T2 value')
                T2 = f(inParam['tissue'])
        else:
            T2 = inParam['T2']

        # print('T2:', T2)

        #TE
        TE_high = inParam['TE_high'] / 1000    # ms --> sec
        TE_low = inParam['TE_low'] / 1000      # ms --> sec
        #TR
        TR_high = inParam['TR_high'] / 1000       # ms --> sec
        TR_low = inParam['TR_low'] / 1000         # ms --> sec
        #Flip angle
        theta = inParam['theta'] * np.pi / 180    # degree --> rad

        #Readout bandwidth
        BW_high = inParam['BW_high']
        BW_low = inParam['BW_low']
        ## Signal Scaling

        E1_h = np.exp(-TR_high / T1_high)
        E1_l = np.exp(-TR_low / T1_low)
        E2_h = np.exp(-TE_high / T2)
        E2_l = np.exp(-TE_low / T2)

        a = torch.tensor(inParam['B_low'] / inParam['B_high'])

        if inParam['sequence'] == 'SpinEcho':
            fx = torch.tensor(((1 - E1_l) / (1 - E1_l * np.cos(theta))) / ((1 - E1_h) / (1 - E1_h * np.cos(theta)))*(E2_l / E2_h))
            scaleS = fx * (a ** 2)
        elif inParam['sequence'] == 'GradientEcho':
            fx = torch.tensor(((1 - E1_l) / (1 - E1_l * np.cos(theta))) / ((1 - E1_h) / (1 - E1_h * np.cos(theta)))*(E2_l / E2_h))
            scaleS = fx * (a ** 2)

        ## Noise Scaling
        b = torch.tensor(BW_low / BW_high)
        scaleN = torch.sqrt(a ** 2 * b - (a ** 4) * (fx ** 2))

        # # #---------- Sanity check ----------# #
        # print('TE_high',TE_high)
        # print('TE_low',TE_low)
        # print('TR_high',TR_high)
        # print('TR_low',TR_low)
        #
        #
        # print('theta', theta)
        # print('BW_high', BW_high)
        # print('BW_low', BW_low)
        # print('E1_h', E1_h)
        # print('E1_l', E1_l)
        # print('E2_h', E2_h)
        # print('E2_l', E2_l)
        #
        # print('a', a)
        #
        # print('fx', fx)
        # print('scaleS', scaleS)
        #
        # print('b', b)
        # print('scaleN', scaleN)
        # # #---------- End ----------# #

        ###### ------------  #####
        # Go to numpy function due to decomposition problem
        # Becuase n_cov is setup \ tool parameter and it's constant per tool I can donsider it like a constant and
        # use regular decomposition without Tensor tracking.

        # U = n_cov[:,:,0].numpy()+ 1j*n_cov[:,:,0].numpy()
        # print(n_cov)
        d, v = np.linalg.eigh(n_cov)   #d - eigenvectors, v - eigenvalues
        d = d * np.eye(d.size)


        # print('EigenVectors matrix:')
        # print('v shape', v.shape)
        # print('EigenValues matrix:')
        # print(d)
        # print('End')
        # print('d shape', d.shape)
        # print((v @ np.sqrt(d)))
        # print(torch.randn(Ncoil, Nkx * Nky * Nkz * Nt).shape)
        eig_torch = T.to_tensor(v @ np.sqrt(d))
        # print('eig', eig_torch.shape)

        noise = torch.zeros(Nkx * Nky * Nkz,Ncoil,2, dtype=torch.float).to(self.device)
        noise[:,:,0] = scaleN * (eig_torch[:,:,0] @ torch.randn(Ncoil, Nkx * Nky * Nkz * Nt, dtype=torch.double)).t()
        noise[:,:,1] = scaleN * (eig_torch[:,:,1] @ torch.randn(Ncoil, Nkx * Nky * Nkz * Nt, dtype=torch.double)).t()


        noise = torch.reshape(noise, (Nkx, Nky, Nkz, Ncoil,2))
        # print('noise shape',noise.shape)
        # print('K_high shape' , inParam['k_high'].shape)
        #Output
        k_low = torch.zeros_like(inParam['k_high'])
        k_low[:,:,:,:,0] = (inParam['k_high'][:,:,:,:,0]) * scaleS + noise[:,:,:,:,0]
        k_low[:,:,:,:,1] = (inParam['k_high'][:,:,:,:,1]) * scaleS + noise[:,:,:,:,0]
        # print('K_low shape:', k_low)

        return k_low
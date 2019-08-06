import torch.nn as nn
import torch
import numpy as np
from fastMRI.data import transforms as T
import scipy
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


def gridkb(d,k,w,n ,osf ,wg ,opt = 'image'):



    d = torch.reshape(d.permute(1,0,2), (-1,2))
    k = torch.reshape(k.permute(1,0,2), (-1,2))
    w = torch.reshape(w.t(), (-1,))

    # print('k',k[0:10,0])
    # print('d',d[0:10,0])
    # print('w',w[0:10])
    # print('d shape:',d.size())
    # print('k shape:',k.size())
    # print('w shape:', w.size())


    kw = torch.div(wg , osf)
    # print(kw)

    dw = w.unsqueeze(-1) * d
    # print('d',d[0:5,:])
    # print('k', k[0:5, :])
    # print('w', w[0:5])
    # print('dw',dw[0:5])
    # print(d[0, 0], d[0, 1])
    # print(dw[0,0], dw[0,1])
    # osf = torch.tensor(2, dtype=torch.double)

    # print('osf:',(osf))

    # compute kernel, assume e1 is 0.001, assuming nearest neighbor
    kosf = torch.floor(0.91/(osf*1e-3))
    # print('ksof:',kosf)
    kwidth = osf * kw / 2

    # beta from the Beatty paper
    beta = np.pi * torch.sqrt((kw * (osf - 0.5))**2 - 0.8)
    # print('beta',beta)
    # compute kernel

    om = torch.arange(0,kosf * kwidth+1) / (kosf * kwidth)
    # print(om[-1])
    # print('om dtype',om.dtype)
    p = scipy.special.iv(0, beta * torch.sqrt(1 - om * om)).double()

    # print(om)
    # print('p type:',p.dtype)
    p = p / p[1]


    # last sample is zero so we can use min() below for samples bigger than kwidth
    p[-1] = 0
    # plt.plot(p)


    # convert k-space samples to matrix indices
    nx = (n * osf / 2 + 1) + osf * n * k[:,0]
    ny = (n * osf / 2 + 1) + osf * n * k[:,1]


    # print(nx.dtype)
    # print(ny.dtype)

    m = torch.zeros((osf.int() * n, osf.int() * n,2), dtype=torch.double)



    # loop over samples in kernel at grid spacing
    for lx in range(-kwidth.int(),kwidth.int()):
        for ly in range(-kwidth.int(),kwidth.int()):




        ################################
        ##### --------------- ##########
            a = torch.zeros_like(m)
        ##### --------------- ##########
        ################################



        # find nearest samples
            nxt = torch.round(nx + lx)
            nyt = torch.round(ny + ly)

            # print('nxt dtype:',nxt.dtype)
            # print('nyt dtype:',nyt.dtype)
        # seperable kernel value
            kkx = torch.min(torch.round(kosf * torch.abs(nx - nxt) + 1), torch.floor(kosf * kwidth) + 1) - 1
            kwx = p[kkx.tolist()]

            kky = torch.min(torch.round(kosf * torch.abs(ny - nyt) + 1), torch.floor(kosf * kwidth) + 1) - 1
            kwy = p[kky.tolist()]

            # print('size kwx:', kwx.size())
            # print('kwx dtype:', kwx.dtype)

        # if data falls outside matrix, put it at the edge, zero out below
        # ---------------------- ADD ZEROS AT THE END OF THE CODE BECUASE OF THE TRANSFORM FROM MATLAB TO Python ARRAY INDICES --------
        #     print('nxt before', max(nxt))
        #     print('nyt before', max(nyt))
            nxt = torch.max(nxt, torch.tensor(0, dtype=torch.double))
            nxt = torch.min(nxt, osf * n - 1)
            nyt = torch.max(nyt, torch.tensor(0, dtype=torch.double))
            nyt = torch.min(nyt, osf * n - 1)


        # accumulate gridded data
        #     print('multiple shape:',(dw*kwx.unsqueeze(-1)*kwy.unsqueeze(-1)).shape)
        #     print('osf * n:',(osf * n).int())
        #     print('nxt',max(nxt))
        #     print('nyt', max(nyt))

            indi = torch.LongTensor([nxt.tolist(),nyt.tolist()])
            # print(indi.size())
            # print((dw[:,0]*kwx*kwy).size())

            a[nxt.tolist(), nyt.tolist(), 0] = dw[:, 0]*kwx*kwy
            a[nxt.tolist(), nyt.tolist(), 1] = dw[:, 1]*kwx*kwy


            # csr_matrix
            # print('b',b)
            # print('a',a)

            m[:,:,0] += a[:,:,0]
            m[:,:,1] += a[:,:,1]


            # print(m)
    # print('m shape:', m)
    m[:,0,:] = 0
    m[:,-1,:] = 0
    m[0,:,:] = 0
    m[-1,:,:] = 0

    if opt == 'k-space':
        return m, p

    im = T.ifftshift(T.ifft2(T.ifftshift(m)))




    # compute deappodization function
    x = torch.arange(-osf * n / 2,osf * n / 2, dtype=torch.double) / (n)


    ########################################
    ########################################
    ########## ------- Here I should do sqrt() for negative numbers, i.e. complex  -------------############

    sqa = torch.sqrt((-1)*(np.pi * np.pi * kw * kw * x * x - beta * beta))


    dax = torch.sinh(sqa)/ (sqa)

    ########################################
    ########################################

    # normalize by DC value
    dax = dax / dax[(osf * n / 2).int()]
    # make it a 2D array
    da = dax.unsqueeze(-1) * dax.unsqueeze(-1).t()

    # deappodize
    # im = im. / da;
    im = im / (da + 1).unsqueeze(-1) # add a constant to reduce deapodization at the edge of FOV, ZW.

    # return the result
    m = im
    return m

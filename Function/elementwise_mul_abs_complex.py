import torch


def elementwise_mul_abs_complex(data):
    assert data.size(-1) == 2

    # sqrt( (a**2 - b**2)**2 + 4 * a**2 * b**2)

    return torch.sqrt(
        (data[:, :, :, :, 0] ** 2 - data[:, :, :, :, 1] ** 2) ** 2 + 4 * data[:, :, :, :, 0] ** 2 * data[:, :, :, :,1] ** 2
                    )



def elementwise_mul_abs_complex2(data):
    assert data.size(-1) == 2

    # sqrt( (a**2 - b**2)**2 + 4 * a**2 * b**2)

    return torch.sqrt(
        (data[:,:,:,0] ** 2 - data[:,:,:,0] ** 2) ** 2 + 4 * data[:,:,:,0] ** 2 * data[:,:,:,0] ** 2
    )
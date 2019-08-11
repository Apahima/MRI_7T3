import torch
from torchvision import transforms

def Normalize_Per_Chan(Normalz_in, eps=0.):
    #Assuming the channels to be normalized located at dim 0
    # Normalized to normal distribution
    mean = Normalz_in.mean(-1).mean(-1)
    std = Normalz_in.std(-1).std(-1)

    output = transforms.functional.normalize(Normalz_in,mean,std)

    return output


def Min_Max_Scaling(Scaling_in):
    # Assuming the channels are in dim 0
    # Scailing each image to be [0,1]
    input_max = Scaling_in.max(-1, keepdim=True)[0].max(-2,keepdim=True)[0]
    input_min = Scaling_in.min(-1,keepdim=True)[0].min(-2,keepdim=True)[0]

    output = (Scaling_in - input_min) / (input_max - input_min)
    return output

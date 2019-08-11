import logging
import os
import pathlib
import random
import shutil
from datetime import datetime
from datetime import date

import numpy as np
import torch
import torchvision
from lowfieldsim import lowfieldsim as simulator_MRI
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fastMRI.common.args import Args
from fastMRI.common.subsample import MaskFunc
from fastMRI.data import transforms
from fastMRI.data.mri_data import SliceData
from fastMRI.models.unet.unet_model import UnetModel
import Function.Normalization_Per_Channel as Norm_Per_Chan

import scipy.io
from fastMRI.data import transforms as Ttorch


def build_model(args):
    model = UnetModel(
        in_chans=args.num_coil, # define the argument
        out_chans=args.num_coil, # define the argument
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    return model


def build_optim(args, params):
    optimizer = torch.optim.RMSprop(params, args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    return optimizer

def get_data(k_high_T):
    #return:
    # low field image - target, all coils as different channels
    # high field image - output_gt image, all coils as different channels
    # low field image \ noise - input_mr, all coils as different channels

    simulator_class_MRI = simulator_MRI()

    low_res_real, high_res_real, _, _ = simulator_class_MRI.lowfieldsim(k_high_T)
    input_mr = low_res_real.permute(2,0,1)
    target = low_res_real.permute(2,0,1)
    output_gt =  high_res_real.permute(2,0,1)


    # #Normalization to normal distibution Per channel
    # input_mr = Norm_Per_Chan.Min_Max_Scaling(input_mr)
    # target = Norm_Per_Chan.Min_Max_Scaling(target)
    # output_gt = Norm_Per_Chan.Min_Max_Scaling(output_gt)
    #
    # #Normalization to normal distibution Per channel
    # input_mr = Norm_Per_Chan.Normalize_Per_Chan(input_mr)
    # target = Norm_Per_Chan.Normalize_Per_Chan(target)
    # output_gt = Norm_Per_Chan.Normalize_Per_Chan(output_gt)


    #Scaling Min-Max [0,1] overall
    input_mr = (input_mr - torch.min(input_mr)) / (torch.max(input_mr)-torch.min(input_mr))
    target = (target - torch.min(target)) / (torch.max(target)-torch.min(target))
    output_gt = (output_gt - torch.min(output_gt)) / (torch.max(output_gt)-torch.min(output_gt))
    ######


    #Sanity check
    # target = torch.randn(8, 128, 128)
    # output_gt = torch.randn(8, 128, 128)
    # input_mr = torch.randn(8,128,128)
    return input_mr, target, output_gt

def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    construct_time_stamp = now.strftime("_%I-%M-%S %p")
    folder_name = str(date.today()) + str(construct_time_stamp)

    writer = SummaryWriter(log_dir=args.exp_dir / folder_name / 'summary')

    mat_contents = scipy.io.loadmat('fat-water@3T-3echo.mat')
    k_high_T = Ttorch.to_tensor(mat_contents['k_high']).to(args.device)

    model = build_model(args)
    print(model)

    # input, target & ground truth.  target is always low field, input can be anything: noise, low-field
    input_mr, target, output_gt, = get_data(k_high_T)
    input_mr = input_mr.unsqueeze(0).to(args.device)
    target = target.to(args.device)


    simulator_class_MRI = simulator_MRI()

    optimizer = build_optim(args, model.parameters())
    print(optimizer)

    best_dev_loss = 1e9
    start_epoch = 0
    is_new_best = 1e10
    logging.info(args)
    logging.info(model)

    for epoch in range(start_epoch, args.num_epochs):
        output = model.forward(input_mr)

        # arrange the output from model to fit MRI simulator size. output = [1, #channels (coils), in_width, in_height]
        output_to_sim = output.unsqueeze(-1)
        output_to_sim = torch.cat((output_to_sim, torch.zeros_like(output_to_sim)), dim=-1)  # fit the out put for the FFT function, add zeros channel at (-1) position,
        # k_space_output = torch.fft(output_to_sim, 2, normalized=False)  # Getting FFT results, this is the k-space [1, #channels (coils), in_width, in_height, complex (2)]
        k_space_output = Ttorch.fft2(output_to_sim)
        k_space_output = k_space_output.unsqueeze(-5)  # insert one more channel and duplicate the existing data, because the MRI simulator have 3 samples for the same image
        # k_space_output = torch.cat((k_space_output, k_space_output, k_space_output), dim=1)  # duplicate the data three times just for MRI simulator compatible, inside the simulator we take only one picture i.e. dim(2) = 1.
        k_space_output = k_space_output.permute(3, 4, 0, 1, 2, 5)  # arracnge the k-space to fit the raw-data order

        target_estimate, _, _, _ = simulator_class_MRI.lowfieldsim(k_space_output)  # feed the simulator with complex frequency data [in_width, in_height, 1,3, #coils (channels),complex (2)] the output is image [in_width, in_height, # coils (channels)]

        target_estimate = target_estimate.permute(2, 0, 1)  # Re-ordering to fit the model expected shape [# coils (channels),in_width, in_height]

        target_estimate = (target_estimate - torch.min(target_estimate)) / (torch.max(target_estimate) - torch.min(target_estimate))
        # Try to normalized the target_estimate with constants

        # loss function
        loss = F.l1_loss(target, target_estimate) # can also be F.mse_loss
        # loss = F.l1_loss(target, target_estimate)  + L2# can also be F.mse_loss


        output = output.squeeze(0)
        output = (output - torch.min(output)) / (torch.max(output) - torch.min(output))
        # Try to normalized the output with constants from the original low field raw data
        # Try normalized to Normal distribution


        # print(loss)
        actual_loss = F.l1_loss(output_gt, output)

        model.forward(input_mr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {loss:.4g} , Actual loss on HF: {actual_loss:.4g} '
        )
        if epoch % args.report_interval == 0:
            print(f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {loss:.4g} , Actual loss on HF:{actual_loss:.4g}')
            visualize(args, epoch, model, input_mr, target_estimate.unsqueeze(0), writer)
        writer.add_scalar('Low field Loss --#Unet-Channels {}, --lr ={}, --epochs - {}'.format(args.num_chans, args.num_epochs, args.lr), loss ,epoch)
        writer.add_scalar('High field Loss --#Unet-Channels {}, --lr ={}, --epochs - {}'.format(args.num_chans, args.num_epochs, args.lr), actual_loss, epoch)
    visualize(args, epoch, model, input_mr, target_estimate.unsqueeze(0), writer)
    save_model(args, args.exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best)
    writer.close()


    torch.save(output, os.path.join('High Field Target - epochs {} -- lr {} -- Unet_channels{}.pt'.format(args.num_epochs, args.lr, args.num_chans)))
    torch.save(target_estimate, os.path.join('Low Field Reconstruction - epochs {} -- lr {} -- Unet_channels{}.pt'.format(args.num_epochs, args.lr, args.num_chans)))
    # torch.save(model.forward(input_mr), args.test_name,'test.pt')

def visualize(args, epoch, model, input_mr,target_estimate, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        input = input_mr
        input = input.to(args.device)
        target = target_estimate.to(args.device)
        target = torch.sqrt(torch.sum(target, 1, keepdim=True))
        output = model(input)

        # Normalization overall not per channel
        output -= output.min()
        output /= output.max()

        output = torch.sqrt(torch.sum(output, 1, keepdim=True))
        save_image(target, 'Low Field Reconstruction')
        save_image(output, 'High Field Target')
        save_image(torch.abs(target - output), 'Error')

def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def create_arg_parser():
    parser = Args()

    # model parameters
    parser.add_argument('--num-pools', type=int, default=3, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=8, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=16, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')
    parser.add_argument('--momentum', type=float, default=0.,
                        help='Momentum factor')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')

    parser.add_argument('--num_coil', type=int, default=8,required=True, help='Number of image input  output channels')


    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)

    #Check the check-in feature on Git

#查看spasity是否是可以强固定学习ratio
import torch
import torch.nn as nn
from mri_tools import  ifft2,fft2

class Memc_LOUPE(nn.Module):
    def __init__(self, input_shape, slope, sample_slope, device, sparsity):
        super(Memc_LOUPE, self).__init__()
        # self.gt=gt
        self.input_shape = input_shape
        self.slope = slope  #、？？
        self.device = device
        self.add_weight_real = nn.Parameter(- torch.log(1. / torch.rand(self.input_shape, dtype=torch.float32) - 1.) / self.slope, requires_grad=True)
        self.add_weight_imag = nn.Parameter(- torch.log(1. / torch.rand(self.input_shape, dtype=torch.float32) - 1.) / self.slope, requires_grad=True)
        self.sample_slope = sample_slope
        self.sparsity = sparsity
        #sparsity定义为可学习的 目前用不到
        # self.sparsity = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.sparsity = sparsity
        # self.noise_val = noise_val
        self.conv = nn.Conv2d(4, 2, 1, 1, 0)

    def calculate_Mask(self, kspace_mc, option=True):
        print('sparsity:',self.sparsity)
        # kspace_mc=kspace_mc.unsqueeze(1)
        # kspace_mc=kspace_mc.repeat(1,2,1,1)  #torch.Size([1, 2, 256, 64])
        logit_weights_real = 0 * kspace_mc[:,0, :, :] + self.add_weight_real
        logit_weights_imag = 0 * kspace_mc[:,1, :, :] + self.add_weight_imag
        prob_mask_tensor = torch.cat((logit_weights_real, logit_weights_imag), dim=1)
        prob_mask_tensor = self.conv(prob_mask_tensor)
        prob_mask_tensor = torch.sigmoid(self.slope * prob_mask_tensor)

        xbar = torch.mean(prob_mask_tensor)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        le = (torch.less_equal(r, 1)).to(dtype=torch.float32)
        prob_mask_tensor = le * prob_mask_tensor * r + (1 - le) * (1 - (1 - prob_mask_tensor) * beta)

        # threshs = torch.rand_like(prob_mask_tensor)
        # thresh_tensor = 0 * prob_mask_tensor + threshs
        threshs = torch.rand(prob_mask_tensor.size(), dtype=torch.float32).to(device=self.device)
        thresh_tensor = 0 * prob_mask_tensor + threshs

        if option:
            last_tensor_mask = torch.sigmoid(self.sample_slope * (prob_mask_tensor - thresh_tensor))
        else:
            last_tensor_mask = (prob_mask_tensor > thresh_tensor) + 0   
        return last_tensor_mask.to(device=self.device)

    def forward(self,mask,option):
        B,C,H,W=mask.shape
        assert H==256 and W==256
        dcmask = self.calculate_Mask(mask, option=True)#inital work
        return  dcmask

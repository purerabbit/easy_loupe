
from .unet.unet_model import UNet#用此方式可以实现包的导入

#从不同文件夹下导入包
import torch
import torch.nn as nn
import torch.nn.functional as F
from mri_tools import  ifft2,fft2
from .cascade import CascadeMRIReconstructionFramework
from .memc_loupe import Memc_LOUPE
from .total_mask_loupe import TOTAL_LOUPE
from data.utils import *
import scipy.io as sio
#从不同文件夹下导入包

class ParallelNetwork(nn.Module):
   
    def __init__(self, num_layers, rank,slope,sample_slope,sparsity):
        super(ParallelNetwork, self).__init__()
        self.num_layers = num_layers
        self.rank = rank
          
        self.net = CascadeMRIReconstructionFramework(
            n_cascade=5  #the formor is 5
        )
        input_shape_one=[1,2,256,256]
        self.Memc_LOUPE_Model = Memc_LOUPE(input_shape_one, slope=slope, sample_slope=sample_slope, device=self.rank, sparsity=0.25)

    def forward(self,gt,option,mode):
        onemask=torch.ones_like(gt)
        if mode=='train':
            under_mask = self.Memc_LOUPE_Model(onemask)  #B H select_W

        else:
            under_mask = self.Memc_LOUPE_Model(onemask,option=False) 
        
        k0_recon=fft2(gt)*under_mask
        im_recon=ifft2(k0_recon)
        output_img=self.net(im_recon ,under_mask,k0_recon)
        return  output_img,under_mask




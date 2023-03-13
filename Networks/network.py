
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
        input_shape_two=[1,2,256,256]
        input_shape_one=[1,2,256,64]  #实现在采样到的部分进行学习
        print('net_sparsity:',sparsity)
        self.Memc_LOUPE_Model = Memc_LOUPE(input_shape_one, slope=slope, sample_slope=sample_slope, device=self.rank, sparsity=sparsity)
        self.TOTAL_LOUPE_Model = TOTAL_LOUPE(input_shape_two, slope, sample_slope, device=self.rank, sparsity_under=0.25,sparsity_select=0.6) #dc learn 0.6
    def get_submask(self,getmask):
        '''
        功能：得到采样到部分矩阵 以及对应坐标
        输入:模拟欠采的mask
        输出:onemask:对应到采样到部分矩阵大小的矩阵   b:采样到部分数据的坐标(tuple b[0]对应横坐标    b[1]对应纵坐标)
        '''
        a=getmask[:,0,:,:]  # B H W
        b=((a == 1).nonzero(as_tuple=True))  #b[0]对应的非零值行坐标  b[1]对应的非零值列坐标
        onemask=torch.ones(a.shape[0],a.shape[1],len(torch.unique(b[2]))).to(self.rank)
        return onemask,b  # onemask.shape->B H select_W     b.len->3

    def recovery_mask(self,mask,sub_mask,b):
        recomask=mask.clone()
        sub_real=sub_mask[:,0,:,:]
        sub_img=sub_mask[:,1,:,:]
        list_mask_real=sub_real.reshape(-1)
        list_mask_img=sub_img.reshape(-1)
        # n=0
        recomask[b[0],0,b[1],b[2]]=list_mask_real
        recomask[b[0],1,b[1],b[2]]=list_mask_img

        return recomask

    def forward(self,mask,gt,option,mode,gdc_mask,gloss_mask):
        
        if mode=='train':
            if option=='onemask':
                # print('onemask')
                onemask,b = self.get_submask(mask)
                sub_dc_mask = self.Memc_LOUPE_Model(onemask)  #B H select_W
                dc_mask = self.recovery_mask(mask,sub_dc_mask,b)
                loss_mask=mask-dc_mask
            elif option=="twomask":
                # print('twomask')
                kspace=fft2(gt)
                mask,dc_mask,loss_mask=self.TOTAL_LOUPE_Model(kspace)
            elif option=="baseline":
                # print('baseline')
                dc_mask=gdc_mask
                loss_mask=gloss_mask

        else:
            if option=='twomask':
                kspace=fft2(gt)
                mask,dc_mask,loss_mask=self.TOTAL_LOUPE_Model(kspace,False) #保证test under_mask为0-1

            loss_mask=dc_mask=mask
        
        k0_recon=fft2(gt)*dc_mask
        im_recon=ifft2(k0_recon)
        output_img=self.net(im_recon ,dc_mask,k0_recon)
        return  output_img,loss_mask,dc_mask,mask




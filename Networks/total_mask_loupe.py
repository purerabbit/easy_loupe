

#查看spasity是否是可以强固定学习ratio
import torch
import torch.nn as nn
from mri_tools import  ifft2,fft2
from .memc_loupe import Memc_LOUPE

class TOTAL_LOUPE(nn.Module):
    def __init__(self, input_shape, slope, sample_slope, device, sparsity_under=0.25,sparsity_select=0.6):
        super(TOTAL_LOUPE, self).__init__()  #这句话是否有用?
        self.sample_slope=sample_slope
        self.sparsity_under=sparsity_under
        self.sparsity_select=sparsity_select
        self.device=device
        self.under_Memc_LOUPE_Model = Memc_LOUPE(input_shape, slope=slope, sample_slope=sample_slope, device=device, sparsity=sparsity_under)
        self.select_Memc_LOUPE_Model = Memc_LOUPE(input_shape, slope=slope, sample_slope=sample_slope, device=device, sparsity=sparsity_select)
    
    def normalsig(self,mask_under,init_mask,sample_slope_init,sparsity_init):
        
        xbar = torch.mean(init_mask)
        r = sparsity_init / xbar
        beta = (1 - sparsity_init) / (1 - xbar)
        le = (torch.less_equal(r, 1)).to(dtype=torch.float32)
        init_mask = le * init_mask * r + (1 - le) * (1 - (1 - init_mask) * beta) #dc mask归一化 实现数据量占总数据的0.6*0.25

        threshs = torch.rand(init_mask.size(), dtype=torch.float32).to(device=self.device)
        thresh_tensor = 0 * init_mask + threshs
        loss_mask = mask_under-init_mask #实现loss mask占总数据量的0.25-0.6*0.25=0.4*0.25
        #将mask归一化 使得数值尽可能接近0 1
        last_mask_dc = torch.sigmoid(sample_slope_init * (init_mask - thresh_tensor))
        last_mask_loss = torch.sigmoid(sample_slope_init * (loss_mask - thresh_tensor))
        
        return last_mask_dc,last_mask_loss
    
    def forward(self,kspace,option=True):

        onemask=torch.ones_like(kspace)
        mask_under=self.under_Memc_LOUPE_Model(onemask,option=option) #加入option 确保test mask为0 1
        mask_select=self.select_Memc_LOUPE_Model(onemask,option=option)
        mask=mask_under   #0-1
        mask_dc=mask*mask_select  #0-1 #预期选0.6的数据量，接下来用正则化和sigmoid函数进行约束 采样数据点一定保持一致，采样数据量不一定保持一致   
        
        sparsity_dc = self.sparsity_select*self.sparsity_under
        # sparsity_loss = self.sparsity_select*(1-self.sparsity_under)
        #归一化和sigmoid
        mask_dc,mask_loss=self.normalsig(mask,mask_dc,self.sample_slope,sparsity_dc)


        return mask,mask_dc,mask_loss    
    
    # 用原始mask进行筛选（相乘）由于起初分给dc的数据量过少，难以训练 
    # 不同mask的参数可能需要调节

    # def forward(self,kspace):

    #     onemask=torch.ones_like(kspace)
    #     mask_under=self.under_Memc_LOUPE_Model(onemask)
    #     mask_select=self.select_Memc_LOUPE_Model(onemask)
        
    #     mask=mask_under   #0-1
    #     mask_dc=mask*mask_select  #0-1
    #     mask_loss=mask-mask_dc  #0-1

    #     return mask,mask_dc,mask_loss
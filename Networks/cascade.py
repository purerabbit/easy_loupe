#共享参数部分

import torch
import torch.nn as nn
from data.utils import *
from .unet.unet_model import UNet#用此方式可以实现包的导入
import sys
sys.path.append("/home/liuchun/Desktop/add_dc/dual_domain/Networks")
from .dudor_gen import get_generator
class DataConsistencyLayer(nn.Module):
    """
    This class support different types k-space data consistency
    """

    def __init__(self, is_data_fidelity=True):
        super().__init__()
        self.is_data_fidelity = is_data_fidelity

    def forward(self, im_recon, mask, k0):
        """
        set is_data_fidelity=True to complete the formulation
        
        :param k: input k-space (reconstructed kspace, 2D-Fourier transform of im) complex
        :param k0: initially sampled k-space complex
        :param mask: sampling pattern
        """
        
        k=image2kspace(pseudo2complex(im_recon))
        k=complex2pseudo(k)
        k_dc = (1 - mask) * k + mask * k0
        im_dc = complex2pseudo(kspace2image(pseudo2complex(k_dc)))  # [B, C=2, H, W]   

        return im_dc


class CascadeMRIReconstructionFramework(nn.Module):
    def __init__(self,  n_cascade: int):
        super().__init__()
        self.cnn = get_generator('DRDN')######使用新的
        self.n_cascade = n_cascade

        assert n_cascade > 0
        dc_layers = [DataConsistencyLayer() for _ in range(n_cascade)]
        self.dc_layers = nn.ModuleList(dc_layers)

    def forward(self,im_init, mask,k0):
        im_recon=im_init
        B, C, H, W = im_recon.shape
        B, C, H, W = k0.shape
        assert C == 2
        assert (B,C, H, W) == tuple(mask.shape)

        for dc_layer in self.dc_layers:
            im_recon = self.cnn(im_recon)
            im_recon = pseudo2real(im_init)+im_recon
            im_recon = dc_layer(im_recon, mask, k0)
        return im_recon


    # def forward(self,im_recon, mask,k0):
    #     B, C, H, W = im_recon.shape
    #     B, C, H, W = k0.shape
    #     assert C == 2
    #     assert (B,C, H, W) == tuple(mask.shape)
   
    #     for dc_layer in self.dc_layers:
    #         im_recon=self.cnn(im_recon)
                
    #         im_recon = dc_layer(im_recon, mask, k0)
    #     return im_recon



# #不共享参数部分
# import torch
# import torch.nn as nn
# from data.utils import *
# from .unet.unet_model import UNet#用此方式可以实现包的导入
# import sys
# sys.path.append("/home/liuchun/Desktop/add_dc/dual_domain/Networks")
# from .dudor_gen import get_generator
# class DataConsistencyLayer(nn.Module):
#     """
#     This class support different types k-space data consistency
#     """

#     def __init__(self, is_data_fidelity=True):
#         super().__init__()
#         self.is_data_fidelity = is_data_fidelity
        

#         # if is_data_fidelity:
#         #     self.data_fidelity = nn.Parameter(torch.tensor(1.0,dtype=torch.float32))
#             # self.data_fidelity = nn.Parameter(torch.ones((1,1),dtype=torch.float32))

#     def forward(self, im_recon, mask, k0):
#         """
#         set is_data_fidelity=True to complete the formulation
        
#         :param k: input k-space (reconstructed kspace, 2D-Fourier transform of im) complex
#         :param k0: initially sampled k-space complex
#         :param mask: sampling pattern
#         """
#         k=image2kspace(pseudo2complex(im_recon))
#         k=complex2pseudo(k)
#         k_dc = (1 - mask) * k + mask * k0 
#         im_dc = complex2pseudo(kspace2image(pseudo2complex(k_dc)))  # [B, C=2, H, W]   
       

#         return im_dc


# class CascadeMRIReconstructionFramework(nn.Module):
#     def __init__(self,  n_cascade: int):
#         super().__init__()
#         self.n_cascade = n_cascade

#         assert n_cascade > 0
#         dc_layers = [DataConsistencyLayer() for _ in range(n_cascade)]
#         self.dc_layers = nn.ModuleList(dc_layers)
#         net = [get_generator("DRDN") for _ in range(n_cascade)]
#         self.net = nn.ModuleList(net)

#     def forward(self,im_recon, mask,k0):
#         B, C, H, W = im_recon.shape
#         B, C, H, W = k0.shape
#         assert C == 2
#         assert (B,C, H, W) == tuple(mask.shape) 
#         for i in range(self.n_cascade):
#             cnn = self.net[i]
#             dc = self.dc_layers[i]   
#             im_recon=cnn(im_recon)     
#             im_recon = dc(im_recon, mask, k0)
#         return im_recon
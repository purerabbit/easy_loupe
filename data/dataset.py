
import random
import pathlib
import scipy.io as sio
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from .utils import normalize_zero_to_one,kspace2image, image2kspace, complex2pseudo, pseudo2real, pseudo2complex
from torch.utils import data as Data 

def build_loader(dataset, batch_size,is_shuffle,    
                 num_workers=4):
    loader=Data.DataLoader(dataset, batch_size=batch_size , shuffle=is_shuffle, num_workers=num_workers)
    return loader

class IXIData(Dataset):
    def __init__(self, data_path, u_mask_path, s_mask_up_path, s_mask_down_path):
        super(IXIData, self).__init__()
        self.data_path = data_path
        self.u_mask_path = u_mask_path
        self.s_mask_up_path = s_mask_up_path
        self.s_mask_down_path = s_mask_down_path
        # self.s_mask_over_path = '/home/liuchun/Desktop/ovlm_parallel_02/mask/selecting_mask/mask_over.mat'

        self.examples = []
        
        data_dict = np.load(data_path)
        # loading dataset
        kspace = data_dict['kspace']  # List[ndarray]
        self.images=kspace2image(kspace)
        self.images=complex2pseudo(self.images)
        self.examples = self.images.astype(np.float32)

        self.mask_under = np.array(sio.loadmat(self.u_mask_path)['mask'])
        self.s_mask_up = np.array(sio.loadmat(self.s_mask_up_path)['mask'])
        self.s_mask_down = np.array(sio.loadmat(self.s_mask_down_path)['mask'])
        #new
        # self.over_mask = np.array(sio.loadmat(self.s_mask_over_path)['data'])

        self.mask_net_up = self.mask_under * self.s_mask_up
        self.mask_net_down = self.mask_under * self.s_mask_down

        self.mask_under = np.stack((self.mask_under, self.mask_under), axis=-1)
        self.mask_under = torch.from_numpy(self.mask_under).float()
        self.mask_net_up = np.stack((self.mask_net_up, self.mask_net_up), axis=-1)
        self.mask_net_up = torch.from_numpy(self.mask_net_up).float()
        self.mask_net_down = np.stack((self.mask_net_down, self.mask_net_down), axis=-1)
        self.mask_net_down = torch.from_numpy(self.mask_net_down).float()
        #new
        # self.over_mask = np.stack((self.over_mask, self.over_mask), axis=-1)
        # self.over_mask = torch.from_numpy(self.over_mask).float()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        label = self.examples[item]
        label = normalize_zero_to_one(label, eps=1e-6)  #归一化部分 是否需要额外进行归一化？ 
        label = torch.from_numpy(label)
        '''mask_up=mask_dc   mask_down=mask_loss'''
        return label, self.mask_under,self.mask_net_up, self.mask_net_down #, file.name, slice_id








'''
以下是实现用1D降采的SSDU代码，上面是parallel的结果

'''





# """
# BME1301
# DO NOT MODIFY anything in this file.
# """
# import matplotlib.pyplot as plt
# from typing import Sequence, List, Union

# import numpy as np
# from numpy.lib.stride_tricks import as_strided
# import torch
# from torch.utils import data as Data
 
# # from data.utils import kspace2image, image2kspace, complex2pseudo, pseudo2real, pseudo2complex
# from .utils import kspace2image, image2kspace, complex2pseudo, pseudo2real, pseudo2complex
# import sys
# import os
# sys.path.append("..")
# from mask.main_mask import uf_mask
# from mask.gen_mask import uniform_selection

# # =============================================================================
# # Utility
# # =============================================================================

# def build_loader(dataset, batch_size,is_shuffle,    
#                  num_workers=4):
#     loader=Data.DataLoader(dataset, batch_size=batch_size , shuffle=is_shuffle, num_workers=num_workers)
#     return loader


# def cartesian_mask(shape, acc, sample_n=24, centred=False):
#     """
#     Sampling density estimated from implementation of kt FOCUSS
#     shape: tuple - (Nslice, Nx, Ny, Ntime)
#     acc: float - doesn't have to be integer 4, 8, etc..
#     """
#     shape=(1,256,256,1)#指定生成尺寸
 
#     mask=uf_mask(shape,center_nums=sample_n,accelerations=acc)
#     mask=mask[0,:,:,0]
   

#     return mask #256 256


# def np_undersample(k0, mask_centered):
#     """
#     input: k0 (H, W), mask_centered (H, W)
#     output: x_u, k_u (H, W)  complex
#     """
#     assert k0.shape == mask_centered.shape
#     # assert k0.dtype == torch.Tensor
#     # print('k0.dtype:',k0.dtype)
#     if isinstance(k0, torch.Tensor):
#         k0 = k0.type(torch.complex64)
#     else:
#         k0 = k0.astype(np.complex64)
#     # print('after-k0.dtype:',k0.dtype)

#     k_u = k0 * mask_centered
#     x_u = kspace2image(k_u)
#     if isinstance(x_u,torch.Tensor):
#         x_u = x_u.type(torch.complex64)
#     else:
#         x_u=x_u.astype(np.complex64)

#     if isinstance(k_u,torch.Tensor):
#         k_u = k_u.type(torch.complex64)
#     else:
#         k_u=k_u.astype(np.complex64)
#     # k_u = k_u.dtype(torch.complex64)
#     return x_u, k_u


# # =============================================================================
# # Dataset
# # =============================================================================
# class FastmriKnee(Data.Dataset):
#     def __init__(self, path: str):
#         """
#         :param augment_fn: perform augmentation on image data [C=2, H, W] if provided.
#         """
#         data_dict = np.load(path)
#         # loading dataset
#         kspace = data_dict['kspace']  # List[ndarray]
#         # viz_indices = data_dict['vis_indices']  # List[int]

#         # preprocessing
#         images = kspace2image(kspace).astype(np.complex64)  # [1000, Nxy, Nxy] complex64
#         images = complex2pseudo(images)  # convert to pseudo-complex representation
#         self.images = images.astype(np.float32)  # [1000, 2, Nxy, Nxy] float32
#         # self.viz_indices = viz_indices.astype(np.int64)  # [N,] int64

#         # inferred parameter
#         self.n_slices = self.images.shape[0]

#     def __getitem__(self, idx):
#         im_gt = self.images[idx]
#         return im_gt  # [2, Nxy, Nxy] float32

#     def __len__(self):
#         # print('self.n_slices-FastmriKnee:',self.n_slices)
#         return self.n_slices


# class DatasetReconMRI(Data.Dataset):
#     def __init__(self, dataset: Data.Dataset, acc=4.0, num_center_lines=24, augment_fn=None):
#         """
#         :param augment_fn: perform augmentation on image data [C=2, H, W] if provided.
#         """
#         self.dataset = dataset

#         # inferred parameter
#         self.n_slices = len(dataset)

#         # parameter for undersampling
#         self.acc = acc
#         self.num_center_lines = num_center_lines
#         self.augment_fn = augment_fn

#     def __getitem__(self, idx):
#         im_gt = self.dataset[idx]  # [2, Nxy, Nxy] float32

#         if self.augment_fn:
#             im_gt = self.augment_fn(im_gt)  # [2, Nxy, Nxy] float32

#         C, H, W = im_gt.shape
#         und_mask = cartesian_mask(shape=(1, H, W, 1), acc=self.acc, sample_n=self.num_center_lines, centred=True
#                                   ).astype(np.float32)#[0, :, :, 0]  # [H, W]
#         k0 = image2kspace(pseudo2complex(im_gt))
        
#         ks=k0[:,:,np.newaxis]
#         select_mask_up,select_mask_down=uniform_selection(ks,und_mask)
   
#         k0=k0.astype(np.complex64) #最后的float32 约束 不会影响网络训练
#         x_und, k_und = np_undersample(k0, und_mask)
        
#         EPS = 1e-8
#         #欠采样之后又进行了归一化  这一步可以去掉  
#         x_und_abs = np.abs(x_und)
#         # norm_min = x_und_abs.min()
#         norm_max = x_und_abs.max()
#         norm_scale = norm_max  # - norm_min + EPS
#         x_und = x_und / norm_scale
#         im_gt = im_gt / norm_scale

#         k_und = image2kspace(x_und)  # [H, W] Complex
#         k_und = complex2pseudo(k_und)  # [C=2, H, W]
#         return (
#             k_und.astype(np.float32),  # [C=2, H, W]
#             und_mask.astype(np.float32),  # [H, W]
#             im_gt.astype(np.float32),  # [C=2, H, W]
#             select_mask_up.astype(np.float32),
#             select_mask_down.astype(np.float32)
#         )

#     def __len__(self):
#         return self.n_slices

import torch
import torch.fft
from data.utils import complex2pseudo,pseudo2complex

def fftshift(x, axes=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(axes, int):
        shift = x.shape[axes] // 2
    else:
        shift = [x.shape[axis] // 2 for axis in axes]
    return torch.roll(x, shift, axes)


def ifftshift(x, axes=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    assert torch.is_tensor(x) is True
    if axes is None:
        axes = tuple(range(x.ndim()))
        shift = [-(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -(x.shape[axes] // 2)
    else:
        shift = [-(x.shape[axis] // 2) for axis in axes]
    return torch.roll(x, shift, axes)


def fft2(x):
    assert x.shape[-3] == 2 #输入的都应该是伪复数 输出也是伪复数
    x=pseudo2complex(x)
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.fft2(x)
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x=complex2pseudo(x)
    return x


def ifft2(x):
    assert x.shape[-3] == 2 #输入的都应该是伪复数   
    x=pseudo2complex(x)  
    x = torch.fft.ifftshift(x, dim=(-2, -1))
    x = torch.fft.ifft2(x)
    x = torch.fft.fftshift(x, dim=(-2, -1))
    x=complex2pseudo(x)
    return x

#1 channel数据 产生模拟虚部
def rfft2(data):
    assert data.shape[-1] == 1
    data = torch.cat([data, torch.zeros_like(data)], dim=-1)
    data = fft2(data)
    return data

#去掉模拟虚部 只保留实数部分
def rifft2(data):
    assert data.shape[-1] == 2
    data = ifft2(data)
    data = data[..., 0].unsqueeze(-1) #只取实部
    return data

#输入kspace 输出加上mask的kspace
def rA(data, mask):
    assert data.shape[-3] == 2
    data = fft2(data) * mask
    return data

#输入kspace  输出加上mask的image
def rAt(data, mask):
    assert data.shape[-3] == 2
    data = ifft2(data * mask)
    return data

#输入图像 输出加上mask的图像
def rAtA(data, mask):
    assert data.shape[-3] == 2#bcwh
 
    data = fft2(data)* mask
    data = ifft2(data)
    return data





















# import torch
# import torch.fft


# '''
# fft2c_new
# input: data[b,w,h,c]  c=2
# output:The FFT of the input
# '''

# '''
# ifft2c_new
# input:data[b,w,h,c]  c=2
# output:The IFFT of the input
# '''

# def fftshift(x, axes=None):
#     """
#     Similar to np.fft.fftshift but applies to PyTorch Tensors
#     """
#     assert torch.is_tensor(x) is True
#     if axes is None:
#         axes = tuple(range(x.ndim()))
#         shift = [dim // 2 for dim in x.shape]
#     elif isinstance(axes, int):
#         shift = x.shape[axes] // 2
#     else:
#         shift = [x.shape[axis] // 2 for axis in axes]
#     return torch.roll(x, shift, axes)


# def ifftshift(x, axes=None):
#     """
#     Similar to np.fft.ifftshift but applies to PyTorch Tensors
#     """
#     assert torch.is_tensor(x) is True
#     if axes is None:
#         axes = tuple(range(x.ndim()))
#         shift = [-(dim // 2) for dim in x.shape]
#     elif isinstance(axes, int):
#         shift = -(x.shape[axes] // 2)
#     else:
#         shift = [-(x.shape[axis] // 2) for axis in axes]
#     return torch.roll(x, shift, axes)


# def fft2(data):
#     assert data.shape[-1] == 2
#     data = torch.view_as_complex(data)
#     data = torch.fft.fftn(data, dim=(-2, -1), norm='ortho')
#     data = torch.view_as_real(data)
#     data = fftshift(data, axes=(-3, -2))
#     return data


# def ifft2(data):
#     #output-shape:[1,256,256,2]
#     assert data.shape[-1] == 2
#     data = ifftshift(data, axes=(-3, -2))
#     data = torch.view_as_complex(data)
#     data = torch.fft.ifftn(data, dim=(-2, -1), norm='ortho')
#     data = torch.view_as_real(data)
    
#     return data


# def rfft2(data):
#     assert data.shape[-1] == 2
#     # data = torch.cat([data, torch.zeros_like(data)], dim=-1)
#     data = fft2(data)
#     return data


# def rifft2(data):
#     assert data.shape[-1] == 2
#     data = ifft2(data)
#     # data = data[..., 0].unsqueeze(-1)
#     return data

# #the mask should be 2 channel

# # def fft2(data):
# #     return fft2c_new(data) 

# # def ifft2(data):
# #     return ifft2c_new(data)

# def rA(data, mask):
#     assert data.shape[-1] == 2
#     # data = torch.cat([data, torch.zeros_like(data)], dim=-1)
#     data = fft2(data) * mask
#     return data


# def rAt(data, mask):
#     assert data.shape[-1] == 2
#     data = ifft2(data * mask)
#     # data = data[..., 0].unsqueeze(-1)
#     return data


# def rAtA(data, mask):
#     # print('rata-data.shape:',data.shape)#rata-data.shape: torch.Size([3, 2, 256, 256])
#     # data=data.permute(0,2,3,1)#[1,256,256,2]
#     # print('rata-mask,shape:',mask.shape)
#     assert data.shape[-1] == 2#channel
#     # data = torch.cat([data, torch.zeros_like(data)], dim=-1)
#     data = fft2(data) * mask
#     data = ifft2(data)
#     # data = data[..., 0].unsqueeze(-1)
#     return data

 
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
 
import matplotlib.pyplot as plt
import math
import numpy as np
import torch
import torchvision.utils
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
 
from skimage.color import rgb2gray
import os
 
 
def get_mask(u_mask_path):
    former_path_select='/home/liuchun/Desktop/experment02/mask/selecting_mask/' #选择mask路径前缀
    former_path_under='/home/liuchun/Desktop/experment02/mask/undersampling_mask/' #欠采mask路径前缀
    #简化传入参数
    if(u_mask_path=='vd'):
        u_mask_path=former_path_under+'vd_mask_under.mat'
        s_mask_up_path=former_path_select+'mask_dc_vd.mat'
        s_mask_down_path=former_path_select+'mask_loss_vd.mat'
    elif(u_mask_path=='p'):
        u_mask_path=former_path_under+'mask_4.00x_acs24.mat'
        s_mask_up_path=former_path_select+'mask_dc.mat'
        s_mask_down_path=former_path_select+'mask_loss.mat'
    # elif(u_mask_path=='c'):
    #     u_mask_path=former_path_under+'cartesian_mask_under.mat'
    #     s_mask_up_path=former_path_select+'cartesian_mask_up.mat'
    #     s_mask_down_path=former_path_select+'cartesian_mask_down.mat'

    # elif(u_mask_path=='r'):
    #     u_mask_path=former_path_under+'random_mask_under.mat'
    #     s_mask_up_path=former_path_select+'random_mask_up.mat'
    #     s_mask_down_path=former_path_select+'random_mask_down.mat'

    # else:
    #     u_mask_path='/home/liuchun/Desktop/0_experiment/mask/undersampling_mask/mask_4.00x_acs24.mat'
    #     s_mask_up_path='/home/liuchun/Desktop/0_experiment/mask/selecting_mask/mask_2.00x_acs16.mat'
    #     s_mask_down_path='/home/liuchun/Desktop/0_experiment/mask/selecting_mask/mask_2.50x_acs16.mat'
    return u_mask_path,s_mask_up_path,s_mask_down_path
# under_img,net_img_up,net_img_down,output_up,*output_down*,mask_under,mask_net_up,mask_net_down,*out_mean*,label,iter_num,path,mode
# def save_images(under_img,net_img_up,net_img_down,output_up,output_down,mask_under,mask_net_up,mask_net_down,out_mean,label,iter_num,path,mode):
def save_images(under_img,output_img,under_mask,label,iter_num,path,mode):
# def save_images(under_img,net_img_up,net_img_down,output_up,mask_under,mask_net_up,mask_net_down,label,iter_num,path,mode):
        img_show=torch.cat((pseudo2real(under_img),pseudo2real(output_img),\
                    pseudo2real(under_mask) ,pseudo2real(label)),0) 
        psnr_under=compute_psnr_q(pseudo2real(label),pseudo2real(under_img))
        ssim_under=compute_ssim(pseudo2real(label),pseudo2real(under_img))
        psnr_up=compute_psnr_q(pseudo2real(label),pseudo2real(output_img))
        ssim_up=compute_ssim(pseudo2real(label),pseudo2real(output_img))

        ratio_mask_under_real=torch.sum(mask_under[0,0,:,:])/(256*256)
        ratio_mask_under_imag=torch.sum(mask_under[0,1,:,:])/(256*256)

        if mode=='train':
            filename2save=f'/home/liuchun/Desktop/easylouperecon/train_save/{path}'
        elif mode=='test':
            filename2save=f'/home/liuchun/Desktop/easylouperecon/test_save/{path}'
        print('--------------filename2save--------:',filename2save)
        if not os.path.exists(filename2save):
            os.makedirs(filename2save)
        imsshow(img_show.data.cpu().numpy(),['underssim: {:.3f} underpsnr: {:.3f}'.format(ssim_under, psnr_under),\
                                            'sussim-in: {:.3f} supsnr-in: {:.3f}'.format(ssim_up, psnr_up),\
                                            'under-real:{:.3f} under-imag:{:.3f}'.format(ratio_mask_under_real, ratio_mask_under_imag),'label'],ncols=2,cmap='gray',is_colorbar=True,filename2save=f'{filename2save}/{iter_num}.png')
        

def center_crop(data, shape):
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def normalize_zero_to_one(data, eps=0.):
    data_min = float(data.min())
    data_max = float(data.max())
    return (data - data_min) / (data_max - data_min + eps)


#####################################################################

def plot_loss(loss):
    plt.figure()
    plt.plot(loss)
    plt.show()
    plt.close('all')


def imgshow(im, cmap=None, rgb_axis=None, dpi=1000, figsize=(6.4, 4.8)):
    if isinstance(im, torch.Tensor):
        im = im.to('cpu').detach().cpu().numpy()
    if rgb_axis is not None:
        im = np.moveaxis(im, rgb_axis, -1)
        im = rgb2gray(im)

    plt.figure(dpi=dpi, figsize=figsize)
    norm_obj = Normalize(vmin=im.min(), vmax=im.max())
    plt.imshow(im, norm=norm_obj, cmap=cmap)
    plt.colorbar()
    plt.show()
    plt.close('all')

def imsshow(imgs, titles=None, ncols=5, dpi=1000, cmap=None, is_colorbar=False, is_ticks=False,
            col_width=4, row_width=3, margin_ratio=0.1, n_images_max=50, filename2save=None, **imshow_kwargs):
    '''
    assume imgs is Sequence[ndarray[Nx, Ny]]
    '''
    num_imgs = len(imgs)

    if num_imgs > n_images_max:
        print(
            f"[WARNING] Too many images ({num_imgs}), clip to argument n_images_max({n_images_max}) for performance reasons.")
        imgs = imgs[:n_images_max]
        num_imgs = n_images_max

    if isinstance(cmap, list):
        assert len(cmap) == len(imgs)
    else:
        cmap = [cmap, ] * num_imgs

    nrows = math.ceil(num_imgs / ncols)

    # compute the figure size, compute necessary size first, then add margin
    figsize = (ncols * col_width, nrows * row_width)
    figsize = (figsize[0] * (1 + margin_ratio), figsize[1] * (1 + margin_ratio))
    fig = plt.figure(dpi=dpi, figsize=figsize)
    for i in range(num_imgs):
        ax = plt.subplot(nrows, ncols, i + 1)
        im = ax.imshow(imgs[i], cmap=cmap[i], **imshow_kwargs)
        if titles:
            plt.title(titles[i])
        if is_colorbar:
            cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.01, ax.get_position().height])
            plt.colorbar(im, cax=cax)
        if not is_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
    if filename2save is not None:
        fig.savefig(filename2save)
    else:
        plt.show()
    plt.close('all')



def make_grid_and_show(ims, nrow=5, cmap=None):
    if isinstance(ims, np.ndarray):
        ims = torch.from_numpy(ims)

    B, C, H, W = ims.shape
    grid_im = torchvision.utils.make_grid(ims, nrow=nrow)
    fig_h, fig_w = nrow * 2 + 1, (B / nrow) + 1
    imgshow(grid_im, cmap=cmap, rgb_axis=0, dpi=200, figsize=(fig_h, fig_w))


def int2preetyStr(num: int):
    s = str(num)
    remain_len = len(s)
    while remain_len - 3 > 0:
        s = s[:remain_len - 3] + ',' + s[remain_len - 3:]
        remain_len -= 3
    return s


def compute_num_params(module, is_trace=False):
    print(int2preetyStr(sum([p.numel() for p in module.parameters()])))
    if is_trace:
        for item in [f"[{int2preetyStr(info[1].numel())}] {info[0]}:{tuple(info[1].shape)}"
                     for info in module.named_parameters()]:
            print(item)


def tonp(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    else:
        return x


def pseudo2real(x):
    """
    :param x: [..., C=2, H, W]
    :return: [..., H, W]
    """
    return (x[..., 0, :, :] ** 2 + x[..., 1, :, :] ** 2) ** 0.5


def complex2pseudo(x):
    """
    :param x: [..., H, W] Complex
    :return: [...., C=2, H, W]
    """
    if isinstance(x, np.ndarray):
        return np.stack([x.real, x.imag], axis=-3)
    elif isinstance(x, torch.Tensor):
        return torch.stack([x.real, x.imag], dim=-3)
    else:
        raise RuntimeError("Unsupported type.")


def pseudo2complex(x):
    """
    :param x:  [..., C=2, H, W]
    :return: [..., H, W] Complex
    """
    return x[..., 0, :, :] + x[..., 1, :, :] * 1j


# ================================
# Preprocessing
# ================================
def minmax_normalize(x, eps=1e-8):
    min = x.min()
    max = x.max()
    return (x - min) / (max - min + eps)


# ================================
# kspace and image domain transform
# reference: [ismrmrd-python-tools/transform.py at master · ismrmrd/ismrmrd-python-tools · GitHub](https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/ismrmrdtools/transform.py)
# ================================
def image2kspace(x):
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.fft2(x)
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.fft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
    else:
        raise RuntimeError("Unsupported type.")


def kspace2image(x):
    
    if isinstance(x, np.ndarray):
        x = np.fft.ifftshift(x, axes=(-2, -1))
        x = np.fft.ifft2(x)
        x = np.fft.fftshift(x, axes=(-2, -1))
        return x
    elif isinstance(x, torch.Tensor):
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.ifft2(x)
        x = torch.fft.fftshift(x, dim=(-2, -1))
        return x
    else:
        raise RuntimeError("Unsupported type.")


# ======================================
# Metrics
# ======================================
def compute_mse(x, y):
    """
    REQUIREMENT: `x` and `y` can be any shape, but their shape have to be same
    """
    assert x.dtype == y.dtype and x.shape == y.shape, \
        'x and y is not compatible to compute MSE metric'

    if isinstance(x, np.ndarray):
        mse = np.mean(np.abs(x - y) ** 2)

    elif isinstance(x, torch.Tensor):
        mse = torch.mean(torch.abs(x - y) ** 2)

    else:
        raise RuntimeError(
            'Unsupported object type'
        )
    return mse

def compute_psnr_q(gt, pred, maxval=None):
    # gt1=minmax_normalize(gt)
    # pred1=minmax_normalize(pred)
    assert type(gt) == type(pred)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    max_val = gt.max() if maxval is None else maxval
    PSNR = peak_signal_noise_ratio(gt, pred, data_range=max_val)  
    return PSNR

def compute_ssim(pred, gt):
    # gt=minmax_normalize(gt2)
    # pred=minmax_normalize(pred2)
    if type(pred) is torch.Tensor:
        gt, pred = gt.detach().cpu().numpy(), pred.detach().cpu().numpy()
    # target_im=minmax_normalize(target_im)
    # reconstructed_im=minmax_normalize(reconstructed_im)
    ssim=structural_similarity(gt.squeeze(), pred.squeeze(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return ssim 

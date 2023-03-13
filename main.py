# System / Python
# import pandas as pd
import os
import argparse
import logging
import random
import shutil
import time
import numpy as np
from tqdm import tqdm
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
# from torch.utils.data import DataLoader
# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler#并行网路
# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
# Custom
from Networks.network import ParallelNetwork as Network#导入网络
# from IXI_dataset import IXIData as Dataset
# from mri_tools import rA, rAtA, rfft2
from data.utils import pseudo2real,complex2pseudo,pseudo2complex,image2kspace,kspace2image,save_images
from metrics import compute_psnr,compute_ssim
# from preprocessing import *
from mask.gen_mask import *
#导入fastmri数据集文件 导入方式可能有问题
# from .data.dataset import FASTMRIDataset as Dataset#引用包问题
# from PIL import Image
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
# from utils import *
from data.utils import get_mask
from data.dataset import IXIData as Dataset
from data.dataset import build_loader
from loss import cal_loss

from data.ploting import imsshow

results_save_path='./results'

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#单GPU进行训练

parser = argparse.ArgumentParser()
#save images
parser.add_argument('--strain', '-strain', type=str, default=None, help='file_name_train')
parser.add_argument('--vtrain', '-vtrain', type=str, default=None, help='file_name_train')
parser.add_argument('--stest', '-stest', type=str, default=None, help='file_name_test')
parser.add_argument('--dc_ratio','-dc_ratio', type=float, default=0.25, help='file_name_test')

parser.add_argument('--exp-name', type=str, default='self-supervised MRI reconstruction', help='name of experiment')
# parameters related to distributed training
parser.add_argument('--init-method', default=f'tcp://localhost:{np.random.randint(1000,2000)}', help='initialization method')
# parser.add_argument('--init-method', default=f'tcp://localhost:1883', help='initialization method')
parser.add_argument('--nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--gpus', type=int, default=1, help='number of gpus per node')#放到一块GPU上进行训练
parser.add_argument('--world-size', type=int, default=None, help='world_size = nodes * gpus')
# parameters related to model
parser.add_argument('--use-init-weights', '-uit', type=bool, default=True, help='whether initialize model weights with defined types')
parser.add_argument('--init-type', type=str, default='xavier', help='type of initialize model weights')
parser.add_argument('--gain', type=float, default=1.0, help='gain in the initialization of model weights')
parser.add_argument('--num-layers', type=int, default=9, help='number of iterations')
# learning rate, batch size, and etc
parser.add_argument('--seed', type=int, default=30, help='random seed number')
parser.add_argument('--lr', '-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=1, help='batch size of single gpu')

parser.add_argument('--num-workers', type=int, default=8, help='number of workers')
parser.add_argument('--warmup-epochs', type=int, default=10, help='number of warmup epochs')
parser.add_argument('--num-epochs', type=int, default=500, help='maximum number of epochs')
# parameters related to data and masks
parser.add_argument('--train-path', type=str, default='/home/liuchun/Desktop/experment02/data/PD_train_852(71).npz', help='path of training data')
parser.add_argument('--val-path', type=str, default='/home/liuchun/Desktop/experment02/data/PD_val_252(21).npz', help='path of validation data')
parser.add_argument('--test-path', type=str, default='/home/liuchun/Desktop/experment02/data/PD_test_252(21).npz', help='path of test data')
#改变了数据集路径      u_mask_path
'''different mask of baseine center big or small  vd=big p=small'''
parser.add_argument('--u-mask-path', '-select_mask', type=str, default='vd', help='undersampling mask')
# parser.add_argument('--u-mask-path', type=str, default='/home/liuchun/Desktop/experment02/mask/undersampling_mask/vd_mask_under.mat', help='undersampling mask')
# #欠采样问题
# parser.add_argument('--s-mask-up-path', type=str, default='/home/liuchun/Desktop/experment02/mask/selecting_mask/vd_mask_up.mat', help='selection mask in up network')
# parser.add_argument('--s-mask-down-path', type=str, default='/home/liuchun/Desktop/experment02/mask/selecting_mask/vd_mask_down.mat', help='selection mask in down network')
parser.add_argument('--train-sample-rate', '-trsr', type=float, default=0.06, help='sampling rate of training data')
parser.add_argument('--val-sample-rate', '-vsr', type=float, default=0.02, help='sampling rate of validation data')
parser.add_argument('--test-sample-rate', '-tesr', type=float, default=0.02, help='sampling rate of test data')
# save path
# parser.add_argument('--model-save-path', type=str, default='./checkpoints/', help='save path of trained model')
parser.add_argument('--model-save-path','-mpath', type=str, default='/home/liuchun/Desktop/experment02/model_save', help='save path of trained model')
parser.add_argument('--loss-curve-path','-lpath', type=str, default='loss_log', help='save path of loss curve in tensorboard')
# others
parser.add_argument('--mode', '-m', type=str, default='train', help='whether training or test model, value should be set to train or test')
parser.add_argument('--pretrained', '-pt', type=bool, default=False, help='whether load checkpoint')
parser.add_argument('--option', '-option', type=str, default='twomask', help='whether load checkpoint') 
'''
onemask:原始模拟欠采mask固定 划分mask用loupe学习
twomask：原始模拟欠采mask和划分mask都用loupe学习
baseline:原始模拟欠采和划分mask都是用手动固定的内容
'''


#新网络参数
 
parser.add_argument('--slope','-s', type=float,  default=5, help='the first param of loupe')
parser.add_argument('--sample_slope','-ss', type=float,  default=200, help='the second param of loupe')
 


# bili_0=0.0#为了记录每训练一个轮次 得到的mask数值
# bili_1=0.0
def create_logger():
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s:\t%(message)s')
    stream_formatter = logging.Formatter('%(levelname)s:\t%(message)s')

    file_handler = logging.FileHandler(filename='logger.txt', mode='a+', encoding='utf-8')
    file_handler.setLevel(level=logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.INFO)
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

#初始化是否去掉
def init_weights(net, init_type='xavier', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('Initialization method {} is not implemented.'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class EarlyStopping:
    def __init__(self, patience=50, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, metrics, loss=True):
        score = -metrics if loss else metrics
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def forward(mode, rank, model, dataloader, criterion, optimizer, log, args, epoch,writer):
# def forward(mode, rank, model, dataloader, criterion, optimizer, log, args):
    # writer = SummaryWriter('test2_dc')
    assert mode in ['train', 'val', 'test']
    loss, psnr, ssim,rermask,imrmask,smask = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    print('dataloader:',len(dataloader))
    t = tqdm(dataloader, desc=mode + 'ing', total=int(len(dataloader))) if rank == 0 else dataloader
    
    for iter_num, data_batch in enumerate(t):

        im_gt = data_batch[0].to(rank, non_blocking=True)
        
        # output,loss_mask,dc_mask,und_mask = model(und_mask.contiguous(),im_gt,args.option,args.mode,dc_mask,loss_mask)#output class 设成1
        output_img,under_mask = model(im_gt,args.option,args.mode)#output class 设成1

        #calculate loss
        gt_kspace = complex2pseudo(image2kspace(pseudo2complex(im_gt)))
        
        output_kspace=complex2pseudo(image2kspace(pseudo2complex(output_img)))
        gt_mask=torch.ones_like(under_mask)
        #有监督
        batch_loss=cal_loss(gt_kspace,output_kspace,gt_mask)
        under_img=complex2pseudo(kspace2image(pseudo2complex(gt_kspace*under_mask))) #得到欠采输入的图像
        

        # 保存生成图像 欠采输入 生成的mask 重建输出
        
        if(args.strain!=None and args.mode=='train'):
            save_images(under_img,output_img,under_mask,im_gt,iter_num,args.strain,mode)
        if(args.stest!=None and args.mode=='test'):
            save_images(under_img,output_img,under_mask,im_gt,iter_num,args.stest,mode)  

        if mode == 'train':
            optimizer.zero_grad()
            batch_loss.backward()
            for name,param in model.named_parameters():
                if param.grad is None:
                    print(name)
            optimizer.step()

        gg=pseudo2real(im_gt)
        out=pseudo2real(output_img)
        # gg=(gg-torch.min(gg))/(torch.max(gg)-torch.min(gg))
        # out=(out-torch.min(out))/(torch.max(out)-torch.min(out))

        #计算学到mask的数值比例
        # rermask=rermask+(torch.sum(loss_mask[:,0,:,:])/torch.sum(under_mask[:,0,:,:]))
        # imrmask=imrmask+(torch.sum(loss_mask[:,1,:,:])/torch.sum(under_mask[:,1,:,:]))
        rermask=100000
        imrmask=100000
        # rermask=rermask+(torch.sum(loss_mask[:,0,:,:])/(256*256))
        # imrmask=imrmask+(torch.sum(loss_mask[:,1,:,:])/(256*256))
        #用于计算的中间量
        # trmask=torch.zeros_like(loss_mask[:,1,:,:])
        # srmask=torch.zeros_like(loss_mask[:,1,:,:])
        
        # #待改进
        # trmask[loss_mask[:,0,:,:]==1]=1
        # trmask[loss_mask[:,1,:,:]==1]=1
        # srmask[und_mask[:,0,:,:]==1]=1
        # srmask[und_mask[:,1,:,:]==1]=1
        # smask=smask+(torch.sum(trmask)/torch.sum(srmask)) #得到不分channel的统计量
        smask=10000 #得到不分channel的统计量

        ssim+=compute_ssim(out,gg)
        psnr+=compute_psnr(out,gg)
           
        loss += batch_loss.item()

    loss /= len(dataloader)
    log.append(loss)
    testimg=  abs(torch.complex(output_img[0, 0], output_img[0, 1])).unsqueeze(0)
    # testimg = (testimg - torch.min(testimg))/(torch.max(testimg) - torch.min(testimg))
    lmask0=loss_mask[0,0].unsqueeze(0)
    lmask1=loss_mask[0,1].unsqueeze(0)
    dmask0=dc_mask[0,0].unsqueeze(0)
    dmask1=dc_mask[0,1].unsqueeze(0)
    if mode =='val':
        writer.add_image('images/val_output', testimg, epoch)
        writer.add_image('images/val_lmask0', lmask0, epoch)
        writer.add_image('images/val_lmask1', lmask1, epoch)
        writer.add_image('images/val_dmask0', dmask0, epoch)
        writer.add_image('images/val_dmask1', dmask1, epoch)
    elif mode =='train':
        writer.add_image('images/train_output', testimg, epoch)
        writer.add_image('images/train_lmask0', lmask0, epoch)
        writer.add_image('images/train_lmask1', lmask1, epoch)
        writer.add_image('images/train_dmask0', dmask0, epoch)
        writer.add_image('images/train_dmask1', dmask1, epoch)
    
    # for name, param in model.named_parameters():
    #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
    #     writer.add_histogram(name + "/grad", param.grad.clone().cpu().data.numpy(), epoch)

    if mode == 'train':
        # curr_lr = optimizer.param_groups[0]['initial_lr']
        curr_lr = 0.0001
        log.append(curr_lr)
    # else:
    psnr /= len(dataloader)
    ssim /= len(dataloader)
    rermask /= len(dataloader)
    imrmask /= len(dataloader)
    smask /= len(dataloader)

    log.append(psnr)
    log.append(ssim)
    log.append(rermask)
    log.append(imrmask)
    log.append(smask)
    return log


def solvers(rank, ngpus_per_node, args):
    if rank == 0:
        logger = create_logger()
        logger.info('Running distributed data parallel on {} gpus.'.format(args.world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method=args.init_method, world_size=args.world_size, rank=rank)
    # set initial value
    start_epoch = 0
    best_ssim_val = 0.0
    best_ssim_train=0.0
    best_psnr_val = 0.0
    best_psnr_train=0.0
 
    # model
    
    model = Network(num_layers=args.num_layers, rank=rank,slope=args.slope,sample_slope=args.sample_slope,sparsity=args.dc_ratio)#默认值输入
    
    # print(model.keys())
    # whether load checkpoint  模型参数保存 改成共享参数可能会变化
    if args.pretrained or args.mode == 'test':
        #更改模型保存路径
        if(len(args.model_save_path.split('/'))>3):  #使用默认路径 3是根据前缀路径的长度确定
            path_of_model= args.model_save_path
        else:  #使用传入的参数路径
            path_of_model='/home/liuchun/Desktop/experment02/model_save/'+args.model_save_path

        model_path = os.path.join(path_of_model, 'best_checkpoint.pth.tar')
        print('model_path:',model_path)
        assert os.path.isfile(model_path)
        checkpoint = torch.load(model_path, map_location='cuda:{}'.format(rank))
        start_epoch = checkpoint['epoch']
        lr = checkpoint['lr']
        args.lr = lr
        best_ssim_val = checkpoint['best_ssim_val']
        best_ssim_train = checkpoint['best_ssim_train']
        best_psnr_val = checkpoint['best_psnr_val']
        best_psnr_train = checkpoint['best_psnr_train']
      
        
        model.load_state_dict(checkpoint['model'])
        if rank == 0:
            logger.info('Load checkpoint at epoch {}.'.format(start_epoch))
            logger.info('Current learning rate is {}.'.format(lr))
            logger.info('Current best ssim in val phase is {}.'.format(best_ssim_val))
            logger.info('Current best psnr in val phase is {}.'.format(best_psnr_val))
            logger.info('Current best ssim in train phase is {}.'.format(best_ssim_train))
            logger.info('Current best psnr in train phase is {}.'.format(best_psnr_train))
            
            
            logger.info('The model is loaded.')
    elif args.use_init_weights:
        init_weights(model, init_type=args.init_type, gain=args.gain)
        if rank == 0:
            logger.info('Initialize model with {}.'.format(args.init_type))
            # print('-------------------come rank==0')
    model = model.to(rank)
    # print('model to device successfully!')
    # writer.add_graph(model,input_to_model=[net_img_up.contiguous(), net_img_down.contiguous(),under_img.contiguous(),und_mask.contiguous()])
    # writer.add_graph(model, input_to_model = torch.rand(1, 3, 224, 224))
    # model = DDP(model, device_ids=[rank])#为了实现单GPU进行训练

    # criterion, optimizer, learning rate scheduler
    #损失函数部分 根据论文进行改变
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
    if not args.pretrained:
        warm_up = lambda epoch: epoch / args.warmup_epochs if epoch <= args.warmup_epochs else 1
        scheduler_wu = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_up)
    scheduler_re = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.3, patience=20)
    early_stopping = EarlyStopping(patience=50, delta=1e-5)

    #数据集部分
    # dataset_train1 = FastmriKnee('/home/liuchun/Desktop/learn_mask_ssdu/3ssdu_lm/ssdu_lm/dual_domain/data/PD_train_319.npz')
    # # print('dataset1:',len(dataset1))
    # dataset_train=DatasetReconMRI(dataset_train1)
    # # print('dataset:',len(dataset))
    # train_loader,val_loader,test_loader=build_loader(dataset,args.batch_size)
    u_mask_path,s_mask_up_path,s_mask_down_path=get_mask(args.u_mask_path)

    dataset_train = Dataset(data_path=args.train_path,u_mask_path=u_mask_path,s_mask_up_path=s_mask_up_path,s_mask_down_path=s_mask_down_path)
    dataset_val = Dataset(data_path=args.val_path,u_mask_path=u_mask_path,s_mask_up_path=s_mask_up_path,s_mask_down_path=s_mask_down_path)
    dataset_test = Dataset(data_path=args.test_path,u_mask_path=u_mask_path,s_mask_up_path=s_mask_up_path,s_mask_down_path=s_mask_down_path)

    train_loader =build_loader(dataset_train,args.batch_size,is_shuffle=True)
    val_loader =build_loader(dataset_val,args.batch_size,is_shuffle=False)
    test_loader =build_loader(dataset_test,args.batch_size,is_shuffle=False)

    # print('train_loader,val_loader,test_loader:',len(train_loader),len(val_loader),len(test_loader))
    # test step  数据集部分更改
    if args.mode == 'test':

        if rank == 0:
            # logger.info('The size of test dataset is {}.'.format(len(test_set)))
            logger.info('Now testing {}.'.format(args.exp_name))
        model.eval()
        with torch.no_grad():
            test_log = []
            start_time = time.time()
            print('test_loader:',len(test_loader))
            writer = SummaryWriter('defe')
            sada=100
            test_log = forward('test', rank, model, test_loader, criterion, optimizer, test_log, args,sada,writer)
            test_time = time.time() - start_time
        # test information
        test_loss = test_log[0]
        test_psnr = test_log[1]
        test_ssim = test_log[2]
       
        if rank == 0:
            
            # logger.info('ratio of real(lossmask/undermask):{:.5f}\tratio of imaginary(lossmask/undermask):{:.5f}\tratio of total(lossmask/undermask):{:.5f}'.format(train_rermask, train_imrmask,train_smask))#输出当前计算的mask比例信息
            logger.info('time:{:.5f}s\ttest_loss:{:.7f}\ttest_psnr:{:.5f}\ttest_ssim:{:.5f}'.format(test_time, test_loss, test_psnr, test_ssim))
        return

    if rank == 0:
        # logger.info('The size of training dataset and validation dataset is {} and {}, respectively.'.format(len(train_set), len(val_set)))
        path_loss='/home/liuchun/Desktop/experment02/loss_save/'
        path_loss=path_loss+args.loss_curve_path
        if not os.path.exists(path_loss):
            os.makedirs(path_loss)
        writer = SummaryWriter(path_loss)
    for epoch in range(start_epoch + 1, args.num_epochs + 1):
        # train_sampler.set_epoch(epoch)
        train_log = [epoch]
        epoch_start_time = time.time()
        model.train()
        # print('train_loader:',len(train_loader))
        train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args, epoch,writer)
        # train_log = forward('train', rank, model, train_loader, criterion, optimizer, train_log, args)
        model.eval()
        with torch.no_grad():
            train_log = forward('val', rank, model, val_loader, criterion, optimizer, train_log, args, epoch,writer)
            # train_log = forward('val', rank, model, val_loader, criterion, optimizer, train_log, args)
        epoch_time = time.time() - epoch_start_time
        # train information
        epoch = train_log[0]
        train_loss = train_log[1]
        lr = train_log[2]
        # train_psnr = train_log[3]
        # train_ssim = train_log[4]
        # # bili_0 = train_log[5]
        # # bili_1 = train_log[6]
        # val_loss = train_log[5]
        # val_psnr = train_log[6]
        # val_ssim = train_log[7]

        train_psnr = train_log[3]
        train_ssim = train_log[4]
        train_imrmask = train_log[5]
        train_rermask = train_log[6]
        train_smask=train_log[7]
        val_loss = train_log[8]
        val_psnr = train_log[9]
        val_ssim = train_log[10]

        is_best_val = val_ssim > best_ssim_val
        best_ssim_val = max(val_ssim, best_ssim_val)
        best_psnr_val = max(val_psnr, best_psnr_val)
        is_best_train = train_ssim > best_ssim_train
        best_ssim_train = max(train_ssim, best_ssim_train)
        best_psnr_train = max(train_psnr, best_psnr_train)
        if rank == 0:
            logger.info('ratio of real(lossmask/undermask):{:.5f}\tratio of imaginary(lossmask/undermask):{:.5f}\tratio of total(lossmask/undermask):{:.5f}'.format(train_rermask, train_imrmask,train_smask))#输出当前计算的mask比例信息

            logger.info('epoch:{:<8d}time:{:.5f}s\tlr:{:.8f}\ttrain_loss:{:.7f}\ttrain_psnr:{:.7f}\ttrain_ssim:{:.7f}\tval_loss:{:.7f}\tval_psnr:{:.5f}\t'
                        'val_ssim:{:.5f}'.format(epoch, epoch_time, lr, train_loss, train_psnr, train_ssim, val_loss, val_psnr, val_ssim))
            writer.add_scalars('loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
            # save checkpoint
            # print(1)#isidchdsih
            checkpoint = {
                'epoch': epoch,
                'lr': lr,
                'best_ssim_val': best_ssim_val,
                'best_ssim_train':best_ssim_train,
                'best_psnr_val': best_psnr_val,
                'best_psnr_train':best_psnr_train,
                'model': model.state_dict(keep_vars=True)
                # 'model': model.module.state_dict()
            }
            if(len(args.model_save_path.split('/'))>3):  #使用默认路径 3是根据前缀路径的长度确定
                path_of_model= args.model_save_path
            else:  #使用传入的参数路径
                path_of_model='/home/liuchun/Desktop/experment02/model_save/'+args.model_save_path

            if not os.path.exists(path_of_model):
                os.makedirs(path_of_model)

            model_path = os.path.join(path_of_model, 'checkpoint.pth.tar')
            best_model_path = os.path.join(path_of_model, 'best_checkpoint.pth.tar')
            torch.save(checkpoint, model_path)
            print('modelpath:',model_path)
            print('save the checkpoints successfully!')
            if is_best_val:
                shutil.copy(model_path, best_model_path) #复制文件内容
        # scheduler
        if epoch <= args.warmup_epochs and not args.pretrained:
            scheduler_wu.step()
        scheduler_re.step(val_ssim)
        # early_stopping(val_ssim, loss=False)
        # if early_stopping.early_stop:
        #     if rank == 0:
        #         logger.info('The experiment is early stop!')
               
        #     break
    if rank == 0:
        writer.close()
    return


def main():
    args = parser.parse_args()
    args.world_size = args.nodes * args.gpus
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.multiprocessing.spawn(solvers, nprocs=args.gpus, args=(args.gpus, args))


if __name__ == '__main__':
    main()
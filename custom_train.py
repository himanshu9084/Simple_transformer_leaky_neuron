#from spikingjelly.activation_based import functional, surrogate, neuron
#from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly_codes.activation_based import functional, surrogate, neuron, layer
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from spikingjelly.datasets.n_mnist import NMNIST
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.nn.functional as F
from torch.cuda import amp
import torch.nn as nn
from set_seed import _seed_
import numpy as np
import logging
import random
import time
import os
import argparse
import datetime
import warnings
import random 
import math
import torch
import sys

random.seed(_seed_)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(_seed_)

import simple_archi as hybrid_models 


def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    '''
    split_to_train_test_set taken from https://github.com/fangwei123456/Spike-Element-Wise-ResNet/blob/main/cifar10dvs/train.py

    :param train_ratio: split the ratio of the origin dataset as the train set
    :type train_ratio: float
    :param origin_dataset: the origin dataset
    :type origin_dataset: torch.utils.data.Dataset
    :param num_classes: total classes number, e.g., ``10`` for the MNIST dataset
    :type num_classes: int
    :param random_split: If ``False``, the front ratio of samples in each classes will
            be included in train set, while the reset will be included in test set.
            If ``True``, this function will split samples in each classes randomly. The randomness is controlled by
            ``numpy.randon.seed``
    :type random_split: int
    :return: a tuple ``(train_set, test_set)``
    :rtype: tuple
    '''
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
        for i in range(num_classes):
            np.random.shuffle(label_idx[i])

    for i in range(num_classes):
        pos = math.ceil(label_idx[i].__len__() * train_ratio)
        train_idx.extend(label_idx[i][0: pos])
        test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def main():
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=20, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:1', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=1024, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')
    
    parser.add_argument('-data-dir', type=str, default='event_vision/datasets/dvsgesture/', help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp',default=False, action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy',default=False, action='store_true', help='use cupy backend')
    
    parser.add_argument('-init_tau',default=2.0, type=float)
    parser.add_argument('-use_plif', action='store_true', default=True)
    parser.add_argument('-alpha_learnable', action='store_true', default=False)
    parser.add_argument('-use_max_pool', action='store_true', default=True)
    parser.add_argument('-number_layer', default=5, type=int)
    parser.add_argument('-detach_reset', action='store_true', default=True)
    
    parser.add_argument('-opt',default='adam', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=64, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-model_name', default='simple_transformer', type=str, help='model name to run.')
    parser.add_argument('-dataset', default='dvsgesture', type=str, help='set seed')
    parser.add_argument('-seed', default=_seed_, type=int, help='set seed')
    
    args = parser.parse_args()
    now = datetime.datetime.now()
    # dd/mm/YY H:M:S
    dt_log_name = now.strftime("%d%m%Y_%H_%M_%S")
    
    out_dir = os.path.join(args.out_dir, f'{dt_log_name}')
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        #print(f'Mkdir {out_dir}.')
    
    logger = logging.getLogger('Hybrid_models')
    logging.basicConfig(filename=out_dir+'/training.log',filemode='a',
    format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%d-%m-%Y %H:%M:%S')
    logger = logging.getLogger()

    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"Start time : {dt_string}")
    logger.info(torch.cuda.is_available())
    logger.info(torch.cuda.device_count())
    logger.info(torch.cuda.current_device())
    logger.info(torch.cuda.device(0))
    logger.info(torch.cuda.get_device_name(0))
    logger.info("\n\n")
    logger.info(dt_log_name)
    logger.info(f"Logs at : {out_dir}")
    logger.info(args)
    
    if args.dataset == 'dvsgesture':
        data_dir = "event_vision/datasets/dvsgesture/"
        class_num = 11
        train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
        logger.info("DVS Gesture Dataset : 11 classes")
        
    elif args.dataset == 'n_mnist':
        data_dir = "event_vision/datasets/n_mnist/"
        class_num = 10
        train_set = NMNIST(root=data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
        test_set = NMNIST(root=data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')
        logger.info("n-MNIST Dataset : 10 classes")
    
    elif args.dataset == 'CIFAR10_DVS':
        data_dir = "event_vision/datasets/CIFAR10_DVS/"
        class_num = 10
        logger.info("CIFAR10DVS Dataset : 10 classes")
        
        train_set_pth = os.path.join(data_dir, f'train_set_{args.T}.pt')
        test_set_pth = os.path.join(data_dir, f'test_set_{args.T}.pt')
        
        if os.path.exists(train_set_pth) and os.path.exists(test_set_pth):
            train_set = torch.load(train_set_pth)
            test_set = torch.load(test_set_pth)
        else:
            orig_set = CIFAR10DVS(root=data_dir, data_type='frame', frames_number=args.T, split_by='number')
            logger.info(f'original samples = {orig_set.__len__()}')
            logger.info('Splitting CIFAR10 DVS into 90% train, 10% test split')
            train_set, test_set = split_to_train_test_set(0.9, orig_set, 10)
            torch.save(train_set, train_set_pth)
            torch.save(test_set, test_set_pth)
        
        
    logger.info(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
    logger.info(f'total samples = {train_set.__len__() + test_set.__len__()}')


    
    if args.model_name == 'simple_transformer':
        net = hybrid_models.Simple_SNN(num_classes=class_num)   
        
    
    logger.info(args.model_name)
    logger.info(f"\n{net}")
    net.to(args.device)
    
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    pytorch_train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logger.info(f"Number of parameters : {pytorch_total_params}")
    logger.info(f"Number of trainable parameters : {pytorch_train_params}")
    
    
    

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )

    
    start_epoch = 0
    max_test_acc = -1
    optimizer = None
    
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)
    
    
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    
    

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']
    
    
    
    #writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        
        net.train()
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.float().to(args.device)
            
            #frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, class_num).float()
            
            out_spikes_counter = net(frame)
            loss = F.mse_loss(out_spikes_counter, label_onehot)
            
            
            loss.backward()
            optimizer.step()
            
            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_spikes_counter.argmax(1) == label).float().sum().item()
            
            functional.reset_net(net)
            
        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
        
        logger.info(f'Epoch {epoch}, train_loss : {train_loss}')
        logger.info(f'Epoch {epoch}, train_acc : {train_acc}')
        #writer.add_scalar('train_loss', train_loss, epoch)
        #writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                
                #frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
                label = label.to(args.device)
                label_onehot = F.one_hot(label, class_num).float()
                
                out_fr = net(frame) #.mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
            
                '''
                # spike regularization
                spike_loss = spike_regularization_loss(net.temporal_processor.lif)
                mse_loss = F.mse_loss(out_fr, label_onehot)
                loss = mse_loss + 0.01 * spike_loss
                '''
                
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
                
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        
        logger.info(f'Epoch {epoch}, test_loss : {test_loss}')
        logger.info(f'Epoch {epoch}, test_acc : {test_acc}')
        #writer.add_scalar('test_loss', test_loss, epoch)
        #writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            best_file = os.path.join(out_dir, 'checkpoint_max.pth')
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
            logger.info(f'Best checkpoint saved : {best_file}')
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
        logger.info(f'Latest checkpoint saved : checkpoint_latest.pth')
        
        #print(args)
        #print(out_dir)
        logger.info(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        logger.info(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        logger.info(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')

        
        '''
        print(f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        '''
    logger.info("Training finished.")
    now = datetime.datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    logger.info(f"End time : {dt_string}")

if __name__ == '__main__':
    main()
    
    
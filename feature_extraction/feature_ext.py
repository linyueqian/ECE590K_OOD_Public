#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 2:18:52 2023

@author: Yike Guo
"""

import os
import argparse
from importlib import import_module
import shutil
import json

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
#import torchsso
#from torchsso.optim import SecondOrderOptimizer, VIOptimizer
#from torchsso.utils import Logger
import numpy as np
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)

def feature_extraction(optimizer='Bayessian',dataset='DATASET_CIFAR10',data_augmentation = False,download = True):
    
    '''
    Return a dictionary containing features extracted from different layers of LeNet5 with batch normalization

    Parameters
    -----------
        optimizer: ['Bayessian','Adam']
        dataset:['DATASET_CIFAR10','DATASET_SVHN','DATASET_MNIST','DATASET_Places365','DATASET_Texture']
        data_augmentation:[True, False]
        download: [True,False]
    
    LeNet5BN Structure
    ------------------
        Two conv layers + BN + maxpool2d
        feature1 = out.view(out.size(0), -1)
        feature2 = F.relu(self.bn3(self.fc1(feature1)))
        feature3 = F.relu(self.bn4(self.fc2(feature2)))
        out = self.fc3(feature3)
    '''   

    if optimizer == 'Bayessian':
        configpath = 'configs/lenet_vogn_feature.json'
        ckptpath = './ckpts/bay_epoch30_ckp.ckpt'
    elif optimizer == 'Adam':
        configpath = 'configs/lenet_adam_feature.json'
        ckptpath = './ckpts/adam_epoch30_ckp.ckpt'
    else:
        print('Not Available Optimizer')   
        return

    ## load config file    
    with open(configpath) as f:
        config = json.load(f)

    ## set up
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    batchsize = config['batch_size']

    ## load data pre-processing
    data_transforms = []
    if data_augmentation == True:
        ## random crop
        data_transforms.append(transforms.RandomCrop(32, padding=4))
        ## random horizontal flip
        data_transforms.append(transforms.RandomHorizontalFlip())
        ## style augmentation
        ## 接口 @Max
    if dataset == 'DATASET_MNIST':
        colorized = transforms.Grayscale(num_output_channels = 3)
        resized = transforms.Resize(32)
        data_transforms.append(colorized)
        data_transforms.append(resized)
    data_transforms.append(transforms.ToTensor())  
    if config['normalizing_data'] == True: 
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_transforms.append(normalize)     
    transform = transforms.Compose(data_transforms)  
    
    ## load data
    if dataset == 'DATASET_CIFAR10':
        # CIFAR-10
        num_classes = 10
        dataset_class = datasets.CIFAR10
    elif dataset == 'DATASET_SVHN':
        ## SVHN
        num_classes = 10
        dataset_class = datasets.SVHN
    elif dataset == 'DATASET_MNIST':
        ## MNIST
        num_classes = 10
        dataset_class = datasets.MNIST
    elif dataset == 'DATASET_Texture':
        ## Texture
        num_classes = 47
        dataset_class = datasets.DTD
    elif dataset == 'DATASET_Places365':
        ## Places365
        num_classes = 434
        dataset_class = datasets.Places365
    else:
        print('Unknown Dataset')
        return
        
    dataset = dataset_class(root='./data', download=download, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=0)    
    print('Data Loading Done')
    
    ## load model  
    _, ext = os.path.splitext(config['arch_file'])
    dirname = os.path.dirname(config['arch_file'])
    module_path = '.'.join(os.path.split(config['arch_file'])).replace(ext, '')
    module = import_module(module_path)
    arch_class = getattr(module, config['arch_name'])
    arch_kwargs = {} if config['arch_args'] == 'None' else config['arch_args']
    arch_kwargs['num_classes'] = num_classes
    model = arch_class(**arch_kwargs)
    setattr(model, 'num_classes', num_classes)
    model.to(device)
    print('Model Loading Done')
    
    
    ## load checkpoint and update model
    checkpoint = torch.load(ckptpath)
    print('checkpoint path: ' + ckptpath)
    model.load_state_dict(checkpoint['model'])
    
    
    ## extract features
    model.eval()

    feature1_ls = []                                                     
    feature2_ls = []
    feature3_ls = []                                                     
    with torch.no_grad():
        for data, target in data_loader:

            data, target = data.to(device), target.to(device)  
            output,feature1,feature2,feature3 = model(data)

            feature1_ls.append(feature1.detach().cpu().numpy().reshape(-1,1))
            feature2_ls.append(feature2.detach().cpu().numpy().reshape(-1,1))
            feature3_ls.append(feature3.detach().cpu().numpy().reshape(-1,1))

    features = {'feature1':feature1_ls,'feature2':feature2_ls,'feature3':feature3_ls}        
    print('Successfully Extract Features !')
    
    return features

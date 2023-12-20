import torch  
import einops
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Subset
import math
from torch.nn import functional as F
from torch.optim import Optimizer
import loralib as lora
from transformers import LlamaConfig
from functools import partial
from torchvision.transforms import Compose, Lambda
from torchvision import datasets, transforms
from peft import LoraConfig
from accelerate import Accelerator
import argparse
from tqdm import tqdm
import os
import random


from metric import cal_mae_mse
from losses import LpLoss
from model import complete_model
from data.class_lmdb_dataset import LmdbDataset
from metric import cal_mae_mse

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        # torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)
    elif isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)

class SimpleModel(nn.Module):
    def __init__(self, dim=16):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = dim, out_channels = 2*dim, kernel_size = (3, 3), stride = (2, 2), padding=(2, 1))  #  (b*t, 2 * dim, 45, 46) -> (b*t, 4 * dim, 23, 23)
        self.conv2 = nn.Conv2d(in_channels = 2*dim, out_channels = 4*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))   #  (b*t, 2 * dim, 23, 23) -> (b*t, 4 * dim, 12, 12)
        self.conv3 = nn.Conv2d(in_channels = 4*dim, out_channels = 8*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))     # (b*t, 4 * dim, 12, 12) ->  (b*t, 4 * dim, 6, 6)
        
        self.fc1 = nn.Linear(in_features=8*dim*12*6, out_features=4096) 
        self.fc2 = nn.Linear(in_features=4096, out_features=8*dim*12*6) 
        
        self.deconv1 = nn.ConvTranspose2d(in_channels = 8*dim, out_channels = 4*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1), output_padding = (0, 1)) # (b*t, 4 * dim, 12, 6) -> (b*t, 4 * dim, 23, 12) 
        self.deconv2 = nn.ConvTranspose2d(in_channels = 4*dim, out_channels = 2*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1), output_padding = (1, 0)) # (b*t, 4 * dim, 23, 12) -> (b*t, 2 * dim, 46, 23) 
        self.deconv3 = nn.ConvTranspose2d(in_channels = 2*dim, out_channels = dim, kernel_size = (3, 3), stride = (2, 2), padding=(2, 1), output_padding = (1, 1))  #  (b*t, 4 * dim, 46, 23) -> (b*t, 2 * dim, 90, 46) 
       
class SimpleModel_4(nn.Module):
    # def __init__(self, dim=64):
    #     super(SimpleModel_4, self).__init__()
        
    #     # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
    #     self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = dim, kernel_size = (5, 5), stride = (4, 4), padding=(2, 2))
    #         #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 90, 46)
    #     self.conv1 = nn.Conv2d(in_channels = dim, out_channels = 2*dim, kernel_size = (3, 3), stride = (1, 1), padding=(1, 1))
    #     self.pool1 = nn.MaxPool2d(2, stride=2, padding = (0, 1))
    #         #  (b*t, 2 * dim, 90, 46) -> (b*t, 4 * dim, 46, 23)
    #     self.conv2 = nn.Conv2d(in_channels = 2*dim, out_channels = 4*dim, kernel_size = (3, 3), stride = (1, 1), padding=(1, 1))
    #     self.pool2 = nn.MaxPool2d(2, stride=2, padding = (1, 0))
    #     #  (b*t, 2 * dim, 46, 23) -> (b*t, 4 * dim, 23, 12)
    #     self.conv3 = nn.Conv2d(in_channels = 4*dim, out_channels = 8*dim, kernel_size = (3, 3), stride = (1, 1), padding=(1, 1))
    #     self.pool3 = nn.MaxPool2d(2, stride=2, padding = (0, 1))
    #     # (b*t, 4 * dim, 23, 12) ->  (b*t, 4 * dim, 12, 6)
    #     self.conv4 = nn.Conv2d(in_channels = 8*dim, out_channels = 16*dim, kernel_size = (3, 3), stride = (1, 1), padding=(1, 1))
    #     self.pool4 = nn.MaxPool2d(2, stride=2, padding = (1, 0))
        
    #     self.fc1 = nn.Linear(in_features=16*dim, out_features=4096)
    #     self.fc2 = nn.Linear(in_features=4096, out_features=16*dim)
    #     # (b*t, 4 * dim, 12, 6) -> (b*t, 4 * dim, 23, 12) 
    #     self.deconv1 = nn.ConvTranspose2d(in_channels = 16*dim, out_channels = 8*dim, kernel_size = (5, 4), stride = (2, 2), padding=(2, 1))
    #     # (b*t, 4 * dim, 23, 12) -> (b*t, 2 * dim, 46, 23) 
    #     self.deconv2 = nn.ConvTranspose2d(in_channels = 16*dim, out_channels = 4*dim, kernel_size = (4, 3), stride = (2, 2), padding=(1, 1))
    #       #  (b*t, 4 * dim, 46, 23) -> (b*t, 2 * dim, 90, 46) 
    #     self.deconv3 = nn.ConvTranspose2d(in_channels = 8*dim, out_channels = 2*dim, kernel_size = (4, 4), stride = (2, 2), padding=(2, 1))
    #     # (b*t, 2 * dim, 90, 46) -> (b*t, dim, 180, 91)
    #     self.deconv4 = nn.ConvTranspose2d(in_channels = 4*dim, out_channels = dim, kernel_size = (4, 3), stride = (2, 2), padding=(1, 1))
    #     #  (b*t, dim, 180, 91) -> (b*t, 4, 720, 361)
    #     self.deconv_surface = nn.ConvTranspose2d(in_channels = dim, out_channels = 4, kernel_size = (6, 5), stride = (4, 4), padding=(1, 2))
        
    #     # self.act = nn.Sigmoid()
    #     self.act = nn.GELU()
    #     # self.act = nn.Identity()
    #     self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
    #     self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim*2, eps=1e-6, affine=True)
    #     self.norm3 = nn.GroupNorm(num_groups=32, num_channels=dim*4, eps=1e-6, affine=True)
    #     self.norm4 = nn.GroupNorm(num_groups=32, num_channels=dim*8, eps=1e-6, affine=True)
    #     self.norm5 = nn.GroupNorm(num_groups=32, num_channels=dim*8, eps=1e-6, affine=True)
    #     self.norm6 = nn.GroupNorm(num_groups=32, num_channels=dim*4, eps=1e-6, affine=True)
    #     self.norm7 = nn.GroupNorm(num_groups=32, num_channels=dim*2, eps=1e-6, affine=True)
    #     self.norm8 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        
        # self.norm1 = nn.Identity()
        # self.norm2 = nn.Identity()
        # self.norm3 = nn.Identity()
        # self.norm4 = nn.Identity()
        # self.norm5 = nn.Identity()
        # self.norm6 = nn.Identity()
        # self.norm7 = nn.Identity()
        # self.norm8 = nn.Identity()
    # def __init__(self, dim=64):
    #     super(SimpleModel_4, self).__init__()
        
    #     # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
    #     self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = dim, kernel_size = (4, 4), stride = (4, 4), padding = (1, 2))
    #         #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 90, 46)
    #     self.conv1 = nn.Conv2d(in_channels = dim, out_channels = 2*dim, kernel_size = (2, 2), stride = (2, 2), padding=(0, 1))
    #         #  (b*t, 2 * dim, 90, 46) -> (b*t, 4 * dim, 46, 23)
    #     self.conv2 = nn.Conv2d(in_channels = 2*dim, out_channels = 4*dim, kernel_size = (2, 2), stride = (2, 2), padding=(1, 0))
    #     #  (b*t, 2 * dim, 46, 23) -> (b*t, 4 * dim, 23, 12)
    #     self.conv3 = nn.Conv2d(in_channels = 4*dim, out_channels = 8*dim, kernel_size = (2, 2), stride = (2, 2), padding=(0, 1))
    #     # (b*t, 4 * dim, 23, 12) ->  (b*t, 4 * dim, 12, 6)
    #     self.conv4 = nn.Conv2d(in_channels = 8*dim, out_channels = 4096, kernel_size = (2, 2), stride = (2, 2), padding=(1, 0))
        
    #     # self.fc1 = nn.Linear(in_features=16*dim, out_features=4096)
    #     # self.fc2 = nn.Linear(in_features=4096, out_features=16*dim)
    #     # (b*t, 4 * dim, 12, 6) -> (b*t, 4 * dim, 24, 12) 
    #     self.deconv1 = nn.ConvTranspose2d(in_channels = 4096, out_channels = 8*dim, kernel_size = (2, 2), stride = (2, 2), padding=(0, 0))
    #     # (b*t, 4 * dim, 24, 12) -> (b*t, 2 * dim, 46, 24) 
    #     self.deconv2 = nn.ConvTranspose2d(in_channels = 8*dim, out_channels = 4*dim, kernel_size = (2, 2), stride = (2, 2), padding=(1, 0))
    #       #  (b*t, 4 * dim, 46, 24) -> (b*t, 2 * dim, 90, 46) 
    #     self.deconv3 = nn.ConvTranspose2d(in_channels = 4*dim, out_channels = 2*dim, kernel_size = (2, 2), stride = (2, 2), padding=(1, 1))
    #     # (b*t, 2 * dim, 90, 46) -> (b*t, dim, 180, 92)
    #     self.deconv4 = nn.ConvTranspose2d(in_channels = 2*dim, out_channels = dim, kernel_size = (2, 2), stride = (2, 2), padding=(0, 0))
    #     #  (b*t, dim, 180, 92) -> (b*t, 4, 720, 362)
    #     self.deconv_surface = nn.ConvTranspose2d(in_channels = dim, out_channels = 4, kernel_size = (4, 4), stride = (4, 4), padding=(0, 3))
        
    #     # self.act = nn.Sigmoid()
    #     self.act = nn.GELU()
        
    #     self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
    #     self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim*2, eps=1e-6, affine=True)
    #     self.norm3 = nn.GroupNorm(num_groups=32, num_channels=dim*4, eps=1e-6, affine=True)
    #     self.norm4 = nn.GroupNorm(num_groups=32, num_channels=dim*8, eps=1e-6, affine=True)
    #     self.norm5 = nn.GroupNorm(num_groups=32, num_channels=dim*8, eps=1e-6, affine=True)
    #     self.norm6 = nn.GroupNorm(num_groups=32, num_channels=dim*4, eps=1e-6, affine=True)
    #     self.norm7 = nn.GroupNorm(num_groups=32, num_channels=dim*2, eps=1e-6, affine=True)
    #     self.norm8 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        
    # def __init__(self, dim=64):
    #     super(SimpleModel_4, self).__init__()
        
    #     # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
    #     self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = dim, kernel_size = (5, 5), stride = (4, 4), padding=(2, 2))
    #         #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 90, 46)
    #     self.conv1 = nn.Conv2d(in_channels = dim, out_channels = 2*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))
    #         #  (b*t, 2 * dim, 90, 46) -> (b*t, 4 * dim, 46, 23)
    #     self.conv2 = nn.Conv2d(in_channels = 2*dim, out_channels = 4*dim, kernel_size = (3, 3), stride = (2, 2), padding=(2, 1))
    #     #  (b*t, 2 * dim, 46, 23) -> (b*t, 4 * dim, 23, 12)
    #     self.conv3 = nn.Conv2d(in_channels = 4*dim, out_channels = 8*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))
    #     # (b*t, 4 * dim, 23, 12) ->  (b*t, 4 * dim, 12, 6)
    #     self.conv4 = nn.Conv2d(in_channels = 8*dim, out_channels = 16*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))
        
    #     self.fc1 = nn.Linear(in_features=16*dim, out_features=4096)
    #     self.fc2 = nn.Linear(in_features=4096, out_features=16*dim)
    #     # (b*t, 4 * dim, 12, 6) -> (b*t, 4 * dim, 23, 12) 
    #     self.deconv1 = nn.ConvTranspose2d(in_channels = 16*dim, out_channels = 8*dim, kernel_size = (5, 4), stride = (2, 2), padding=(2, 1))
    #     # (b*t, 4 * dim, 23, 12) -> (b*t, 2 * dim, 46, 23) 
    #     self.deconv2 = nn.ConvTranspose2d(in_channels = 8*dim, out_channels = 4*dim, kernel_size = (4, 3), stride = (2, 2), padding=(1, 1))
    #       #  (b*t, 4 * dim, 46, 23) -> (b*t, 2 * dim, 90, 46) 
    #     self.deconv3 = nn.ConvTranspose2d(in_channels = 4*dim, out_channels = 2*dim, kernel_size = (4, 4), stride = (2, 2), padding=(2, 1))
    #     # (b*t, 2 * dim, 90, 46) -> (b*t, dim, 180, 91)
    #     self.deconv4 = nn.ConvTranspose2d(in_channels = 2*dim, out_channels = dim, kernel_size = (4, 3), stride = (2, 2), padding=(1, 1))
    #     #  (b*t, dim, 180, 91) -> (b*t, 4, 720, 361)
    #     self.deconv_surface = nn.ConvTranspose2d(in_channels = dim, out_channels = 4, kernel_size = (6, 5), stride = (4, 4), padding=(1, 2))
        
    #     # self.act = nn.Sigmoid()
    #     self.act = nn.ReLU()
        
    #     self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
    #     self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim*2, eps=1e-6, affine=True)
    #     self.norm3 = nn.GroupNorm(num_groups=32, num_channels=dim*4, eps=1e-6, affine=True)
    #     self.norm4 = nn.GroupNorm(num_groups=32, num_channels=dim*8, eps=1e-6, affine=True)
    #     self.norm5 = nn.GroupNorm(num_groups=32, num_channels=dim*8, eps=1e-6, affine=True)
    #     self.norm6 = nn.GroupNorm(num_groups=32, num_channels=dim*4, eps=1e-6, affine=True)
    #     self.norm7 = nn.GroupNorm(num_groups=32, num_channels=dim*2, eps=1e-6, affine=True)
    #     self.norm8 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
    
    
    def __init__(self, dim=192):
        super(SimpleModel_4, self).__init__()
        
        # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = dim, kernel_size = (15, 15), stride = (4, 4), padding=(7, 7))
            #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 90, 46)
        self.conv1 = nn.Conv2d(in_channels = dim, out_channels = 2*dim, kernel_size = (9, 9), stride = (2, 2), padding=(4, 4))
            #  (b*t, 2 * dim, 90, 46) -> (b*t, 4 * dim, 46, 23)
        self.conv2 = nn.Conv2d(in_channels = 2*dim, out_channels = 4*dim, kernel_size = (7, 7), stride = (2, 2), padding=(4, 3))
        #  (b*t, 2 * dim, 46, 23) -> (b*t, 4 * dim, 23, 12)
        self.conv3 = nn.Conv2d(in_channels = 4*dim, out_channels = 8*dim, kernel_size = (5, 5), stride = (2, 2), padding=(2, 2))
        # (b*t, 4 * dim, 23, 12) ->  (b*t, 4 * dim, 12, 6)
        self.conv4 = nn.Conv2d(in_channels = 8*dim, out_channels = 16*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))
        
        # self.fc1 = nn.Linear(in_features=4, out_features=4096)
        # self.fc2 = nn.Linear(in_features=4096, out_features=4)
        # (b*t, 4 * dim, 12, 6) -> (b*t, 4 * dim, 23, 12) 
        self.deconv1 = nn.ConvTranspose2d(in_channels = 16*dim, out_channels = 8*dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1), output_padding=(0, 1))
        # (b*t, 4 * dim, 23, 12) -> (b*t, 2 * dim, 46, 23) 
        self.deconv2 = nn.ConvTranspose2d(in_channels = 8*dim, out_channels = 4*dim, kernel_size = (5, 5), stride = (2, 2), padding=(2, 2), output_padding=(1, 0))
        
          #  (b*t, 4 * dim, 46, 23) -> (b*t, 2 * dim, 90, 46) 
        self.deconv3 = nn.ConvTranspose2d(in_channels = 4*dim, out_channels = 2*dim, kernel_size = (7, 7), stride = (2, 2), padding=(4, 3), output_padding=(1, 1))
        # self.deconv3 = nn.Identity()
        # (b*t, 2 * dim, 90, 46) -> (b*t, dim, 180, 91)
        self.deconv4 = nn.ConvTranspose2d(in_channels = 2*dim, out_channels = dim, kernel_size = (9, 9), stride = (2, 2), padding=(4, 4), output_padding=(1, 0))
        #  (b*t, dim, 180, 91) -> (b*t, 4, 720, 361)
        self.deconv_surface = nn.ConvTranspose2d(in_channels = dim, out_channels = 4,  kernel_size = (15, 15), stride = (4, 4), padding=(7, 7), output_padding=(3, 0))
        
        # self.act = nn.Sigmoid()
        self.act = nn.GELU()
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim*2, eps=1e-6, affine=True)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=dim*4, eps=1e-6, affine=True)
        self.norm4 = nn.GroupNorm(num_groups=32, num_channels=dim*8, eps=1e-6, affine=True)
        self.norm5 = nn.GroupNorm(num_groups=32, num_channels=dim*8, eps=1e-6, affine=True)
        self.norm6 = nn.GroupNorm(num_groups=32, num_channels=dim*4, eps=1e-6, affine=True)
        self.norm7 = nn.GroupNorm(num_groups=32, num_channels=dim*2, eps=1e-6, affine=True)
        self.norm8 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        # self.act = nn.Identity()
        
        # self.norm1 = TransposeLN(64)
        # self.norm2 = TransposeLN(2*64)
        # self.norm3 = TransposeLN(2*64)
        # self.norm4 = TransposeLN(64)
        # self.norm1 = nn.Identity()
        # self.norm2 = nn.Identity()
        # self.norm3 = nn.Identity()
        # self.norm4 = nn.Identity()
        # self.norm5 = nn.Identity()
        # self.norm6 = nn.Identity()
        # self.norm7 = nn.Identity()
        # self.norm8 = nn.Identity()
    
    # def forward(self, x):           # (1, 4, 720, 361)
    #     # 卷积
    #     x = self.conv_surface(x)       # torch.Size([1, 64, 180, 91])
 
    #     x1 = self.pool1(self.conv1(self.norm1(self.act(x))))         #  (b*t, 128, 90, 46)
    #     x2 = self.pool2(self.conv2(self.norm2(self.act(x1))))         #  (b*t, 256, 46, 23)
    #     x3 = self.pool3(self.conv3(self.norm3(self.act(x2))))         #  (b*t, 512, 23, 12)
    #     x4 = self.pool4(self.conv4(self.norm4(self.act(x3))))         #  (b*t, 1024, 12, 6)
    #     # x = self.fc1(x.transpose(1, -1))
    #     # # 反卷积
    #     # x = self.fc2(x).transpose(1, -1)
    #     x = self.act(self.norm5(self.deconv1(x4)))   # torch.Size([1, 512, 23, 12])
    #     x = self.act(self.norm6(self.deconv2(x + x3)))    # torch.Size([1, 256, 46, 23])
    #     x = self.act(self.norm7(self.deconv3(x + x2)))    # torch.Size([1, 128, 90, 46])
    #     x = self.act(self.norm8(self.deconv4(x + x1)))    # torch.Size([1, 64, 180, 91])
   
    #     x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])
      
    #     return x
    
    # def forward(self, x):           # (1, 4, 720, 361)
    #     # 卷积
    #     x = self.conv_surface(x)       # torch.Size([1, 64, 180, 91])
 
    #     x1 = self.pool1(self.conv1(self.norm1(self.act(x))))         #  (b*t, 128, 90, 46)
    #     x2 = self.pool2(self.conv2(self.norm2(self.act(x1))))         #  (b*t, 256, 46, 23)
    #     x3 = self.pool3(self.conv3(self.norm3(self.act(x2))))         #  (b*t, 512, 23, 12)
    #     x4 = self.pool4(self.conv4(self.norm4(self.act(x3))))         #  (b*t, 1024, 12, 6)
    #     # x = self.fc1(x.transpose(1, -1))
    #     # # 反卷积
    #     # x = self.fc2(x).transpose(1, -1)
    #     x = self.act(self.norm5(self.deconv1(x4)))   # torch.Size([1, 512, 23, 12])
    #     x = self.act(self.norm6(self.deconv2(torch.cat((x, x3), dim = 1))))    # torch.Size([1, 256, 46, 23])
    #     x = self.act(self.norm7(self.deconv3(torch.cat((x, x2), dim = 1))))    # torch.Size([1, 128, 90, 46])
    #     x = self.act(self.norm8(self.deconv4(torch.cat((x, x1), dim = 1))))    # torch.Size([1, 64, 180, 91])
   
    #     x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])
      
    #     return x
    
    def forward(self, x):           # (1, 4, 720, 361)
        # 卷积
        x = self.conv_surface(x)       # torch.Size([1, 64, 180, 91])
 
        x = self.conv1(self.norm1(self.act(x)))         #  (b*t, 128, 90, 46)
        x = self.conv2(self.norm2(self.act(x)))         #  (b*t, 256, 46, 23)
        x = self.conv3(self.norm3(self.act(x)))         #  (b*t, 512, 23, 12)
        x = self.conv4(self.norm4(self.act(x)))          #  (b*t, 1024, 12, 6)
        # x = self.fc1(x.transpose(1, -1))
        # # 反卷积
        # x = self.fc2(x).transpose(1, -1)
        x = self.act(self.norm5(self.deconv1(x)))    # torch.Size([1, 512, 23, 12])
        x = self.act(self.norm6(self.deconv2(x)))    # torch.Size([1, 256, 46, 23])
        x = self.act(self.norm7(self.deconv3(x)))    # torch.Size([1, 128, 90, 46])
        x = self.act(self.norm8(self.deconv4(x)))    # torch.Size([1, 64, 180, 91])
   
        x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])
      
        return x
    
class SimpleModel_3(nn.Module):
    def __init__(self):
        super(SimpleModel_3, self).__init__()
        
        # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (7, 7), stride = (4, 4), padding=(3, 3))
            #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 46, 23)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (5, 5), stride = (4, 4), padding=(3, 1))
        #  (b*t, 2 * dim, 46, 23) -> (b*t, 4 * dim, 23, 12)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))
        # (b*t, 4 * dim, 23, 12) ->  (b*t, 4 * dim, 12, 6)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 4096, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))
        
    
        
       # (b*t, 4 * dim, 12, 6) -> (b*t, 4 * dim, 23, 12) 
        self.deconv1 = nn.ConvTranspose2d(in_channels = 4096, out_channels = 256, kernel_size = (3, 4), stride = (2, 2), padding=(1, 1))
        # (b*t, 4 * dim, 23, 12) -> (b*t, 2 * dim, 46, 23) 
        self.deconv2 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = (4, 3), stride = (2, 2), padding=(1, 1))
        # (b*t, 2 * dim, 46, 23) -> (b*t, dim, 180, 91)
        self.deconv3 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (6, 5), stride = (4, 4), padding=(3, 1))
        #  (b*t, dim, 180, 91) -> (b*t, 4, 720, 361)
        self.deconv_surface = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = (8, 7), stride = (4, 4), padding=(2, 3))
        
        # self.act = nn.Sigmoid()
        self.act = nn.ELU()
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=64*4, eps=1e-6, affine=True)
        self.norm4 = nn.GroupNorm(num_groups=32, num_channels=64*4, eps=1e-6, affine=True)
        self.norm5 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        self.norm6 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        # self.act = nn.Identity()
        
        # self.norm1 = TransposeLN(64)
        # self.norm2 = TransposeLN(2*64)
        # self.norm3 = TransposeLN(2*64)
        # self.norm4 = TransposeLN(64)
        # self.norm1 = nn.Identity()
        # self.norm2 = nn.Identity()
        # self.norm3 = nn.Identity()
        # self.norm4 = nn.Identity()
    
    def forward(self, x):           # (1, 4, 720, 361)
        # 卷积
        x = self.conv_surface(x)       # torch.Size([1, 64, 180, 91])
 
        x = self.conv1(self.norm1(self.act(x)))         #  (b*t, 256, 46, 23)
        x = self.conv2(self.norm2(self.act(x)))         #  (b*t, 256, 23, 12)
        x = self.conv3(self.norm3(self.act(x)))         #  (b*t, 512, 12, 6)
        # 反卷积
        x = self.act(self.norm4(self.deconv1(x)))    # torch.Size([1, 256, 23, 12])
        x = self.act(self.norm5(self.deconv2(x)))    # torch.Size([1, 128, 46, 23])
        x = self.act(self.norm6(self.deconv3(x)))    # torch.Size([1, 64, 180, 91])
   
        x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])
      
        return x
    
class SimpleModel_2(nn.Module):
    def __init__(self):
        super(SimpleModel_2, self).__init__()
        
        # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (4, 4), stride = (4, 4), padding=(0, 2))
            #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 46, 23)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 512, kernel_size = (4, 4), stride = (4, 4), padding=(2, 1))
            #  (b*t, 2 * dim, 46, 23) -> (b*t, 4 * dim, 12, 6)
        self.conv2 = nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size = (4, 4), stride = (4, 4), padding=(1, 1))

    
          #  (b*t, 4 * dim, 12, 6) -> (b*t, 2 * dim, 46, 24) 
        self.deconv1 = nn.ConvTranspose2d(in_channels = 4096, out_channels = 512, kernel_size = (4, 4), stride = (4, 4), padding=(1, 0))
        # (b*t, 2 * dim, 46, 24)  -> (b*t, dim, 180, 92)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 64, kernel_size = (4, 4), stride = (4, 4), padding=(2, 2))
        #  (b*t, dim, 180, 92) -> (b*t, 4, 720, 362)
        self.deconv_surface = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = (4, 4), stride = (4, 4), padding=(0, 3))
        
        # self.act = nn.Sigmoid()
        self.act = nn.GELU()
        # self.act = nn.Identity()
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=64*8, eps=1e-6, affine=True)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=64*8, eps=1e-6, affine=True)
        self.norm4 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)

    def forward(self, x):           # (1, 4, 720, 361)
        # 卷积
        x = self.conv_surface(x)       # torch.Size([1, 64, 180, 91])
 
        x = self.conv1(self.norm1(self.act(x)))         #  (b*t, 128, 90, 46)
        x = self.conv2(self.norm2(self.act(x)))         #  (b*t, 256, 46, 23)

        # 反卷积
        x = self.act(self.norm3(self.deconv1(x)))    #torch.Size([1, 128, 90, 46])
        x = self.act(self.norm4(self.deconv2(x)))    # torch.Size([1, 64, 180, 91])

   
        x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])
      
        return x[:,:,:,:-1]
    
class SimpleModel_1(nn.Module):
    def __init__(self):
        super(SimpleModel_1, self).__init__()
        
        # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (7, 7), stride = (8, 8), padding=(3, 3))
        # self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (32, 31), stride = (16, 16), padding=(14, 10))
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 4096, kernel_size = (32, 31), stride = (16, 16), padding=(14, 10))
    
         
        self.deconv1 = nn.ConvTranspose2d(in_channels = 4096, out_channels = 64, kernel_size = (32, 31), stride = (16, 16), padding=(14, 10))
        self.deconv_surface = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = (8, 7), stride = (4, 4), padding=(2, 3))
        self.act = nn.ELU()
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
    
    def forward(self, x):           # (1, 4, 720, 361)
        # 卷积
        x = self.conv_surface(x)       # torch.Size([1, 64, 180, 91])
 
        x = self.conv1(self.norm1(self.act(x)))         #  (b*t, 128, 90, 46)
        # 反卷积
        x = self.act(self.norm2(self.deconv1(x)))    # torch.Size([1, 64, 180, 91])
   
        x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])
      
        return x
    
class SimpleModel_0(nn.Module):
    def __init__(self):
        super(SimpleModel_0, self).__init__()
        
        # (b*t, 4, 720, 361) -> (b*t, dim, 12, 6)
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 4096, kernel_size = (120, 121), stride = (60, 60))
        self.deconv_surface = nn.ConvTranspose2d(in_channels = 4096, out_channels = 4, kernel_size = (120, 121), stride = (60, 60))
        
        # self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 2048, kernel_size=(180, 180), stride=(60, 60), padding=(60, 60))
        # self.deconv_surface = nn.ConvTranspose2d(in_channels = 2048, out_channels = 4, kernel_size=(180, 181), stride=(60, 60), padding=(60, 60))
        
        # Conv2d(4, 4096, kernel_size=(120, 120), stride=(60, 60), padding=(60, 60))
        # (deconv_surface): ConvTranspose2d(4096, 4, kernel_size=(120, 121), stride=(60, 60), padding=(60, 60))
    def forward(self, x):           # (1, 4, 720, 361)
        # 卷积
        x = self.conv_surface(x)       # torch.Size([1, 64, 180, 91])
        x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])

        return x
    
        
class TransposeLN(nn.Module):
    def __init__(self, num_features, dim1=1, dim2=-1):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return self.norm(x.transpose(self.dim1, self.dim2)).transpose(self.dim1, self.dim2)
    
    
def load_static():         
    weather_surface_mean = torch.load("/home/lbk/llm/llama4pde/data/all_surface_mean.pt")   # (4, )
    weather_surface_std = torch.load("/home/lbk/llm/llama4pde/data/all_surface_std.pt")  # (4, )
    weather_upper_mean = torch.load("/home/lbk/llm/llama4pde/data/all_upper_mean.pt")    # (5, 13)
    weather_upper_std = torch.load("/home/lbk/llm/llama4pde/data/all_upper_std.pt")     # (5, 13)
    return weather_surface_mean, weather_surface_std, weather_upper_mean, weather_upper_std

seed = 1234
random.seed(seed)                  # 设置Python标准库中random模块的随机数种子
np.random.seed(seed)               # 设置NumPy库中random模块的随机数种子
torch.manual_seed(seed)            # 设置PyTorch库中的随机数种子
torch.cuda.manual_seed_all(seed)
batch_size = 16
epochs = 500
# accelerator = Accelerator(gradient_accumulation_steps = 1)
accelerator = Accelerator()

phi =  torch.linspace(90, -90, 361)        # 纬度上面(对应721那个维度)
cos_phi = torch.cos((math.pi*phi)/180)
cos_phi = cos_phi * 361 / (torch.sum(cos_phi))
        
# (1, 4, t = 1, 720, 361)
lat_weight_surface = cos_phi.contiguous().reshape(1, 1, 1, 1, 361).repeat(
            1, 4, 1, 720, 1)
lat_weight_surface = lat_weight_surface.to(accelerator.device)    
        
weather_surface_mean, weather_surface_std, weather_upper_mean, weather_upper_std = load_static()
weather_surface_mean = weather_surface_mean[:, None, None].to(accelerator.device)
weather_surface_std = weather_surface_std[:, None, None].to(accelerator.device)
weather_upper_mean = weather_upper_mean[:, :, None, None].to(accelerator.device)
weather_upper_std = weather_upper_std[:, :, None, None].to(accelerator.device)

model = SimpleModel_4()
model.apply(init_weights)
print(model)
myloss = LpLoss()


optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)


surface_data_path = "/data/lbk/pangu_data/cli_download_data/tensored_years_surface_data/surface_data_720_361.pt"

dataset = torch.load(surface_data_path)         # (4383, 4, 720, 361)
# dataset = torch.randn(10, 4, 720, 361)
print("load over!!")

train_dataset = dataset[:4018]
test_dataset = dataset[-365:]

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers=4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 8*len(train_loader), gamma=0.5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=5e-4, 
#                               epochs=epochs, steps_per_epoch=len(train_loader))

model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)

for ep in range(epochs):
    model.train()
    total_loss = 0
    for train_batch_index, surface_data in enumerate(tqdm(train_loader)):               # (1, 4, 720, 361)
        # with accelerator.accumulate(model):
        #     input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
        #     output_norm_surface_data = model(input_norm_surface_data)
            
        #     surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_norm_surface_data.unsqueeze(dim = 2), 
        #                                         target = input_norm_surface_data.unsqueeze(dim = 2), 
        #                                         lat_weight = lat_weight_surface)
        #     loss = torch.mean(surface_MSE_loss)
        #     accelerator.backward(loss)
        #     total_loss += loss.item()
            
            
        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()
     
        input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
        # input_norm_surface_data = surface_data
        output_norm_surface_data = model(input_norm_surface_data)

        
    
        # lat_weight_surface[:,:,:,:,:-1]
        # surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_norm_surface_data.unsqueeze(dim = 2), 
        #                                         target = input_norm_surface_data.unsqueeze(dim = 2), 
        #                                         lat_weight = lat_weight_surface)
        
        # surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_norm_surface_data.unsqueeze(dim = 2), 
        #                                         target = input_norm_surface_data.unsqueeze(dim = 2), 
        #                                         lat_weight = torch.ones(output_norm_surface_data.unsqueeze(dim = 2).shape).to(accelerator.device))
        # accelerator.print("training surface_MSE_loss: ", surface_MSE_loss)

        # (bsz, 4, 720, 361)

        loss = myloss(output_norm_surface_data.flatten(0, 1).flatten(1), input_norm_surface_data.flatten(0, 1).flatten(1)) / (4 * output_norm_surface_data.shape[0])      
        # loss = 0.1 * surface_MAE_loss[0] + 0.4 * surface_MAE_loss[1] + 0.4 * surface_MAE_loss[2] + 0.1 * surface_MAE_loss[3]
        
        
        # loss = torch.mean((input_norm_surface_data - output_norm_surface_data)**2)
        accelerator.backward(loss)
        total_loss += loss.detach().item()
            
            
        optimizer.step()

        optimizer.zero_grad()
        scheduler.step()
    total_loss = total_loss / (train_batch_index + 1)
    accelerator.print(f'epoch {ep}: Learning rate {scheduler.get_lr()[0]}')
    accelerator.print("train_loss:", total_loss)
    
        
        
    model.eval()
    with torch.no_grad():
        total_loss = torch.zeros(4,).to(accelerator.device)
        total_loss_norm = torch.zeros(4,).to(accelerator.device)
        for test_batch_index, surface_data in enumerate(tqdm(test_loader)):               # (1, 4, 720, 361)
            input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
            
       
            output_norm_surface_data = model(input_norm_surface_data)
        
        
            output_surface_data = output_norm_surface_data * weather_surface_std + weather_surface_mean
            surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_surface_data.unsqueeze(dim = 2), 
                                                    target = surface_data.unsqueeze(dim = 2), 
                                                    lat_weight = lat_weight_surface)
            surface_MAE_loss_norm, surface_MSE_loss_norm = cal_mae_mse(logits = output_norm_surface_data.unsqueeze(dim = 2), 
                                                    target = input_norm_surface_data.unsqueeze(dim = 2), 
                                                    lat_weight = lat_weight_surface)
            total_loss_norm += surface_MSE_loss_norm
            # surface_MAE_losses, surface_MSE_losses = accelerator.gather_for_metrics((surface_MAE_loss, surface_MSE_loss))
            total_loss += surface_MSE_loss
            # print(test_batch_index, total_loss[-1])
        total_loss_norm = total_loss_norm / (test_batch_index + 1)
        total_loss = total_loss / (test_batch_index + 1)
        accelerator.print("test_loss:", total_loss, total_loss_norm)


# for ep in range(epochs):
#     model.train()
#     total_loss = 0
#     for train_batch_index, surface_data in enumerate(tqdm(train_loader)):               # (1, 4, 720, 361)
#         # with accelerator.accumulate(model):
#         #     input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
#         #     output_norm_surface_data = model(input_norm_surface_data)
            
#         #     surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_norm_surface_data.unsqueeze(dim = 2), 
#         #                                         target = input_norm_surface_data.unsqueeze(dim = 2), 
#         #                                         lat_weight = lat_weight_surface)
#         #     loss = torch.mean(surface_MSE_loss)
#         #     accelerator.backward(loss)
#         #     total_loss += loss.item()
            
            
#         #     optimizer.step()
#         #     scheduler.step()
#         #     optimizer.zero_grad()

#         output_surface_data = model(surface_data)
        
    
#         # lat_weight_surface[:,:,:,:,:-1]
#         surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_surface_data.unsqueeze(dim = 2), 
#                                                 target = surface_data.unsqueeze(dim = 2), 
#                                                 lat_weight = torch.ones(surface_data.shape).to(accelerator.device))
#         loss = torch.mean(surface_MSE_loss)
#         accelerator.backward(loss)
#         total_loss += loss.item()
            
            
#         optimizer.step()

#         optimizer.zero_grad()
#         scheduler.step()
        
#     total_loss = total_loss / (train_batch_index + 1)
    
#     accelerator.print(f'epoch {ep}: Learning rate {scheduler.get_lr()[0]}')
#     accelerator.print("train_loss:", total_loss)
    
        
        
#     model.eval()
#     with torch.no_grad():
#         total_loss = torch.zeros(4,).to(accelerator.device)
#         for test_batch_index, surface_data in enumerate(tqdm(test_loader)):               # (1, 4, 720, 361)
#             output_surface_data = model(surface_data)
#             surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_surface_data.unsqueeze(dim = 2), 
#                                                     target = surface_data.unsqueeze(dim = 2), 
#                                                     lat_weight = torch.ones(output_surface_data.shape).to(accelerator.device))
            
#             # accelerator.print(surface_MSE_loss)
#             # surface_MAE_losses, surface_MSE_losses = accelerator.gather_for_metrics((surface_MAE_loss, surface_MSE_loss))
#             total_loss += surface_MSE_loss
#             # print(test_batch_index, total_loss[-1])
#         total_loss = total_loss / (test_batch_index + 1)
#         accelerator.print("test_loss:", total_loss)





        
        




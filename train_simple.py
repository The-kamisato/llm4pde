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
from model import complete_model
from data.class_lmdb_dataset import LmdbDataset

class TransposeLN(nn.Module):
    def __init__(self, num_features, dim1=1, dim2=-1):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return self.norm(x.transpose(self.dim1, self.dim2)).transpose(self.dim1, self.dim2)
    
    
def load_static():         
    weather_surface_mean = torch.load("/home/lbk/llm/llama4pde/data/all_surface_mean.pt").cuda()    # (4, )
    weather_surface_std = torch.load("/home/lbk/llm/llama4pde/data/all_surface_std.pt").cuda()    # (4, )
    weather_upper_mean = torch.load("/home/lbk/llm/llama4pde/data/all_upper_mean.pt").cuda()       # (5, 13)
    weather_upper_std = torch.load("/home/lbk/llm/llama4pde/data/all_upper_std.pt").cuda()      # (5, 13)
    return weather_surface_mean, weather_surface_std, weather_upper_mean, weather_upper_std

random.seed(1234)                  # 设置Python标准库中random模块的随机数种子
np.random.seed(1234)               # 设置NumPy库中random模块的随机数种子
torch.manual_seed(1234)            # 设置PyTorch库中的随机数种子
torch.cuda.manual_seed_all(1234)
weather_surface_mean, weather_surface_std, weather_upper_mean, weather_upper_std = load_static()
weather_surface_mean = weather_surface_mean[:, None, None]
weather_surface_std = weather_surface_std[:, None, None]
weather_upper_mean = weather_upper_mean[:, :, None, None]
weather_upper_std = weather_upper_std[:, :, None, None]

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (7, 7), stride = (4, 4), padding=(3, 3))
            #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 90, 46)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))
            #  (b*t, 2 * dim, 90, 46) -> (b*t, 4 * dim, 46, 23)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = (2, 2), padding=(2, 1))
        
          #  (b*t, 4 * dim, 46, 23) -> (b*t, 2 * dim, 90, 46) 
        self.deconv1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = (4, 4), stride = (2, 2), padding=(2, 1))
        # (b*t, 2 * dim, 90, 46) -> (b*t, dim, 180, 91)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (4, 3), stride = (2, 2), padding=(1, 1))
        #  (b*t, dim, 180, 91) -> (b*t, 4, 720, 361)
        self.deconv_surface = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = (8, 7), stride = (4, 4), padding=(2, 3))
        
        # self.act = nn.GELU()
        
        # self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        # self.norm2 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        # self.norm3 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        # self.norm4 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        self.act = nn.Identity()
        
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.norm3 = nn.Identity()
        self.norm4 = nn.Identity()
    
    def forward(self, x):           # (1, 4, 720, 361)
        # 卷积
        x = self.conv_surface(x)       # torch.Size([1, 64, 180, 91])
 
        x = self.conv1(self.norm1(self.act(x)))         #  (b*t, 128, 90, 46)
        x = self.conv2(self.norm2(self.act(x)))         #  (b*t, 256, 46, 23)
        # 反卷积
        x = self.act(self.norm3(self.deconv1(x)))    # torch.Size([1, 256, 180, 91])
        x = self.act(self.norm4(self.deconv2(x)))    # torch.Size([1, 128, 180, 91])
   
        x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])
      
        return x
    
model = SimpleModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4018*4, gamma=0.5)
surface_data_path = "/data/lbk/pangu_data/cli_download_data/tensored_years_surface_data/surface_data_720_361.pt"
print("load over!!")
dataset = torch.load(surface_data_path)         # (4383, 4, 720, 361)

train_dataset = dataset[:4018]
test_dataset = dataset[-365:]

train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

for ep in range(20):
    model.train()
    total_loss = 0
    for surface_data in tqdm(train_loader):               # (1, 4, 720, 361)
        surface_data = surface_data.cuda()
        input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
        output_norm_surface_data = model(input_norm_surface_data)
        
        loss = torch.sqrt(torch.mean((output_norm_surface_data - input_norm_surface_data) ** 2))
        loss.backward()
        total_loss += loss.item()
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
    total_loss = total_loss / 4018
    print("train_loss:", total_loss)
        
        
    model.eval()
    total_loss = 0
    for surface_data in test_loader:               # (1, 4, 720, 361)
        surface_data = surface_data.cuda()
        input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
        output_norm_surface_data = model(input_norm_surface_data)
        output_surface_data = output_norm_surface_data * weather_surface_std + weather_surface_mean
        
        loss = torch.sqrt(torch.mean((output_surface_data - surface_data) ** 2, dim = (0, 2, 3)))
        total_loss += loss
        
    total_loss = total_loss / 365
    print("test_loss:", total_loss)
        
        



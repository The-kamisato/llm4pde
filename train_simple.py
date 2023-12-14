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
from metric import cal_mae_mse


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
batch_size = 1

accelerator = Accelerator(gradient_accumulation_steps = 1)

phi =  torch.linspace(90, -90, 361)        # 纬度上面(对应721那个维度)
cos_phi = torch.cos((math.pi*phi)/180)
cos_phi = cos_phi * 361 / (torch.sum(cos_phi))
        
# (1, 4, t = 1, 720, 361)
lat_weight_surface = cos_phi.contiguous().reshape(1, 1, 1, 1, 361).repeat(
            batch_size, 4, 1, 720, 1)
lat_weight_surface = lat_weight_surface.to(accelerator.device)    
        
weather_surface_mean, weather_surface_std, weather_upper_mean, weather_upper_std = load_static()
weather_surface_mean = weather_surface_mean[:, None, None].to(accelerator.device)
weather_surface_std = weather_surface_std[:, None, None].to(accelerator.device)
weather_upper_mean = weather_upper_mean[:, :, None, None].to(accelerator.device)
weather_upper_std = weather_upper_std[:, :, None, None].to(accelerator.device)



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
        
        # self.act = nn.Sigmoid()
        self.act = nn.ELU()
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        self.norm4 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
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
 
        x = self.conv1(self.norm1(self.act(x)))         #  (b*t, 128, 90, 46)
        x = self.conv2(self.norm2(self.act(x)))         #  (b*t, 256, 46, 23)
        # 反卷积

        x = self.act(self.norm3(self.deconv1(x)))    # torch.Size([1, 256, 180, 91])
        x = self.act(self.norm4(self.deconv2(x)))    # torch.Size([1, 128, 180, 91])
   
        x = self.deconv_surface(x)      # torch.Size([1, 4, 720, 361])
      
        return x
    
model = SimpleModel()

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)


# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1600, gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
surface_data_path = "/data/lbk/pangu_data/cli_download_data/tensored_years_surface_data/surface_data_720_361.pt"
print("load over!!")
dataset = torch.load(surface_data_path)         # (4383, 4, 720, 361)

train_dataset = dataset[:4018]
test_dataset = dataset[-365:]

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

model, optimizer, train_loader, test_loader = accelerator.prepare(model, optimizer, train_loader, test_loader)
for ep in range(30):
    model.train()
    total_loss = 0
    for train_batch_index, surface_data in enumerate(tqdm(train_loader)):               # (1, 4, 720, 361)
        with accelerator.accumulate(model):
            input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
            output_norm_surface_data = model(input_norm_surface_data)
            
            surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_norm_surface_data.unsqueeze(dim = 2), 
                                                target = input_norm_surface_data.unsqueeze(dim = 2), 
                                                lat_weight = lat_weight_surface)
            loss = torch.mean(surface_MAE_loss)
            accelerator.backward(loss)
            total_loss += loss.item()
            accelerator.print(f'epoch {ep}: Learning rate {scheduler.get_lr()[0]}')
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    total_loss = total_loss / (train_batch_index + 1)
    accelerator.print("train_loss:", total_loss)
    
        
        
    model.eval()
    total_loss = 0
    for test_batch_index, surface_data in enumerate(test_loader):               # (1, 4, 720, 361)
        input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
        output_norm_surface_data = model(input_norm_surface_data)
        output_surface_data = output_norm_surface_data * weather_surface_std + weather_surface_mean
        
        _, surface_MSE_loss = cal_mae_mse(logits = output_surface_data.unsqueeze(dim = 2), 
                                                target = surface_data.unsqueeze(dim = 2), 
                                                lat_weight = lat_weight_surface)
        
        
        surface_MSE_losses = accelerator.gather_for_metrics((surface_MSE_loss))
        total_loss += surface_MSE_losses
    total_loss = total_loss / (test_batch_index + 1)
    accelerator.print("test_loss:", total_loss)
        
        



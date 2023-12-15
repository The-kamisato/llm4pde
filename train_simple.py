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

class SimpleModel_4(nn.Module):
    def __init__(self):
        super(SimpleModel_4, self).__init__()
        
        # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (9, 5), stride = (4, 4), padding=(4, 2))
            #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 90, 46)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (5, 3), stride = (2, 2), padding=(2, 1))
            #  (b*t, 2 * dim, 90, 46) -> (b*t, 4 * dim, 46, 23)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = (2, 2), padding=(2, 1))
        #  (b*t, 2 * dim, 46, 23) -> (b*t, 4 * dim, 23, 12)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (5, 3), stride = (2, 2), padding=(2, 1))
        # (b*t, 4 * dim, 23, 12) ->  (b*t, 4 * dim, 12, 6)
        self.conv4 = nn.Conv2d(in_channels = 512, out_channels = 4096, kernel_size = (5, 3), stride = (2, 2), padding=(2, 1))
        
        # (b*t, 4 * dim, 12, 6) -> (b*t, 4 * dim, 23, 12) 
        self.deconv1 = nn.ConvTranspose2d(in_channels = 4096, out_channels = 512, kernel_size = (5, 4), stride = (2, 2), padding=(2, 1))
        # (b*t, 4 * dim, 23, 12) -> (b*t, 2 * dim, 46, 23) 
        self.deconv2 = nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = (6, 3), stride = (2, 2), padding=(2, 1))
          #  (b*t, 4 * dim, 46, 23) -> (b*t, 2 * dim, 90, 46) 
        self.deconv3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = (4, 4), stride = (2, 2), padding=(2, 1))
        # (b*t, 2 * dim, 90, 46) -> (b*t, dim, 180, 91)
        self.deconv4 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (6, 3), stride = (2, 2), padding=(2, 1))
        #  (b*t, dim, 180, 91) -> (b*t, 4, 720, 361)
        self.deconv_surface = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = (12, 5), stride = (4, 4), padding=(4, 2))
        
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU()
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=64*4, eps=1e-6, affine=True)
        self.norm4 = nn.GroupNorm(num_groups=32, num_channels=64*8, eps=1e-6, affine=True)
        self.norm5 = nn.GroupNorm(num_groups=32, num_channels=64*8, eps=1e-6, affine=True)
        self.norm6 = nn.GroupNorm(num_groups=32, num_channels=64*4, eps=1e-6, affine=True)
        self.norm7 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        self.norm8 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
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
        x = self.conv3(self.norm3(self.act(x)))         #  (b*t, 512, 23, 12)
        x = self.conv4(self.norm4(self.act(x)))         #  (b*t, 1024, 12, 6)
        # 反卷积
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
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (7, 7), stride = (4, 4), padding=(3, 3))
            #  (b*t, dim, 180, 91) -> (b*t, 2 * dim, 46, 23)
        self.conv1 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (5, 5), stride = (4, 4), padding=(3, 1))
            #  (b*t, 2 * dim, 46, 23) -> (b*t, 4 * dim, 12, 6)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 4096, kernel_size = (5, 5), stride = (4, 4), padding=(2, 1))

    
          #  (b*t, 4 * dim, 12, 6) -> (b*t, 2 * dim, 46, 23) 
        self.deconv1 = nn.ConvTranspose2d(in_channels = 4096, out_channels = 128, kernel_size = (6, 5), stride = (4, 4), padding=(2, 1))
        # (b*t, 2 * dim, 46, 23)  -> (b*t, dim, 180, 91)
        self.deconv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (6, 5), stride = (4, 4), padding=(3, 1))
        #  (b*t, dim, 180, 91) -> (b*t, 4, 720, 361)
        self.deconv_surface = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = (8, 7), stride = (4, 4), padding=(2, 3))
        
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU()
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=64, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
        self.norm3 = nn.GroupNorm(num_groups=32, num_channels=64*2, eps=1e-6, affine=True)
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
      
        return x
    
class SimpleModel_1(nn.Module):
    def __init__(self):
        super(SimpleModel_1, self).__init__()
        
        # (b*t, 4, 720, 361) -> (b*t, dim, 180, 91)
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (7, 7), stride = (4, 4), padding=(3, 3))
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
        self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 4096, kernel_size = (60, 60), stride = (60, 60))
        self.deconv_surface = nn.ConvTranspose2d(in_channels = 4096, out_channels = 4, kernel_size = (60, 61), stride = (60, 60))
        
        # self.conv_surface = nn.Conv2d(in_channels = 4, out_channels = 4096, kernel_size=(120, 120), stride=(60, 60), padding=(60, 60))
        # self.deconv_surface = nn.ConvTranspose2d(in_channels = 4096, out_channels = 4, kernel_size=(120, 121), stride=(60, 60), padding=(60, 60))
        
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
    weather_surface_mean = torch.load("/home/lbk/llm/llama4pde/data/all_surface_mean.pt").cuda()    # (4, )
    weather_surface_std = torch.load("/home/lbk/llm/llama4pde/data/all_surface_std.pt").cuda()    # (4, )
    weather_upper_mean = torch.load("/home/lbk/llm/llama4pde/data/all_upper_mean.pt").cuda()       # (5, 13)
    weather_upper_std = torch.load("/home/lbk/llm/llama4pde/data/all_upper_std.pt").cuda()      # (5, 13)
    return weather_surface_mean, weather_surface_std, weather_upper_mean, weather_upper_std

seed = 1234
random.seed(seed)                  # 设置Python标准库中random模块的随机数种子
np.random.seed(seed)               # 设置NumPy库中random模块的随机数种子
torch.manual_seed(seed)            # 设置PyTorch库中的随机数种子
torch.cuda.manual_seed_all(seed)
batch_size = 1
epochs = 50
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
print(model)


optimizer = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)


surface_data_path = "/data/lbk/pangu_data/cli_download_data/tensored_years_surface_data/surface_data_720_361.pt"
print("load over!!")
dataset = torch.load(surface_data_path)         # (4383, 4, 720, 361)

train_dataset = dataset[:4018]
test_dataset = dataset[-365:]

train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 4, gamma=0.5)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=2e-3, 
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
        output_norm_surface_data = model(input_norm_surface_data)
        
    
        # lat_weight_surface[:,:,:,:,:-1]
        surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_norm_surface_data.unsqueeze(dim = 2), 
                                                target = input_norm_surface_data.unsqueeze(dim = 2), 
                                                lat_weight = lat_weight_surface)
        loss = torch.mean(surface_MAE_loss)
        accelerator.backward(loss)
        total_loss += loss.item()
            
            
        optimizer.step()

        optimizer.zero_grad()
    scheduler.step()
    total_loss = total_loss / (train_batch_index + 1)
    accelerator.print(f'epoch {ep}: Learning rate {scheduler.get_lr()[0]}')
    accelerator.print("train_loss:", total_loss)
    
        
        
    model.eval()
    with torch.no_grad():
        total_loss = torch.zeros(4,).to(accelerator.device)
        for test_batch_index, surface_data in enumerate(tqdm(test_loader)):               # (1, 4, 720, 361)
            input_norm_surface_data = (surface_data - weather_surface_mean) / weather_surface_std
            output_norm_surface_data = model(input_norm_surface_data)
            output_surface_data = output_norm_surface_data * weather_surface_std + weather_surface_mean
            surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = output_surface_data.unsqueeze(dim = 2), 
                                                    target = surface_data.unsqueeze(dim = 2), 
                                                    lat_weight = lat_weight_surface)
            
            # accelerator.print(surface_MSE_loss)
            # surface_MAE_losses, surface_MSE_losses = accelerator.gather_for_metrics((surface_MAE_loss, surface_MSE_loss))
            total_loss += surface_MSE_loss
            # print(test_batch_index, total_loss[-1])
        total_loss = total_loss / (test_batch_index + 1)
        accelerator.print("test_loss:", total_loss)

        
        



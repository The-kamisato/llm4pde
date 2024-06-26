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
import wandb
wandb.login()

from metric import cal_mae_mse
from losses import LpLoss
from model import complete_model
from data.class_lmdb_dataset import LmdbDataset

ckpt_foler = "/data/lbk"

def parse_args():
    parser = argparse.ArgumentParser()
    ## 
    parser.add_argument('--gpu', nargs='+', type=int, default=[], help='GPU device indices')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    ## data_args
    parser.add_argument('--S0', type=int, default=13)
    parser.add_argument('--S1', type=int, default=1440)
    parser.add_argument('--S2', type=int, default=721)
    parser.add_argument('--downsample', type=int, default=2)
    
    ## train_args
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--ntrain', type=int, default=4018)
    parser.add_argument('--ntest', type=int, default=359)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--scheduler_type', type=str, default="steplr")
    parser.add_argument('--scheduler_step', type=int, default=10*4018)
    parser.add_argument('--scheduler_gamma', type=float, default=0.5)
    parser.add_argument('--train_object', type=str, default="l2")
    # parser.add_argument('--lora_r', type=int, default=None)
    # parser.add_argument('--lora_alpha', type=float, default=None)
    parser.add_argument('--lora_r_llama', type=int, default=8)
    parser.add_argument('--lora_alpha_llama', type=float, default=16)
    parser.add_argument('--train_auto', action='store_true', default=False)
    parser.add_argument('--save_ckpt_epoch', type=int, default=10)
    
    ## model_args
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--final_dim', type=int, default=4096)
    parser.add_argument('--surface_in_chans', type=int, default=4)
    parser.add_argument('--upper_in_chans', type=int, default=5)
    parser.add_argument('--if_surface_const_mask', type=bool, default=True)
    parser.add_argument('--act_type', type=str, default="gelu")
    parser.add_argument('--norm_type', type=str, default="gn")
    parser.add_argument('--frozen_enc_dec', action='store_true', default=False)
    parser.add_argument('--load_pretrained_enc_dec', action='store_true', default=False)
    parser.add_argument('--llama_body', action='store_true', default=False)
    parser.add_argument('--frozen_llama', action='store_true', default=False)
    parser.add_argument('--time_embed', action='store_true', default=False)
    parser.add_argument('--pos_embed', action='store_true', default=False)
    
    args = parser.parse_args()
    return args

def set_visible_devices(gpu_indices):
    if gpu_indices:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in gpu_indices)
    else:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        print(name)
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def load_static():         
    weather_surface_mean = torch.load("/home/bokai/llm/llama4pde/data/all_surface_mean.pt")  # (4, )
    weather_surface_std = torch.load("/home/bokai/llm/llama4pde/data/all_surface_std.pt")   # (4, )
    weather_upper_mean = torch.load("/home/bokai/llm/llama4pde/data/all_upper_mean.pt")      # (5, 13)
    weather_upper_std = torch.load("/home/bokai/llm/llama4pde/data/all_upper_std.pt")       # (5, 13)
    return weather_surface_mean, weather_surface_std, weather_upper_mean, weather_upper_std

class Llama4PdeTrainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator(gradient_accumulation_steps = args.accumulation_steps)
        
        self.train_loader, self.test_loader = self.data_loader()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        
        self.lat_weight_surface, self.lat_weight_upper = self.get_lat_weight()
        self.lat_weight_surface, self.lat_weight_upper = self.lat_weight_surface.to(self.accelerator.device), self.lat_weight_upper.to(self.accelerator.device)
        self.weather_surface_mean, self.weather_surface_std, self.weather_upper_mean, self.weather_upper_std = load_static()
        self.weather_surface_mean = self.weather_surface_mean[:, None, None].to(self.accelerator.device)
        self.weather_surface_std = self.weather_surface_std[:, None, None].to(self.accelerator.device)
        self.weather_upper_mean = self.weather_upper_mean[:, :, None, None].to(self.accelerator.device)
        self.weather_upper_std = self.weather_upper_std[:, :, None, None].to(self.accelerator.device)
        
        
        self.model, self.optimizer, self.train_loader, self.test_loader = self.accelerator.prepare(
                                                                            self.model, self.optimizer, self.train_loader, self.test_loader)
        
    def normalize(self, surface_u, upper_u):
        '''
        surface_u.shape = torch.Size([bsz, T, C, H, W])
        upper_u.shape = torch.Size([bsz, T, C, p, H, W])
        weather_surface_mean, weather_surface_std.shape = torch.Size([C, 1, 1])
        weather_upper_mean, weather_upper_std.shape = torch.Size([C, p, 1, 1])
        '''
        normalized_surface_u = ((surface_u - self.weather_surface_mean) / self.weather_surface_std)
        normalized_upper_u = ((upper_u - self.weather_upper_mean) / self.weather_upper_std)
        
        return normalized_surface_u, normalized_upper_u

    def unnormalize(self, normalized_surface_u, normalized_upper_u):
        
        surface_u = (normalized_surface_u * self.weather_surface_std + self.weather_surface_mean)
        upper_u = (normalized_upper_u * self.weather_upper_std + self.weather_upper_mean)
        
        return surface_u, upper_u

    def get_model(self):
        print("model building...")
        
        if self.args.llama_body:
            # llama的配置
            n_token_per_time = 72 
            
            llama_config = LlamaConfig()
            llama_config.n_token_per_time = n_token_per_time
            
            lora_llama_config = LoraConfig(
                    r=self.args.lora_r_llama,
                    lora_alpha=self.args.lora_alpha_llama,
                    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'],
                    fan_in_fan_out=False,
                    lora_dropout=0.05,
                    inference_mode=False,
                    bias="none",
                )
            model = complete_model(surface_in_chans = self.args.surface_in_chans, upper_in_chans = self.args.upper_in_chans, dim = self.args.dim, final_dim = self.args.final_dim, 
                        if_surface_const_mask = self.args.if_surface_const_mask, act_type=self.args.act_type, norm_type=self.args.norm_type, 
                        frozen_enc_dec = self.args.frozen_enc_dec, load_pretrained_enc_dec = self.args.load_pretrained_enc_dec, 
                        llama_body = True, llama_config = llama_config, llama_lora_config = lora_llama_config, 
                        frozen_llama = self.args.frozen_llama, time_embed = self.args.time_embed, pos_embed = self.args.pos_embed)

        else:
            model = complete_model(surface_in_chans = self.args.surface_in_chans, upper_in_chans = self.args.upper_in_chans, dim = self.args.dim, final_dim = self.args.final_dim, 
                        if_surface_const_mask = self.args.if_surface_const_mask, act_type=self.args.act_type, norm_type=self.args.norm_type, 
                        frozen_enc_dec = self.args.frozen_enc_dec, load_pretrained_enc_dec = self.args.load_pretrained_enc_dec, llama_body = False)
            
        return model

    def get_optimizer(self):
        print("set optimizer...")
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        return optimizer
    
    def get_scheduler(self):
        def _get_cosine_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float):
            if current_step < num_warmup_steps:                                             # 正处于预热阶段，lr正在上升
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))        # 计算预热之后的训练的进度
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        def get_cosine_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
            """
            Create a schedule with a learning rate that decreases following the values of the cosine function between the
            initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
            initial lr set in the optimizer.

            Args:
                optimizer ([`~torch.optim.Optimizer`]):
                    The optimizer for which to schedule the learning rate.
                num_warmup_steps (`int`):
                    The number of steps for the warmup phase.
                num_training_steps (`int`):
                    The total number of training steps.
                num_cycles (`float`, *optional*, defaults to 0.5):
                    The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
                    following a half-cosine).
                last_epoch (`int`, *optional*, defaults to -1):
                    The index of the last epoch when resuming training.

            Return:
                `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
            """

            lr_lambda = partial(
                _get_cosine_schedule_with_warmup_lr_lambda,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )
            # torch.optim.lr_scheduler.LambdaLR 是 PyTorch 库中的一种学习率调度器，它使用一个 lambda 函数作为学习率更新的规则。这个 lambda 函数将接收优化器的状态（例如，当前步数）作为输入，并返回一个更新后的学习率
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch) 
        
        print("set scheduler...")
        if self.args.scheduler_type == "cosine_warmup":
            scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=(0.1 * self.args.epochs * self.args.ntrain) / (self.args.accumulation_steps * len(self.args.gpu)), 
                                                num_training_steps = ((self.args.epochs * self.args.ntrain) / (self.args.accumulation_steps * len(self.args.gpu))) + 10, num_cycles = 0.5)
            
        if self.args.scheduler_type == "cosine_anneal":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10, eta_min=0)
            
        if self.args.scheduler_type == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.scheduler_step, gamma=self.args.scheduler_gamma)
            
        if self.args.scheduler_type == "onecycle":  
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.optimizer, max_lr=2e-4, 
                              epochs=self.args.epochs, steps_per_epoch=len(self.train_loader))
            
        return scheduler
    
    def data_loader(self):
        print("set data_loader...")
        dataset = LmdbDataset()
        
        print(len(dataset))
        
        train_dataset = Subset(dataset, indices = range(self.args.ntrain - 1))
        # test_dataset = Subset(dataset, indices = range(self.args.ntrain, len(dataset) - 6))
        test_dataset = Subset(dataset, indices = range(self.args.ntrain, self.args.ntrain + self.args.ntest))
      
        train_loader = DataLoader(train_dataset, batch_size = self.args.batch_size, shuffle = False, num_workers = 4)
        test_loader = DataLoader(test_dataset, batch_size = self.args.batch_size, shuffle = False, num_workers = 4)
        
        print(len(train_loader))
        print(len(test_loader))
        return train_loader, test_loader
    
    def get_lat_weight(self):
        phi =  torch.linspace(90, -90, math.ceil(self.args.S2 / self.args.downsample))        # 纬度上面(对应721那个维度)
        cos_phi = torch.cos((math.pi*phi)/180)
        cos_phi = cos_phi * math.ceil(self.args.S2 / self.args.downsample) / (torch.sum(cos_phi))
        
        # (1, 4, t = 1, 720, 361)
        lat_weight_surface = cos_phi.contiguous().reshape(1, 1, 1, 1, math.ceil(self.args.S2 / self.args.downsample)).repeat(
            self.args.batch_size, self.args.surface_in_chans, 1, math.ceil(self.args.S1 / self.args.downsample), 1)
        
        # (1, 5, t = 1, 13, 720, 361), 作自回归的时候t=time, 用了一个repeat
        lat_weight_upper = cos_phi.contiguous().reshape(1, 1, 1, 1, 1, math.ceil(self.args.S2 / self.args.downsample)).repeat(
            self.args.batch_size, self.args.upper_in_chans, 1, self.args.S0, math.ceil(self.args.S1 / self.args.downsample), 1)

        return lat_weight_surface, lat_weight_upper
    
    
    def train_epoch(self):
        self.model.train()
        
        total_loss = 0
        for train_batch_idx, batch in enumerate(tqdm(self.train_loader)):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                train_surface_u, train_upper_u = batch
                # 归一化
                normalized_train_surface_u, normalized_train_upper_u = self.normalize(train_surface_u, train_upper_u)
                
                # 取出相应的时间作为输入
                normalized_surface_u_input = normalized_train_surface_u[:, :-1, :, :, :]
                normalized_upper_u_input = normalized_train_upper_u[:, :-1, :, :, :, :]

                
                # 经过模型并计算出相应的loss
                normalized_surface_u_output, normalized_upper_u_output = self.model(normalized_surface_u_input, normalized_upper_u_input)       # (batch, T, C, H, W)
                
                if self.args.train_object == "mae" or self.args.train_object == "mse":
                    surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = normalized_surface_u_output[:, -1:, :, :, :].transpose(1,2), 
                                                    target = normalized_train_surface_u[:, -1:, :, :, :].transpose(1,2), 
                                                    lat_weight = self.lat_weight_surface)
                    
                    upper_MAE_all_loss, upper_MSE_all_loss, _, _ = cal_mae_mse(logits = normalized_upper_u_output[:, -1:, :, :, :, :].transpose(1,2), 
                                                    target = normalized_train_upper_u[:, -1:, :, :, :, :].transpose(1,2), 
                                                    lat_weight = self.lat_weight_upper)
                    
                    wandb.log({"surface_l2_loss": torch.mean(surface_MSE_loss).item()})
                    wandb.log({"upper_l2_loss": torch.mean(upper_MSE_all_loss).item()})
                    
                    if self.args.train_object == "mae":
                        loss = 0.25 * torch.mean(surface_MAE_loss) + torch.mean(upper_MAE_all_loss)
                    if self.args.train_object == "mse":
                        loss = 0.25 * torch.mean(surface_MSE_loss) + torch.mean(upper_MSE_all_loss)
                        
                    wandb.log({"loss": torch.mean(loss).item()})
                if self.args.train_object == "l2":
                    myloss = LpLoss()
                    surface_l2_loss = myloss(normalized_surface_u_output[:, -1:, :, :, :].flatten(0, 2).flatten(1), 
                                            normalized_train_surface_u[:, -1:, :, :, :].flatten(0, 2).flatten(1)) / (normalized_surface_u_output[:, -1:, :, :, :].flatten(0, 2).shape[0])
                    upper_l2_loss = myloss(normalized_upper_u_output[:, -1:, :, :, :, :].flatten(0, 2).flatten(1), 
                                            normalized_train_upper_u[:, -1:, :, :, :, :].flatten(0, 2).flatten(1)) / (normalized_upper_u_output[:, -1:, :, :, :, :].flatten(0, 2).shape[0])
                    loss = 0.25 * surface_l2_loss + upper_l2_loss
                    
                    wandb.log({"surface_l2_loss": surface_l2_loss.item()})
                    wandb.log({"upper_l2_loss": upper_l2_loss.item()})
                    wandb.log({"training_l2_loss": loss.item()})
                    
                total_loss += loss.detach().item()
            
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.unscale_gradients(self.optimizer)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.step()
                self.scheduler.step()
                
        average_loss = total_loss / (train_batch_idx + 1)
        return average_loss
                
    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_surface_MSE_loss = torch.zeros(4, ).to(self.accelerator.device)
            total_upper_MSE_height_loss = torch.zeros(5, 13).to(self.accelerator.device)
            
            for test_batch_idx, batch in enumerate(tqdm(self.test_loader)):
                test_surface_u, test_upper_u = batch            # 都是前6天预测第7天
                
                # 归一化
                normalized_test_surface_u, normalized_test_upper_u = self.normalize(test_surface_u, test_upper_u)
                
                # 取出相应的时间
                normalized_surface_u_input = normalized_test_surface_u[:, :-1, :, :, :]
                normalized_upper_u_input = normalized_test_upper_u[:, :-1, :, :, :, :]
                
                normalized_surface_u_output, normalized_upper_u_output = self.model(normalized_surface_u_input, normalized_upper_u_input)
                
                # 去归一化
                surface_u_output, upper_u_output = self.unnormalize(normalized_surface_u_output, normalized_upper_u_output) 
                
                # 计算loss
                _, surface_MSE_loss = cal_mae_mse(logits = surface_u_output[:, -1:, :, :, :].transpose(1,2), 
                                                                target = test_surface_u[:, -1:, :, :, :].transpose(1,2), 
                                                                lat_weight = self.lat_weight_surface)
                
                _, _, _, upper_MSE_height_loss = cal_mae_mse(
                                                    logits = upper_u_output[:, -1:, :, :, :, :].transpose(1,2), 
                                                    target = test_upper_u[:, -1:, :, :, :, :].transpose(1,2), 
                                                    lat_weight = self.lat_weight_upper)
                
                wandb.log({"training Z500": upper_MSE_height_loss[0, 5], 
                           "training T850": upper_MSE_height_loss[2, 2],
                           "training T2M": surface_MSE_loss[3],
                           "training U10": surface_MSE_loss[1]})
        
                
                total_surface_MSE_loss += surface_MSE_loss
                total_upper_MSE_height_loss += upper_MSE_height_loss
        
        average_surface_MSE_loss = total_surface_MSE_loss / (test_batch_idx + 1)
        average_upper_MSE_height_loss = total_upper_MSE_height_loss / (test_batch_idx + 1)
        
        average_surface_MSE_losses, average_upper_MSE_height_losses = self.accelerator.gather_for_metrics((average_surface_MSE_loss, average_upper_MSE_height_loss))
      
        return average_surface_MSE_losses, average_upper_MSE_height_losses
    
    def train_auto_epoch(self):
        self.model.train()
        
        total_loss = 0
        for train_batch_idx, batch in enumerate(tqdm(self.train_loader)):
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad()
                train_surface_u, train_upper_u = batch      # (1, time, 4, 720, 361)      (1, time, 5, 13, 720, 361)
                
                normalized_train_surface_u, normalized_train_upper_u = self.normalize(train_surface_u, train_upper_u)
                
                # (1, time, 4, 720, 361) + (1, time, 5, 13, 720, 361) -卷积-> (bsz = 1, 12 * 6 * time, 4096) -> (1, time, 4, 720, 361) + (1, time, 5, 13, 720, 361)
                normalized_surface_u_output, normalized_upper_u_output = self.model(normalized_train_surface_u, normalized_train_upper_u)
                
                if self.args.train_object == "mae" or self.args.train_object == "mse":
                    # cal_mae_mse函数里要求(batch, T, C, H, W)需要转为 (batch, C, T, H, W)
                    surface_MAE_loss, surface_MSE_loss = cal_mae_mse(logits = normalized_surface_u_output.transpose(1, 2), 
                                                target = normalized_train_surface_u.transpose(1, 2), 
                                                lat_weight = self.lat_weight_surface.repeat(1, 1, normalized_surface_u_output.shape[1], 1, 1))
                    # cal_mae_mse函数里要求(batch, T, C, p, H, W)需要转为 (batch, C, T, p, H, W)
                    upper_MAE_all_loss, upper_MSE_all_loss, _, _ = cal_mae_mse(logits = normalized_upper_u_output.transpose(1, 2), 
                                                 target = normalized_train_upper_u.transpose(1, 2), 
                                                 lat_weight = self.lat_weight_upper.repeat(1, 1, normalized_upper_u_output.shape[1], 1, 1, 1))
                    
                    if self.args.train_object == "mse":
                        loss = 0.25 * torch.mean(surface_MAE_loss) + torch.mean(upper_MAE_all_loss)
                    if self.args.train_object == "mse":
                        loss = 0.25 * torch.mean(surface_MSE_loss) + torch.mean(upper_MSE_all_loss)
                        
                    wandb.log({"training_surface_loss": torch.mean(surface_MSE_loss).item()})
                    wandb.log({"training_upper_loss": torch.mean(upper_MSE_all_loss).item()})
                
                if self.args.train_object == "l2":
                    myloss = LpLoss()
                    surface_l2_loss = myloss(normalized_surface_u_output.flatten(0, 2).flatten(1), 
                                            normalized_train_surface_u.flatten(0, 2).flatten(1)) / (normalized_surface_u_output.flatten(0, 2).shape[0])
                    upper_l2_loss = myloss(normalized_upper_u_output.flatten(0, 2).flatten(1), 
                                            normalized_train_upper_u.flatten(0, 2).flatten(1)) / (normalized_upper_u_output.flatten(0, 2).shape[0])
                    loss = 0.25 * surface_l2_loss + upper_l2_loss
                    
                    wandb.log({"surface_l2_loss": surface_l2_loss.item()})
                    wandb.log({"upper_l2_loss": upper_l2_loss.item()})
                    wandb.log({"training_l2_loss": loss.item()})
                    
                total_loss += loss.detach().item()
                
                # self.accelerator.print("index:", train_batch_idx, "loss when training: ", loss.item(), loss.device)
                
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.unscale_gradients(self.optimizer)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.optimizer.step()
                self.scheduler.step()
        
        average_loss = total_loss / (train_batch_idx + 1)
        
        return average_loss
    
    def test_auto_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_surface_MSE_loss = torch.zeros(4, ).to(self.accelerator.device)
            total_upper_MSE_height_loss = torch.zeros(5, 13).to(self.accelerator.device)
            
            for test_batch_idx, batch in enumerate(tqdm(self.test_loader)):
                test_surface_u, test_upper_u = batch            
                
                # 归一化
                normalized_test_surface_u, normalized_test_upper_u = self.normalize(test_surface_u, test_upper_u)
                
                normalized_surface_u_output, normalized_upper_u_output = self.model(normalized_test_surface_u, normalized_test_upper_u)
                
                # 去归一化
                surface_u_output, upper_u_output = self.unnormalize(normalized_surface_u_output, normalized_upper_u_output) 
                
                # 计算loss
                _, surface_MSE_loss = cal_mae_mse(logits = surface_u_output.transpose(1,2), 
                                                    target = test_surface_u.transpose(1,2), 
                                                    lat_weight = self.lat_weight_surface.repeat(1, 1, surface_u_output.shape[1], 1, 1))
                
                _, _, _, upper_MSE_height_loss = cal_mae_mse(
                                                    logits = upper_u_output.transpose(1,2), 
                                                    target = test_upper_u.transpose(1,2),
                                                    lat_weight = self.lat_weight_upper.repeat(1, 1, upper_u_output.shape[1], 1, 1, 1))

                
                total_surface_MSE_loss += surface_MSE_loss
                total_upper_MSE_height_loss += upper_MSE_height_loss
        
        average_surface_MSE_loss = total_surface_MSE_loss / (test_batch_idx + 1)
        average_upper_MSE_height_loss = total_upper_MSE_height_loss / (test_batch_idx + 1)
        
        # 每个进程上的(4) (5, 13)在第一维度上concat (4 * num_process), (5 * num_process, 13)
        # average_surface_MSE_losses, average_upper_MSE_height_losses = self.accelerator.gather_for_metrics((average_surface_MSE_loss, average_upper_MSE_height_loss))
        # self.accelerator.print(average_surface_MSE_losses, average_upper_MSE_height_losses)
        return average_surface_MSE_loss, average_upper_MSE_height_loss

    def train(self):
        count_parameters(self.model)
        print("*****************start training*****************")
        
        for ep in range(self.args.epochs):
            
            train_loss = self.train_epoch()
            test_loss = self.test_epoch()

            wandb.log({"lr": self.scheduler.get_lr()[0], 
                       "train_loss": train_loss,
                       "Z500": test_loss[1][0, 5], 
                       "T850": test_loss[1][2, 2],
                       "T2M": test_loss[0][3],
                       "U10": test_loss[0][1]})
            
            self.accelerator.print(f'epoch {ep}: Learning rate {self.scheduler.get_lr()[0]}')
            self.accelerator.print("train_loss = ", train_loss)
            self.accelerator.print("Z500 = ", test_loss[1][0, 5])
            self.accelerator.print("T850 = ", test_loss[1][2, 2])
            self.accelerator.print("T2M = ", test_loss[0][3])
            self.accelerator.print("U10 = ", test_loss[0][1])
            
            if (ep + 1) % self.args.save_ckpt_epoch == 0:
                torch.save(self.model.encoder.state_dict(), os.path.join(ckpt_foler, "encoder_2.ckpt"))
                torch.save(self.model.decoder.state_dict(), os.path.join(ckpt_foler, "decoder_2.ckpt"))
                
        self.accelerator.wait_for_everyone()
            
    def train_auto(self):
        count_parameters(self.model)
        print("*****************start training auto*****************")
        
        for ep in range(self.args.epochs):
            
            train_loss = self.train_auto_epoch()
            test_loss = self.test_auto_epoch()

            # 只看第一个进程上的loss
            wandb.log({"lr": self.scheduler.get_lr()[0], 
                       "train_loss": train_loss,
                       "Z500": test_loss[1][0, 5], 
                       "T850": test_loss[1][2, 2],
                       "T2M": test_loss[0][3],
                       "U10": test_loss[0][1]})
            
            self.accelerator.print(f'epoch {ep}: Learning rate {self.scheduler.get_lr()[0]}')
            self.accelerator.print("train_loss = ", train_loss)
            self.accelerator.print("Z500 = ", test_loss[1][0, 5])       
            self.accelerator.print("T850 = ", test_loss[1][2, 2])
            self.accelerator.print("T2M = ", test_loss[0][3])
            self.accelerator.print("U10 = ", test_loss[0][1])
            
            if (ep + 1) % self.args.save_ckpt_epoch == 0:
                torch.save(self.model.encoder.state_dict(), os.path.join(ckpt_foler, "encoder_1.ckpt"))
                torch.save(self.model.decoder.state_dict(), os.path.join(ckpt_foler, "decoder_1.ckpt"))
        
        self.accelerator.wait_for_everyone()
def main():
    args = parse_args()
    print(args)
    set_visible_devices(args.gpu)
    
    wandb.init(project="llama4pde_frozen", name = "train_auto_enc_dec", config = parse_args())

    random.seed(args.seed)                  # 设置Python标准库中random模块的随机数种子
    np.random.seed(args.seed)               # 设置NumPy库中random模块的随机数种子
    torch.manual_seed(args.seed)            # 设置PyTorch库中的随机数种子
    torch.cuda.manual_seed_all(args.seed)   # 设置CUDA随机数生成器的种子，以确保在每次运行时得到相同的随机数序列
    
    torch.set_float32_matmul_precision('medium')  

    trainer = Llama4PdeTrainer(args)
    
    if not args.train_auto:
        trainer.train()
    else:
        trainer.train_auto()
        
    wandb.finish()
    
if __name__ == "__main__":
    main()

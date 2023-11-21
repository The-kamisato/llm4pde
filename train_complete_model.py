import torch  
import einops
import numpy as np
import torch.nn as nn
import math
from torch.nn import functional as F
from torch.optim import Optimizer
import scipy.io as sio
from tqdm import tqdm
from transformers import LlamaConfig
from transformers import LlamaModel, LlamaForCausalLM, AutoModelForCausalLM
from lightning.fabric import Fabric
from functools import partial


from losses import LpLoss

import loralib as lora
from metric import cal_mae_mse
from encoder_decoder import PdeEncodingToToken, TokenDecodingToPde
from complete_model import complete_model

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

epochs = 20
sub = 1
S = 90
S_1 = 90
S_2 = 180
T_in = 12
T = 24
h_t = 1

ntrain = 3652
ntest = 730


batch_size = 1
patch_size = 30
in_chans = 4
out_chans = 4               # 盘古模型的数据时，应该有几个变量？
# n_token_per_time = (S // patch_size) ** 2
n_token_per_time = (S_1 // patch_size) * (S_2 // patch_size)
proj_method = 'vit'
accumulation_steps = 20

lora_r_llama = 8
lora_alpha_llama = 16
lora_r = None
lora_alpha = None

scheduler_step = 100
scheduler_gamma = 0.5

phi =  torch.linspace(90, 0, S_1 + 1)
cos_phi = torch.cos((math.pi*phi)/180)[:-1]
cos_phi = cos_phi*90/(torch.sum(cos_phi))
lat_weight = cos_phi.reshape(1,1,1,S_1,1).repeat(batch_size, out_chans, 1, 1, S_2)


import random

seed = 1235

## specify randomness
random.seed(seed)                  # 设置Python标准库中random模块的随机数种子
np.random.seed(seed)               # 设置NumPy库中random模块的随机数种子
torch.manual_seed(seed)            # 设置PyTorch库中的随机数种子
torch.cuda.manual_seed_all(seed)   # 设置CUDA随机数生成器的种子，以确保在每次运行时得到相同的随机数序列      

# DATA_PATH = '/home/lbk/ddno/NavierStokes_V1e-5_N1200_T20.mat'

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:                                             # 正处于预热阶段，lr正在上升
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))        # 计算预热之后的训练的进度
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
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

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

def load_static(u_dataset):         # u_dataset torch.Size([4382, 24, 4, 90, 180])
    weather_surface_mean = torch.mean(u_dataset, dim = (0, 1, 3, 4))
    weather_surface_std = torch.std(u_dataset, dim = (0, 1, 3, 4))       
    return weather_surface_mean, weather_surface_std 


# print("reading...")
# mat_contents = sio.loadmat(DATA_PATH)
# u_dataset = mat_contents['u']
# u_dataset = torch.from_numpy(u_dataset)
# u_dataset = u_dataset.permute(0, 3, 1, 2)
# print(u_dataset.shape)


# train_u = u_dataset[:ntrain,:,::sub,::sub]
# train_u_truth = u_dataset[:ntrain,:,::sub,::sub]
# test_u = u_dataset[-ntest:,:T_in,::sub,::sub]
# test_u_truth = u_dataset[-ntest:,T_in:T,::sub,::sub]

# 对于era5的数据以(b t c h w)的形式输入，不同的batch取不同的时间段
# file_path_1 = "./Pangu_data/2022_surface_data_8_8_364.pt"  # .pt 文件的路径
# file_path = "/data/lbk/tensored_years_surface_data/2011_2020_surface_data_8_8_364.pt"
# file_path = "/data/lbk/tensored_years_surface_data/2022_2_surface_data_8_8_364.pt"
file_path = "/data/lbk/pangu_data/cli_download_data/tensored_years_surface_data/2011_2022_surface_data_8_8_364.pt"

land_const_file_path = "/home/lbk/Pangu-Weather/constant_masks/land_mask.npy"
soil_type_file_path = "/home/lbk/Pangu-Weather/constant_masks/soil_type.npy"
topography_file_path = "/home/lbk/Pangu-Weather/constant_masks/topography.npy"

u_dataset = torch.load(file_path)     # (b t c h w)
print(u_dataset.shape)



train_u = u_dataset[:ntrain,:,:,:S_1,:S_2]
train_u_truth = u_dataset[:ntrain,:,:,:S_1,:S_2]
test_u = u_dataset[-ntest:,:T_in,:,:S_1,:S_2]
test_u_truth = u_dataset[-ntest:,T_in:T,:,:S_1,:S_2]

# train_u = ((train_u.transpose(2,4) - weather_surface_mean) / weather_surface_std).transpose(2,4)
# test_u = ((test_u.transpose(2,4) - weather_surface_mean) / weather_surface_std).transpose(2,4)

# print(train_u.shape)
# print(train_u_truth.shape)
# print(test_u.shape)
# print(test_u_truth.shape)

torch.set_float32_matmul_precision('medium')  
fabric = Fabric(accelerator = "cuda", precision="bf16-mixed", devices=7, strategy="DDP")
fabric.launch()

# lit_llama_config = LLaMAConfig.from_name("7B")
# lit_llama_config.block_size = 40
config = LlamaConfig()
config.n_token_per_time = n_token_per_time
lora_config = LoraConfig(
        r=lora_r_llama,
        lora_alpha=lora_alpha_llama,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj', 'up_proj'],
        # target_modules=['c_attn', 'c_proj'],    # , 'c_fc1', 'c_fc2', 'lm_head'
        fan_in_fan_out=False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
    )

myloss = LpLoss()
weather_surface_mean, weather_surface_std = load_static(u_dataset)
weather_surface_mean, weather_surface_std = weather_surface_mean.cuda(), weather_surface_std.cuda()
lat_weight = lat_weight.cuda()

print("model building...")



model = complete_model(llama_config = config, img_size = [S_1, S_2], embed_dim = config.hidden_size, n_head = config.num_attention_heads, 
                   patch_size = patch_size, proj_method = proj_method, in_chans = in_chans, out_chans = out_chans, num_token_per_time = n_token_per_time, 
                 if_gird = False, if_const_mask = True, lora_r = lora_r, lora_alpha = lora_alpha, llama_lora_config = lora_config, 
                 weather_surface_mean = weather_surface_mean, weather_surface_std = weather_surface_std, lat_weight = lat_weight, time_embed = False, pos_embed = False)

if lora_r is not None:
    lora.mark_only_lora_as_trainable(model.encoder)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_u, train_u_truth), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_u, test_u_truth), batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=(0.1 * epochs * ntrain) / (accumulation_steps * 6), 
#                                             num_training_steps = ((epochs * ntrain) / (accumulation_steps * 6)) + 10, num_cycles = 0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, div_factor=1e2, final_div_factor=1e2,
#                        steps_per_epoch=10, epochs=epochs)



# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
model, optimizer = fabric.setup(model, optimizer)
train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
# model, encoder, optimizer = fabric.setup(model, encoder, optimizer)
# train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=epochs,steps_per_epoch=len(train_loader))

count_parameters(model)


    
for ep in range(epochs):
    model.train()
    
    train_l2 = 0
    
    for batch_idx, batch in enumerate(train_loader):
        l2 = 0
        
        is_accumulating = (batch_idx + 1) % accumulation_steps != 0
        train_u, train_u_truth = batch
        
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            train_u = ((train_u.transpose(2,4) - weather_surface_mean) / weather_surface_std).transpose(2,4)
        
            T_end_batch = torch.randint(low = T_in , high = T, size = (batch_size, 1), device='cuda')   # 生成(batch_size, 1)size 的一个10-20随机整数的张量
            # u_input = torch.cat([train_u[b, T_end.item() - T_in : T_end.item(), :, :, :].unsqueeze(dim = 0) for b, T_end in enumerate(T_end_batch)], dim=0)
            u_input = torch.cat([train_u[b, : T_end.item(), :, :, :].unsqueeze(dim = 0) for b, T_end in enumerate(T_end_batch)], dim=0)
        
            # _hidden_state由(0:T_end)预测(1:T_end+1)
            # pred_u_ground_truth = torch.cat([train_u_truth[b, T_end.item() - T_in + 1: T_end.item() + 1, :, :, :].unsqueeze(dim = 0) for b, T_end in enumerate(T_end_batch)], dim=0)
            pred_u_ground_truth = torch.cat([train_u_truth[b, 1: T_end.item() + 1, :, :, :].unsqueeze(dim = 0) for b, T_end in enumerate(T_end_batch)], dim=0)
            pred_u, l2 = model(u_input, pred_u_ground_truth)
            
            l2 /= accumulation_steps  # 将损失除以累积步数
        
            fabric.backward(l2) 
        
        if not is_accumulating:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        train_l2 += l2.item() * accumulation_steps
    
    
    model.eval()
    
    test_l2 = 0.0
    with torch.no_grad():
        # for test_u, test_u_truth in test_loader:
        # print("testing...")
        # for batch in tqdm(test_loader, desc='Testing progress', leave=True):
        total_MAE_loss = 0 
        total_RMSE_loss = 0
        for batch in test_loader:
            test_u, test_u_truth = batch
            # test_u, test_u_truth = test_u.cuda(), test_u_truth.cuda()
            # pred_u = model(test_u)
            test_u = ((test_u.transpose(2,4) - weather_surface_mean) / weather_surface_std).transpose(2,4)
            for _ in range(T - T_in):
                # pred_u = model(test_u[:, -T_in:, :, :, :])
                pred_u = model(test_u)
                test_u = torch.cat((test_u, pred_u[:, -1:, :, :, :]), dim=1)
                
            pred_u = test_u[:, T_in: T, :, :, :]
            pred_u = (pred_u.transpose(2,4) * weather_surface_std + weather_surface_mean).transpose(2,4)
            
            l2 = myloss(pred_u.view(batch_size, -1), test_u_truth.view(batch_size, -1))
            test_l2 += l2.item()
            
            MAE_loss, RMSE_loss = cal_mae_mse(logits = pred_u[:, -1, :, :, :].reshape(batch_size, out_chans, 1, S_1, S_2), 
                                               target = test_u_truth[:, -1, :, :, :].reshape(batch_size, out_chans, 1, S_1, S_2), lat_weight = lat_weight)
            
            total_MAE_loss = total_MAE_loss + MAE_loss
            total_RMSE_loss = total_RMSE_loss + RMSE_loss
            
    train_l2 /= ntrain
    test_l2 /= ntest
    
    average_MAE_loss = total_MAE_loss / ntest
    average_RMSE_loss = total_RMSE_loss / ntest

    print(f'epoch {ep}: Learning rate {scheduler.get_lr()[0]}')
    print(train_l2, test_l2)
    print(average_MAE_loss)
    print(average_RMSE_loss)
    


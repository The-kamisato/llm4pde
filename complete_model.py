import torch  
import einops
import numpy as np
import torch.nn as nn
import math
from torch.nn import functional as F
import scipy.io as sio
from tqdm import tqdm
from transformers import LlamaConfig
from transformers import LlamaModel, LlamaForCausalLM, AutoModelForCausalLM
from lightning.fabric import Fabric
from functools import partial

from losses import LpLoss

from metric import cal_mae_mse
from encoder_decoder import PdeEncodingToToken, TokenDecodingToPde

import loralib as lora
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

class complete_model(nn.Module):
    def __init__(self, llama_config, img_size, embed_dim, n_head, patch_size = 4, proj_method = 'lsm', in_chans = 4, out_chans = 4, num_token_per_time = 18, 
                 if_gird = True, if_const_mask = True, lora_r = None, lora_alpha=None, llama_lora_config = None, 
                 weather_surface_mean = None, weather_surface_std = None, lat_weight = None, time_embed = True, pos_embed = True):
        super(complete_model, self).__init__()
        self.out_chans = out_chans
        self.S_1 = img_size[0]
        self.S_2 = img_size[1]
        self.lat_weight = lat_weight
        self.weather_surface_mean = weather_surface_mean
        self.weather_surface_std = weather_surface_std
        
        self.encoder = PdeEncodingToToken(img_size = img_size, embed_dim = embed_dim,
                                          n_head = n_head, patch_size = patch_size, proj_method = proj_method, 
                                          in_chans=in_chans,
                                          num_token_per_time = num_token_per_time, if_gird = if_gird, if_const_mask = if_const_mask,
                                          lora_r = lora_r, lora_alpha=lora_alpha).cuda()
        
        
        llama_config.time_embed = time_embed
        llama_config.pos_embed = pos_embed
        self.llama = get_peft_model(AutoModelForCausalLM.from_pretrained("/home/lbk/llama2-lora-fine-tuning/models/daryl149/llama-2-7b-chat-hf", 
                                                                         config = llama_config, load_in_8bit = True), llama_lora_config)
        if time_embed:
            self.llama.model.model.time_embedding = nn.Parameter(torch.empty(size=(50, llama_config.hidden_size), dtype=torch.float32, device=self.llama.device, requires_grad=True))
            nn.init.kaiming_normal_(self.llama.model.model.time_embedding)
        else:
            self.llama.model.model.time_embedding = None
        if pos_embed:
            for i in range(llama_config.num_hidden_layers):
                self.llama.model.model.layers[i].self_attn.earth_specific_bias = nn.Parameter(torch.empty(size=(llama_config.n_token_per_time, llama_config.n_token_per_time, llama_config.num_attention_heads), 
                                                                                                      dtype=torch.float32, device=self.llama.device, requires_grad=True))
                nn.init.kaiming_normal_(self.llama.model.model.layers[i].self_attn.earth_specific_bias)
        
        self.head_layer = nn.Linear(in_features = embed_dim, out_features = patch_size**2*out_chans, bias=False).cuda()
        
        self.decoder = TokenDecodingToPde(img_size = img_size, embed_dim = embed_dim, out_chans = out_chans,
                             n_head = n_head, patch_size = patch_size, proj_method = proj_method, 
                             num_token_per_time = num_token_per_time).cuda()
        
    def forward(self, u_input, pred_u_ground_truth = None):
        inputs_embeds = self.encoder(u_input)       # inputs_embeds: (b, t*num_patch, n_embd)
        u_hidden_state = self.llama(input_ids = None, inputs_embeds = inputs_embeds, output_hidden_states = True).hidden_states[-1]
        u_hidden_state = self.head_layer(u_hidden_state)
        # u_hidden_state = self.head_layer(inputs_embeds)
        pred_u = self.decoder(u_hidden_state, u_input)
        
        if pred_u_ground_truth is None:
            return pred_u
        else:
            pred_u = (pred_u.transpose(2,4) * self.weather_surface_std + self.weather_surface_mean).transpose(2,4)
            l2_mae, l2_rmse = cal_mae_mse(logits = pred_u[:, -1, :, :, :].reshape(-1, self.out_chans, 1, self.S_1, self.S_2),
                            target = pred_u_ground_truth[:, -1, :, :, :].reshape(-1, self.out_chans, 1, self.S_1, self.S_2), lat_weight = self.lat_weight)
            l2 = torch.mean(l2_rmse)
            return pred_u, l2
     

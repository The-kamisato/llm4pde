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

from metric import cal_mae_mse
from enc_dec.encoder_decoder_low_res import Era5DataEncodingToTokenSmall, TokenDecodingToEra5DataSmall

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
    def __init__(self, surface_in_chans = 4, upper_in_chans = 5, dim = 64, final_dim = 4096, if_surface_const_mask = True, 
                 act_type='newgelu', norm_type='ln', frozen_enc_dec = True, load_pretrained_enc_dec = True, 
                 llama_body = False, llama_config = None, llama_lora_config = None, frozen_llama = True, time_embed = False, pos_embed = False):
        super(complete_model, self).__init__()
        self.encoder = Era5DataEncodingToTokenSmall(surface_in_chans = surface_in_chans, upper_in_chans = upper_in_chans, act_type = act_type, norm_type = norm_type,
                                                    dim = dim, final_dim = final_dim, if_surface_const_mask = if_surface_const_mask)
        self.decoder = TokenDecodingToEra5DataSmall(surface_in_chans = surface_in_chans, upper_in_chans = upper_in_chans, act_type = act_type, norm_type = norm_type,
                                                    dim = dim, final_dim = final_dim)
        
        if load_pretrained_enc_dec:
            self.encoder.load_state_dict(torch.load('enc_dec_ckpt/encoder.pt'))
            self.decoder.load_state_dict(torch.load('enc_dec_ckpt/decoder.pt'))
            
        if frozen_enc_dec:
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()
            
        self.llama_body = llama_body
        if llama_body:
            self.llama = AutoModelForCausalLM.from_pretrained("/home/lbk/llama2-lora-fine-tuning/models/daryl149/llama-2-7b-chat-hf", config = llama_config, load_in_8bit = True)
            if frozen_llama:
                llama_config.time_embed = False
                llama_config.pos_embed = False
                for param in self.llama.parameters():
                    param.requires_grad = False
                    self.llama = self.llama.eval()
            else:
                llama_config.time_embed = time_embed
                llama_config.pos_embed = pos_embed
                self.llama = get_peft_model(self.llama, llama_lora_config)
                
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
        else:
            self.llama = None
            
            
    def forward(self, surface_u_input, upper_u_input, p = 1, h = 12, w = 6):
        token_embed = self.encoder(surface_u_input, upper_u_input)          # (1, 504, 4096)
        if self.llama_body:
            token_embed = self.llama(input_ids = None, inputs_embeds = token_embed, output_hidden_states = True).hidden_states[-1]
            
        surface_u_output, upper_u_output = self.decoder(token_embed, p = p, h = h, w = w)
        
        return surface_u_output, upper_u_output
            

import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np
import einops

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    
def load_constant_mask(downsample = 8):
    land_mask = torch.tensor(np.load("/home/lbk/Pangu-Weather/constant_masks/land_mask.npy"))
    soil_type = torch.tensor(np.load("/home/lbk/Pangu-Weather/constant_masks/soil_type.npy"))
    topography = torch.tensor(np.load("/home/lbk/Pangu-Weather/constant_masks/topography.npy"))

    # 下采样之后shape都是torch.Size([1440//8, 720//8]) = torch.Size([180, 91])
    land_mask = land_mask[::downsample, ::downsample].float()
    soil_type = soil_type[::downsample, ::downsample].float()
    topography = topography[::downsample, ::downsample].float()
    
    land_mask = (land_mask - land_mask.mean()) / land_mask.std()
    soil_type = (soil_type - soil_type.mean()) / soil_type.std()
    topography = (topography - topography.mean()) / topography.std()
    
    return land_mask.cuda(), soil_type.cuda(), topography.cuda() 


class SurfaceDataEncodingToToken(nn.Module):
    def __init__(self, in_chans = 4, if_const_mask = True):
        super(SurfaceDataEncodingToToken, self).__init__()
        
        self.conv = nn.Conv2d(in_channels = in_chans + 3 if if_const_mask else in_chans, out_channels = 256, kernel_size = 2, stride = 2, padding=(0, 1))
        
        self.ln = nn.LayerNorm(256)
        self.fc1 = nn.Linear(in_features = 256, out_features = 1024)
        self.act = NewGELU()
        self.fc2 = nn.Linear(in_features = 1024, out_features = 256)
        
        self.land_mask, self.soil_type, self.topography = load_constant_mask()
        self.if_const_mask = if_const_mask
    
    def concat_mask(self, x):
        B, T, H, W = x.size(0), x.size(1), x.size(-2), x.size(-1)
        broad_land_mask = self.land_mask.reshape(1, 1, 1, H, W).repeat(B, T, 1, 1, 1)
        broad_soil_type = self.soil_type.reshape(1, 1, 1, H, W).repeat(B, T, 1, 1, 1)
        broad_topography = self.topography.reshape(1, 1, 1, H, W).repeat(B, T, 1, 1, 1)
        x = torch.cat((x, broad_land_mask, broad_soil_type, broad_topography), dim = 2)
        return x

    def forward(self, x):
        if self.if_const_mask:
            x = self.concat_mask(x)         # (B = 1, T = 12, C + 3 = 7, H = 180, W = 90)
            
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        
        x = self.ln(self.conv(x).transpose(-1, 1))                          # self.conv(x): (b * t, 256, H // 2 = 90, W // 2 = 46)
        x = self.fc2(self.act(self.fc1(x))).transpose(-1, 1)                # (b * t, 256, H // 2 = 90, W // 2 = 46)
        
        # x = einops.rearrange(x, '(b t) C h w -> b (h w t) C ', b = B, t = T, h = H // 32, w = W // 32)       # (b, t*num_patch, n_embd = 2048)
        return x
        
        
class UpperDataEncodingToToken(nn.Module):
    def __init__(self, in_chans = 5, if_const_mask = True):
        super(UpperDataEncodingToToken, self).__init__()
        
        self.conv = nn.Conv3d(in_channels = in_chans + 3 if if_const_mask else in_chans, out_channels = 256, kernel_size = 2, stride = (1, 2, 2), padding=(0, 0, 1))
        
        self.ln = nn.LayerNorm(256)
        self.fc1 = nn.Linear(in_features = 256, out_features = 1024)
        self.act = NewGELU()
        self.fc2 = nn.Linear(in_features = 1024, out_features = 256)
        
        self.land_mask, self.soil_type, self.topography = load_constant_mask()
        self.if_const_mask = if_const_mask
    
    def concat_mask(self, x):
        B, T, P, H, W = x.size(0), x.size(1), x.size(-3), x.size(-2), x.size(-1)
        broad_land_mask = self.land_mask.reshape(1, 1, 1, 1, H, W).repeat(B, T, 1, P, 1, 1)
        broad_soil_type = self.soil_type.reshape(1, 1, 1, 1, H, W).repeat(B, T, 1, P, 1, 1)
        broad_topography = self.topography.reshape(1, 1, 1, 1, H, W).repeat(B, T, 1, P, 1, 1)
        x = torch.cat((x, broad_land_mask, broad_soil_type, broad_topography), dim = 2)
        return x                    

    def forward(self, x):
        if self.if_const_mask:
            x = self.concat_mask(x)         # (B = 1, T = 12, C + 3 = 8, P = 7, H = 180, W = 90)
            
        x = einops.rearrange(x, 'b t c p h w -> (b t) c p h w')
        
        x = self.ln(self.conv(x).transpose(-1, 1))     # self.conv(x) : (b * t, 256, P = 6, H // 2 = 90, W // 2 = 45)
        x = self.fc2(self.act(self.fc1(x))).transpose(-1, 1)            # (b * t, 256, P = 6, H // 2 = 90, W // 2 = 45)   
        # x = einops.rearrange(x, '(b t) C p h w -> b (h w t) (p C) ', b = B, t = T, p = 2, h = H // 32, w = W // 32)       # (b, t*num_patch, n_embd = 1024*2)

        return x
        
class Era5DataEncodingToToken(nn.Module):
    def __init__(self, surface_in_chans = 4, upper_in_chans = 5, if_surface_const_mask = True, if_upper_const_mask = True):
        super(Era5DataEncodingToToken, self).__init__()
        self.surface_encode = SurfaceDataEncodingToToken(in_chans = surface_in_chans, if_const_mask = if_surface_const_mask)
        self.upper_encoder = UpperDataEncodingToToken(in_chans = upper_in_chans, if_const_mask = if_upper_const_mask)
        
        self.conv1 = nn.Conv3d(in_channels = 256, out_channels = 512, kernel_size = 2, stride = (2, 2, 2), padding = (1, 0, 0))         
        self.conv2 = nn.Conv3d(in_channels = 512, out_channels = 1024, kernel_size = 2, stride = (1, 2, 2), padding = (0, 1, 1))
        self.conv3 = nn.Conv3d(in_channels = 1024, out_channels = 2048, kernel_size = 2, stride = (1, 2, 2), padding = (0, 1, 0))
        self.conv4 = nn.Conv3d(in_channels = 2048, out_channels = 4096, kernel_size = 2, stride = (1, 2, 2), padding = (0, 0, 0))
        
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(1024)
        self.ln3 = nn.LayerNorm(2048)
        self.ln4 = nn.LayerNorm(4096)
        
        self.act = NewGELU()
        
    def forward(self, surface_u, upper_u):
        T_1 = surface_u.size(1)
        T_2 = upper_u.size(1)
        assert(T_1 == T_2)
        
        surface_data_conv = self.surface_encode(surface_u)         # (b*t, 256, 90, 46)
        upper_data_conv = self.upper_encoder(upper_u)              # (b*t, 256, 6, 90, 46)
        data_conv = torch.cat((surface_data_conv[:, :, None, :, :], upper_data_conv), dim = 2)       # (b*t, 256, 7, 90, 46)
        
        data_conv = self.act(self.ln1(self.conv1(data_conv).transpose(1,-1)).transpose(1,-1))                       # (b*t, 512, 4, 45, 23)
        data_conv = self.act(self.ln2(self.conv2(data_conv).transpose(1,-1)).transpose(1,-1))                       # (b*t, 1024, 3, 23, 12)
        data_conv = self.act(self.ln3(self.conv3(data_conv).transpose(1,-1)).transpose(1,-1))                       # (b*t, 2048, 2, 12, 6)
        data_conv = self.act(self.ln4(self.conv4(data_conv).transpose(1,-1)).transpose(1,-1))                       # (b*t, 4096, 1, 6, 3)
        
        token_embed = einops.rearrange(data_conv, '(b t) C p h w -> b (h w t) (p C) ', b = 1, t = T_1, p = 1, h = 6, w = 3)         # (b*t, 4096, 1, 6, 3) -> (b, t*6*3, 4096)
        return token_embed 

if __name__ == "__main__":
    surface_u = torch.randn(1, 12, 4, 180, 91).cuda()
    upper_u = torch.randn(1, 12, 5, 7, 180, 91).cuda()
    
    encoder = Era5DataEncodingToToken( if_surface_const_mask = True, if_upper_const_mask = False).cuda()
    token_embed = encoder(surface_u, upper_u)
    
    print(token_embed.shape)            # (bsz = 1, (T = 12) * (n_token_per_time = 6 * 3), 4096)

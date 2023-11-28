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

    # 下采样之后shape都是torch.Size([720//8, 1440//8]) = torch.Size([90, 180])
    land_mask = land_mask[:-1, :][::downsample, ::downsample]
    soil_type = soil_type[:-1, :][::downsample, ::downsample]
    topography = topography[:-1, :][::downsample, ::downsample]
    
    # 由于topography是海拔，数值比较大，因此在这里做一个归一化：
    topography = F.normalize(topography, dim = None)
    
    return land_mask.cuda(), soil_type.cuda(), topography.cuda() 


class SurfaceDataEncodingToToken(nn.Module):
    def __init__(self, in_chans = 4, if_const_mask = True):
        super(SurfaceDataEncodingToToken, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = in_chans + 3 if if_const_mask else in_chans, out_channels = 128, kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 2, stride = 2)
        self.conv4 = nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 2, stride = 2)
        self.conv5 = nn.Conv2d(in_channels = 1024, out_channels = 2048, kernel_size = 2, stride = 2)
        
        self.ln1 = nn.LayerNorm(128)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(512)
        self.ln4 = nn.LayerNorm(1024)
        self.ln5 = nn.LayerNorm(2048)

        self.act = NewGELU()
        
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
        B, T, H, W = x.size(0), x.size(1), x.size(-2), x.size(-1)
        
        if self.if_const_mask:
            x = self.concat_mask(x)         # (B = 1, T = 12, C + 3 = 7, H = 180, W = 90)
            
        x = einops.rearrange(x, 'b t c h w -> (b t) c h w')
        
        x = self.act(self.ln1(self.conv1(x).transpose(-1, 1)).transpose(-1, 1))
        x = self.act(self.ln2(self.conv2(x).transpose(-1, 1)).transpose(-1, 1))
        x = self.act(self.ln3(self.conv3(x).transpose(-1, 1)).transpose(-1, 1))
        x = self.act(self.ln4(self.conv4(x).transpose(-1, 1)).transpose(-1, 1))
        x = self.act(self.ln5(self.conv5(x).transpose(-1, 1)).transpose(-1, 1))     # (b * t, 2048, H // 32, W // 32)
        
        x = einops.rearrange(x, '(b t) C h w -> b (h w t) C ', b = B, t = T, h = H // 32, w = W // 32)       # (b, t*num_patch, n_embd = 2048)
        
        
class UpperDataEncodingToToken(nn.Module):
    def __init__(self, in_chans = 5, if_const_mask = True):
        super(UpperDataEncodingToToken, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels = in_chans + 3 if if_const_mask else in_chans, out_channels = 64, kernel_size = 2, stride = (1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels = 64, out_channels = 128, kernel_size = 2, stride = (1, 2, 2))
        self.conv3 = nn.Conv3d(in_channels = 128, out_channels = 256, kernel_size = 2, stride = (1, 2, 2))
        self.conv4 = nn.Conv3d(in_channels = 256, out_channels = 512, kernel_size = 2, stride = (1, 2, 2))
        self.conv5 = nn.Conv3d(in_channels = 512, out_channels = 1024, kernel_size = 2, stride = (1, 2, 2))
        
        self.ln1 = nn.LayerNorm(64)
        self.ln2 = nn.LayerNorm(128)
        self.ln3 = nn.LayerNorm(256)
        self.ln4 = nn.LayerNorm(512)
        self.ln5 = nn.LayerNorm(1024)

        self.act = NewGELU()
        
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
        B, T, P, H, W = x.size(0), x.size(1), x.size(-3), x.size(-2), x.size(-1)
        
        if self.if_const_mask:
            x = self.concat_mask(x)         # (B = 1, T = 12, C + 3 = 8, P = 7, H = 180, W = 90)
            
        x = einops.rearrange(x, 'b t c p h w -> (b t) c p h w')
        
        x = self.act(self.ln1(self.conv1(x).transpose(-1, 1)).transpose(-1, 1))
        x = self.act(self.ln2(self.conv2(x).transpose(-1, 1)).transpose(-1, 1))
        x = self.act(self.ln3(self.conv3(x).transpose(-1, 1)).transpose(-1, 1))
        x = self.act(self.ln4(self.conv4(x).transpose(-1, 1)).transpose(-1, 1))
        x = self.act(self.ln5(self.conv5(x).transpose(-1, 1)).transpose(-1, 1))      # (b * t, 1024, P = 2, H // 32, W // 32)
        
        x = einops.rearrange(x, '(b t) C p h w -> b (h w t) (p C) ', b = B, t = T, p = 2, h = H // 32, w = W // 32)       # (b, t*num_patch, n_embd = 1024*2)
        
class Era5DataEncodingToToken(nn.Module):
    def __init__(self, surface_in_chans = 4, upper_in_chans = 5, if_surface_const_mask = True, if_upper_const_mask = True):
        super(Era5DataEncodingToToken, self).__init__()
        self.surface_encode = SurfaceDataEncodingToToken(in_chans = surface_in_chans, if_const_mask = if_surface_const_mask)
        self.upper_encoder = UpperDataEncodingToToken(in_chans = upper_in_chans, if_const_mask = if_upper_const_mask)
        self.fc = nn.Linear(in_features = 4096, out_features = 4096)
        
    def forward(self, surface_u, upper_u):
        surface_data_embed = self.surface_encode(surface_u)         # (b, t*num_patch, n_embd = 2048)
        upper_data_embed = self.upper_encoder(upper_u)               # (b, t*num_patch, n_embd = 1024*2)
        token_embed = self.fc(torch.cat((surface_data_embed, upper_data_embed), dim = -1))      # (b, t*num_patch, n_embd = 4096)
        return token_embed 

if __name__ == "__main__":
    surface_u = torch.randn(1, 12, 4, 90, 180).cuda()
    upper_u = torch.randn(1, 12, 5, 7, 90, 180).cuda()
    
    encoder = Era5DataEncodingToToken().cuda()
    token_embed = encoder(surface_u, upper_u)
    
    print(token_embed.shape)

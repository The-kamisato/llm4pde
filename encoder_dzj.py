import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np
import einops

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        print(name, parameter.numel())
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def swish(x):
    # swish
    return x*torch.sigmoid(x)

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
    return land_mask, soil_type, topography

class Era5DataEncodingToToken(nn.Module):
    def __init__(self, surface_in_chans = 4, upper_in_chans = 5, dim=64, final_dim=4096, if_surface_const_mask = True, act_type='newgelu', norm_type='ln'):
        super(Era5DataEncodingToToken, self).__init__()
        self.if_surface_const_mask = if_surface_const_mask
        
        land_mask, soil_type, topography = load_constant_mask()
        self.register_buffer("land_mask", land_mask)
        self.register_buffer("soil_type", soil_type)
        self.register_buffer("topography", topography)

        self.conv_surface = nn.Conv2d(in_channels = surface_in_chans + 3 if if_surface_const_mask else surface_in_chans, out_channels = dim, kernel_size = (3, 3), stride = (2, 2), padding=(1, 1))
        self.conv_upper = nn.Conv3d(in_channels = upper_in_chans, out_channels = dim, kernel_size = (3, 3, 3), stride = (2, 2, 2), padding=(1, 1, 1))

        self.conv1 = nn.Conv3d(in_channels = dim, out_channels = dim * 2, kernel_size = (2, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))         
        self.conv2 = nn.Conv3d(in_channels = dim * 2, out_channels = dim * 4, kernel_size = (2, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        self.conv3 = nn.Conv3d(in_channels = dim * 4, out_channels = dim * 8, kernel_size = (2, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        self.conv4 = nn.Conv3d(in_channels = dim * 8, out_channels = dim * 16, kernel_size = (2, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        self.conv5 = nn.Conv3d(in_channels = dim * 16, out_channels = final_dim, kernel_size = (1, 1, 1), stride = (1, 1, 1), padding = (0, 0, 0))
        # self.conv5 = nn.Conv3d(in_channels = dim * 16, out_channels = final_dim, kernel_size = (1, 1, 1), stride = (1, 2, 2), padding = (0, 1, 1))

        if act_type == 'newgelu':
            self.act = NewGELU()
        elif act_type == 'swish':
            self.act = swish
        else:
            self.act = nn.ReLU()

        if norm_type == 'ln':
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim * 2)
            self.ln3 = nn.LayerNorm(dim * 4)
            self.ln4 = nn.LayerNorm(dim * 8)
            self.ln5 = nn.LayerNorm(dim * 16)
        elif norm_type == 'gn':
            self.ln1 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
            self.ln2 = nn.GroupNorm(num_groups=32, num_channels=dim * 2, eps=1e-6, affine=True)
            self.ln3 = nn.GroupNorm(num_groups=32, num_channels=dim * 4, eps=1e-6, affine=True)
            self.ln4 = nn.GroupNorm(num_groups=32, num_channels=dim * 8, eps=1e-6, affine=True)
            self.ln5 = nn.GroupNorm(num_groups=32, num_channels=dim * 16, eps=1e-6, affine=True)
        else:
            raise NotImplementedError

    def concat_mask(self, x):
        B, T, H, W = x.size(0), x.size(1), x.size(-2), x.size(-1)
        broad_land_mask = self.land_mask.reshape(1, 1, 1, H, W).repeat(B, T, 1, 1, 1)
        broad_soil_type = self.soil_type.reshape(1, 1, 1, H, W).repeat(B, T, 1, 1, 1)
        broad_topography = self.topography.reshape(1, 1, 1, H, W).repeat(B, T, 1, 1, 1)
        x = torch.cat((x, broad_land_mask, broad_soil_type, broad_topography), dim = 2)
        return x         

    def forward(self, surface_u, upper_u):
        B, T_1 = surface_u.size(0), surface_u.size(1)
        T_2 = upper_u.size(1)
        assert(T_1 == T_2)

        if self.if_surface_const_mask:
            surface_u = self.concat_mask(surface_u)

        surface_embedding = self.conv_surface(surface_u.flatten(0, 1)) # (b*t, dim, 90, 46)
        print(surface_embedding.shape)

        upper_embedding = self.conv_upper(upper_u.flatten(0, 1)) # (b*t, dim, 7, 90, 46)
        print(upper_embedding.shape)

        x = torch.cat((surface_embedding[:, :, None, :, :], upper_embedding), dim = 2) #  (b*t, dim, 5, 90, 46)

        x = self.conv1(self.act(self.ln1(x.transpose(1,-1)).transpose(1,-1))) # (b*t, dim * 2, 4, 45, 23)        
        x = self.conv2(self.act(self.ln2(x.transpose(1,-1)).transpose(1,-1))) # (b*t, dim * 4, 3, 23, 12)
        x = self.conv3(self.act(self.ln3(x.transpose(1,-1)).transpose(1,-1))) # (b*t, dim * 8, 2, 12, 6)
        x = self.conv4(self.act(self.ln4(x.transpose(1,-1)).transpose(1,-1))) # (b*t, dim * 16, 1, 6, 3)
        x = self.conv5(self.act(self.ln5(x.transpose(1,-1)).transpose(1,-1))) # (b*t, final_dim, 1, 6, 3)
        
        x = einops.rearrange(x, '(b t) C p h w -> b (t p h w) C', b = B, t = T_1, p = x.size(2), h = x.size(3), w = x.size(4)) # (b*t, final_dim, 1, 6, 3) -> (b, t*1*6*3, final_dim)

        return x 

if __name__ == "__main__":
    surface_u = torch.randn(1, 12, 4, 180, 91).cuda()
    upper_u = torch.randn(1, 12, 5, 7, 180, 91).cuda()

    encoder = Era5DataEncodingToToken().cuda()
    count_parameters(encoder)
    token_embed = encoder(surface_u, upper_u)

    print(token_embed.shape)            # (bsz = 1, (T = 12) * (n_token_per_time = 6 * 3), 4096)

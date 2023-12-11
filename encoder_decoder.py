import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import numpy as np
import einops
from torch.profiler import profile, record_function, ProfilerActivity

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

class TransposeLN(nn.Module):
    def __init__(self, num_features, dim1=1, dim2=-1):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return self.norm(x.contiguous().transpose(self.dim1, self.dim2).contiguous()).transpose(self.dim1, self.dim2).contiguous()

def load_constant_mask(downsample = 1):
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
    return land_mask.contiguous().transpose(0,1).contiguous(), soil_type.contiguous().transpose(0,1).contiguous(), topography.contiguous().transpose(0,1).contiguous()

class Era5DataEncodingToTokenSmall(nn.Module):
    def __init__(self, surface_in_chans = 4, upper_in_chans = 5, dim=64, final_dim=4096, if_surface_const_mask = True, act_type='newgelu', norm_type='ln'):
        super(Era5DataEncodingToTokenSmall, self).__init__()
        self.if_surface_const_mask = if_surface_const_mask

        land_mask, soil_type, topography = load_constant_mask()     # torch.Size([721, 1440])
        self.register_buffer("land_mask", land_mask)
        self.register_buffer("soil_type", soil_type)
        self.register_buffer("topography", topography)

        self.conv_surface = nn.Conv2d(in_channels = surface_in_chans + 3 if if_surface_const_mask else surface_in_chans, out_channels = dim, kernel_size = (7, 7), stride = (4, 4), padding=(3, 3))
        self.conv_upper = nn.Conv3d(in_channels = upper_in_chans, out_channels = dim, kernel_size = (3, 7, 7), stride = (2, 4, 4), padding=(1, 3, 3))
        
        self.conv1 = nn.Conv3d(in_channels = dim, out_channels = dim * 2, kernel_size = (3, 7, 7), stride = (2, 2, 2), padding = (1, 3, 3))         
        self.conv2 = nn.Conv3d(in_channels = dim * 2, out_channels = dim * 4, kernel_size = (2, 7, 7), stride = (1, 2, 2), padding = (0, 3, 3))
        self.conv3 = nn.Conv3d(in_channels = dim * 4, out_channels = dim * 8, kernel_size = (2, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        self.conv4 = nn.Conv3d(in_channels = dim * 8, out_channels = dim * 16, kernel_size = (2, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        # self.conv5 = nn.Conv3d(in_channels = dim * 16, out_channels = final_dim, kernel_size = (1, 1, 1), stride = (1, 1, 1), padding = (0, 0, 0))
        self.fc = nn.Linear(in_features = dim * 16, out_features = final_dim)
        if act_type == 'newgelu':
            self.act = NewGELU()
        elif act_type == 'swish':
            self.act = swish
        else:
            self.act = nn.ReLU()

        if norm_type == 'ln':
            self.norm1 = TransposeLN(dim)
            self.norm2 = TransposeLN(dim * 2)
            self.norm3 = TransposeLN(dim * 4)
            self.norm4 = TransposeLN(dim * 8)
            self.norm5 = nn.LayerNorm(dim * 16)
        elif norm_type == 'gn':
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim * 2, eps=1e-6, affine=True)
            self.norm3 = nn.GroupNorm(num_groups=32, num_channels=dim * 4, eps=1e-6, affine=True)
            self.norm4 = nn.GroupNorm(num_groups=32, num_channels=dim * 8, eps=1e-6, affine=True)
            self.norm5 = nn.GroupNorm(num_groups=32, num_channels=dim * 16, eps=1e-6, affine=True)
        else:
            raise NotImplementedError

    def concat_mask(self, x, downsample = 2):
        B, T, H, W = x.size(0), x.size(1), x.size(-2), x.size(-1)
        broad_land_mask = self.land_mask[None, None, None, ::downsample, ::downsample].repeat(B, T, 1, 1, 1)
        broad_soil_type = self.soil_type[None, None, None, ::downsample, ::downsample].repeat(B, T, 1, 1, 1)
        broad_topography = self.topography[None, None, None, ::downsample, ::downsample].repeat(B, T, 1, 1, 1)
        x = torch.cat((x, broad_land_mask, broad_soil_type, broad_topography), dim = 2)
        return x         

    def forward(self, surface_u, upper_u):
        B, T_1 = surface_u.size(0), surface_u.size(1)
        T_2 = upper_u.size(1)
        assert(T_1 == T_2)

        if self.if_surface_const_mask:
            surface_u = self.concat_mask(surface_u)

        surface_embedding = self.conv_surface(surface_u.flatten(0, 1)) # (b*t, dim, 180, 91)
  
        upper_embedding = self.conv_upper(upper_u.flatten(0, 1)) # (b*t, dim, 7, 180, 91)
       

        x = torch.cat((surface_embedding[:, :, None, :, :], upper_embedding), dim = 2) #  (b*t, dim, 8, 180, 91)

        x = self.conv1(self.act(self.norm1(x))) # (b*t, dim * 2, 4, 90, 46)  
        x = self.conv2(self.act(self.norm2(x))) # (b*t, dim * 4, 3, 45, 23)
        x = self.conv3(self.act(self.norm3(x))) # (b*t, dim * 8, 2, 23, 12)
        x = self.conv4(self.act(self.norm4(x))) # (b*t, dim * 16, 1, 12, 6)
        x = self.fc(self.act(self.norm5(x.transpose(1, -1)))).transpose(1, -1) # (b*t, final_dim, 1, 12, 6)       

        x = einops.rearrange(x, '(b t) C p h w -> b (t p h w) C', b = B, t = T_1, p = x.size(2), h = x.size(3), w = x.size(4)) # (b*t, final_dim, 1, 6, 3) -> (b, t*1*6*3, final_dim)

        return x
    
class TokenDecodingToEra5DataSmall(nn.Module):
    def __init__(self, surface_in_chans = 4, upper_in_chans = 5, dim=64, final_dim=4096, act_type='newgelu', norm_type='ln'):
        super(TokenDecodingToEra5DataSmall, self).__init__()

        # self.deconv1 = nn.ConvTranspose3d(in_channels = final_dim, out_channels = dim * 16, kernel_size = (1, 1, 1), stride = (1, 1, 1), padding = (0, 0, 0))
        self.fc = nn.Linear(in_features = final_dim, out_features = dim * 16)
        self.deconv2 = nn.ConvTranspose3d(in_channels = dim * 16, out_channels = dim * 8, kernel_size = (2, 2, 2), stride = (1, 2, 2), padding = (0, 0, 0))     
        self.deconv3 = nn.ConvTranspose3d(in_channels = dim * 8, out_channels = dim * 4, kernel_size = (2, 4, 4), stride = (1, 2, 2), padding = (0, 1, 1))     
        self.deconv4 = nn.ConvTranspose3d(in_channels = dim * 4, out_channels = dim * 2, kernel_size = (2, 8, 8), stride = (1, 2, 2), padding = (0, 3, 3))     
        self.deconv5 = nn.ConvTranspose3d(in_channels = dim * 2, out_channels = dim, kernel_size = (2, 8, 8), stride = (2, 2, 2), padding = (0, 3, 3))

        self.deconv_surface = nn.ConvTranspose2d(in_channels = dim, out_channels = surface_in_chans, kernel_size = (8, 9), stride = (4, 4), padding=(2, 2))
        self.deconv_upper = nn.ConvTranspose3d(in_channels = dim, out_channels = upper_in_chans, kernel_size = (3, 8, 9), stride = (2, 4, 4), padding=(1, 2, 2))

        if act_type == 'newgelu':
            self.act = NewGELU()
        elif act_type == 'swish':
            self.act = swish
        else:
            self.act = nn.ReLU()

        if norm_type == 'ln':
            self.norm1 = nn.LayerNorm(dim * 16)
            self.norm2 = TransposeLN(dim * 8)
            self.norm3 = TransposeLN(dim * 4)
            self.norm4 = TransposeLN(dim * 2)
            self.norm5 = TransposeLN(dim)
        elif norm_type == 'gn':
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=dim * 16, eps=1e-6, affine=True)
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=dim * 8, eps=1e-6, affine=True)
            self.norm3 = nn.GroupNorm(num_groups=32, num_channels=dim * 4, eps=1e-6, affine=True)
            self.norm4 = nn.GroupNorm(num_groups=32, num_channels=dim * 2, eps=1e-6, affine=True)
            self.norm5 = nn.GroupNorm(num_groups=32, num_channels=dim, eps=1e-6, affine=True)
        else:
            raise NotImplementedError

    def forward(self, x, p=1, h=12, w=6):
        B = x.shape[0]
        x = einops.rearrange(x, 'b (t p h w) C -> (b t) p h w C', b = B, p = p, h = h, w = w) # (b, t*1*12*6, final_dim) -> (b*t, 1, 12, 6, final_dim)
        
        x = self.act(self.norm1(self.fc(x)).permute(0, 4, 1, 2, 3))                   # (b*t, dim * 16, 2, 12, 6)
        x = self.act(self.norm2(self.deconv2(x)))                   # (b*t, dim * 8, 2, 24, 12)
        x = self.act(self.norm3(self.deconv3(x[:, :, :, :-1, :]))) # (b*t, dim * 4, 3, 46, 24) 
        x = self.act(self.norm4(self.deconv4(x[:, :, :, :-1, :-1]))) # (b*t, dim * 2, 4, 90, 46) 
        x = self.act(self.norm5(self.deconv5(x[:, :, :, :, :-1]))) # (b*t, dim, 8, 180, 90)   

        surface = self.deconv_surface(x[:, :, 0]) # (b*t, surface_in_chans, 1440, 721)
        
        upper = self.deconv_upper(x[:, :, 1:]) # (b*t, upper_in_chans, 13, 1440, 721)
        
        surface = einops.rearrange(surface,  '(b t) C h w -> b t C h w', b=B)
        upper = einops.rearrange(upper,  '(b t) C p h w -> b t C p h w', b=B)
        return surface, upper
    
    
if __name__ == "__main__":
    surface_in_chans = 4
    upper_in_chans = 5
    
    encoder = Era5DataEncodingToTokenSmall(surface_in_chans = surface_in_chans, upper_in_chans = upper_in_chans, dim = 64, final_dim=4096, if_surface_const_mask = False).cuda()
    decoder = TokenDecodingToEra5DataSmall(surface_in_chans = surface_in_chans, upper_in_chans = upper_in_chans, dim = 64, final_dim=4096).cuda()
    
    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         surface_u = torch.randn(1, 6, surface_in_chans, 720, 361).cuda()
    #         upper_u = torch.randn(1, 6, upper_in_chans, 13, 720, 361).cuda()
            
    #         encoder = Era5DataEncodingToTokenSmall(surface_in_chans = surface_in_chans, upper_in_chans = upper_in_chans, dim = 64, final_dim=4096, if_surface_const_mask = False).cuda()
    #         decoder = TokenDecodingToEra5DataSmall(surface_in_chans = surface_in_chans, upper_in_chans = upper_in_chans, dim = 64, final_dim=4096).cuda()
    #         token_embed = encoder(surface_u, upper_u)
    #         token_embed_ = torch.nn.functional.relu(token_embed)
    #         surface_u_, upper_u_ = decoder(token_embed_)
    #         recon_surface = torch.nn.functional.mse_loss(surface_u_, surface_u)
    #         recon_upper = torch.nn.functional.mse_loss(upper_u_, upper_u)
            
    #         (recon_upper + recon_surface).backward()
            
            
    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage", row_limit=10))
    
    
    encoder = Era5DataEncodingToTokenSmall(surface_in_chans = surface_in_chans, upper_in_chans = upper_in_chans, dim = 64, final_dim=4096, if_surface_const_mask = False).cuda()
    decoder = TokenDecodingToEra5DataSmall(surface_in_chans = surface_in_chans, upper_in_chans = upper_in_chans, dim = 64, final_dim=4096).cuda()
    for i in range(5000):
        surface_u = torch.randn(1, 7, surface_in_chans, 1440, 721).cuda()
        upper_u = torch.randn(1, 7, upper_in_chans, 13, 1440, 721).cuda()
        count_parameters(encoder)
        count_parameters(decoder)


        token_embed = encoder(surface_u, upper_u)

        token_embed = torch.randn(1, 504, 4096).cuda()
        token_embed_ = torch.nn.functional.relu(token_embed)
        surface_u_, upper_u_ = decoder(token_embed_)
        print(surface_u_.shape, upper_u_.shape)         # torch.Size([7, 4, 1440, 657]) torch.Size([7, 5, 13, 1440, 657])

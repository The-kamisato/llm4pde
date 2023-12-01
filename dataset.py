import torch
import xarray as xr
import numpy as np
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Lambda
from torchvision import datasets, transforms
from typing import Callable, Optional
import netCDF4 as nc

def is_nc_file(filename):
    return filename.endswith('.nc')

def nc_loader(path):
    surface_data = np.zeros((4, 721, 1440), dtype=np.float32) 
    
    with nc.Dataset(path) as nc_file:
        surface_data[0] = nc_file.variables['msl'][0].astype(np.float32)  
        surface_data[1] = nc_file.variables['u10'][0].astype(np.float32)
        surface_data[2] = nc_file.variables['v10'][0].astype(np.float32)
        surface_data[3] = nc_file.variables['t2m'][0].astype(np.float32)

    # 将数据转换为张量
    surface_data_tensor = torch.from_numpy(surface_data)
    surface_data_tensor.transpose(-1, -2)

    return surface_data_tensor

class NCDatasetFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader=nc_loader,
            extensions=None,  # 不再使用 IMG_EXTENSIONS
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        
        

transform_surface = transforms.Compose([
    transforms.Lambda(lambda x: x[:, ::2, ::2]),  # 对最后两个维度进行下采样
])
# 使用 NCDatasetFolder 来处理一系列 .nc 文件
dataset = NCDatasetFolder('/data/lbk/pangu_data/cli_download_data/nc_data', is_valid_file=is_nc_file, transform=transform_surface)
for sample in dataset:
    data_tensor = sample[0]
    print(data_tensor.shape)

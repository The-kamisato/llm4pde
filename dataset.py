import torch
import xarray as xr
import numpy as np
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Lambda
from torchvision import datasets, transforms
from typing import Callable, Optional, Tuple, Any
import netCDF4 as nc
from datetime import datetime, timedelta


def get_year_month_day(index):
    start_date = datetime(year=2010, month=1, day=1)  # 设置起始日期
    target_date = start_date + timedelta(days=index-1)  # 计算目标日期

    year = target_date.year
    month = target_date.month
    day = target_date.day

    return year, month, day

def is_nc_file(filename):
    return filename.endswith('.nc')

def nc_surface_loader(path):
     # "/data/lbk/pangu_data/cli_download_data/nc_data/years_surface_data/2001-2-surface.nc"
    year = int(path.split("/")[-1].split("-")[0])
    month = int(path.split("/")[-1].split("-")[1])
    if month in [1, 3, 5, 7, 8, 10, 12]:
        day = 31
    elif month == 2:
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            day = 29
        else:
            day = 28
    else:
        day = 30
    surface_data = np.zeros((4, day, 721, 1440), dtype=np.float32) 
    
    with nc.Dataset(path) as nc_file:
        surface_data[0] = nc_file.variables['msl'][::12].astype(np.float32)  
        surface_data[1] = nc_file.variables['u10'][::12].astype(np.float32)
        surface_data[2] = nc_file.variables['v10'][::12].astype(np.float32)
        surface_data[3] = nc_file.variables['t2m'][::12].astype(np.float32)

    # 将数据转换为张量
    surface_data_tensor = torch.from_numpy(surface_data)
    surface_data_tensor = surface_data_tensor.transpose(-1, -2)

    return surface_data_tensor

def nc_upper_loader(path):
    upper_data = np.zeros((5, 13, 721, 1440), dtype=np.float32) 
    # 加载 .nc 文件
    with nc.Dataset(path) as nc_file:
        upper_data[0] = nc_file.variables['z'][:].astype(np.float32)
        upper_data[1] = nc_file.variables['q'][:].astype(np.float32)
        upper_data[2] = nc_file.variables['t'][:].astype(np.float32)
        upper_data[3] = nc_file.variables['u'][:].astype(np.float32)
        upper_data[4] = nc_file.variables['v'][:].astype(np.float32)

    # 将数据转换为张量
    upper_data_tensor = torch.from_numpy(upper_data)
    upper_data_tensor = upper_data_tensor.transpose(-1, -2)
    
    return upper_data_tensor

class NCDatasetFolder(DatasetFolder):
    def __init__(
        self,
        root: str,
        loader_surface = nc_surface_loader,
        loader_upper = nc_upper_loader,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader=nc_surface_loader,
            extensions=None,  # 不再使用 IMG_EXTENSIONS
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        
        self.surface_keys = 'years_surface_data'
        self.upper_keys = 'years_upper_data'
        self.surface_to_index = {self.surface_keys: self.class_to_idx[self.surface_keys]}
        self.upper_to_index = {self.upper_keys: self.class_to_idx[self.upper_keys]}
        self.samples_surface = self.make_dataset(self.root, class_to_idx = self.surface_to_index, extensions = None, is_valid_file = is_valid_file)
        self.samples_upper = self.make_dataset(self.root, class_to_idx = self.upper_to_index, extensions = None, is_valid_file = is_valid_file)
        self.loader_surface = loader_surface
        self.loader_upper = loader_upper
    
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:       # 定义了对象在使用索引操作符 []
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        print("index:", index)
        
        # 取surface数据, (sample = self.loader(path)的shape为[4, 31/30/29/28, 1440, 721])
        year, month, day = get_year_month_day(index)
        
        file_index = (year - 2010) * 12 + month - 1             # 月份sruface.nc的文件位置
        
        sample_length = torch.randint(low = 2 , high = 8,size = (1, ))
        
        path, target = self.samples_surface[file_index]
        surface_sample_1 = self.loader_surface(path)
        
        print("surface_sample_path:", path)
        
        if (day + sample_length > surface_sample_1.shape[1]):   # 如果跨越到了下个月
            path, target = self.samples_surface[file_index + 1]      # 为了天的连续性，需要取出下个月份的前几天数据
            surface_sample_2 = self.loader_surface(path)
            surface_sample_1 = torch.cat((surface_sample_1, surface_sample_2[:, :6, :, :]), dim = 1)
            
        surface_sample = surface_sample_1[:, day - 1: day - 1 + sample_length]
            
        if self.transform is not None:
            surface_sample = self.transform(surface_sample)         # 对后两个维度下采样  
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        # 取upper数据, (sample = self.loader(path)的shape为[5, 13, 1440, 721])
        upper_sample = torch.zeros(5, sample_length, 13, 720, 361)
        for i in range(sample_length):
            path, target = self.samples_upper[index + i]
            upper_sample_part = self.loader_upper(path)
            print("upper_sample_part_path:", path)
            
            if self.transform is not None:
                upper_sample_part = self.transform(upper_sample_part)      # [5, 13, 1440, 721])对后两个维度下采样      
            if self.target_transform is not None:
                target = self.target_transform(target)
                
            upper_sample[:, i, :, :, :] = upper_sample_part
            
        return surface_sample, upper_sample, target
        
        

transform = transforms.Compose([
    transforms.Lambda(lambda x: x[:, :, ::2, ::2]),  # 对最后两个维度进行下采样
])

# 使用 NCDatasetFolder 来处理一系列 .nc 文件
dataset = NCDatasetFolder('/data/lbk/pangu_data/cli_download_data/nc_data', is_valid_file = is_nc_file, 
                          loader_surface = nc_surface_loader, loader_upper = nc_upper_loader, transform = transform)
sample = dataset[2001]
surface_data_tensor = sample[0]
upper_data_tensor = sample[1]
print(surface_data_tensor.shape)
print(upper_data_tensor.shape)

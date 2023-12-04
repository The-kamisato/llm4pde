import os
import netCDF4 as nc
import numpy as np
import torch
from class_dataset import nc_surface_loader, nc_upper_loader

extension = '.nc'
surface_folder_path = '/data/lbk/pangu_data/cli_download_data/nc_data/years_surface_data' 
upper_folder_path = '/data/lbk/pangu_data/cli_download_data/nc_data/years_upper_data'
def cal_mean_std(surface_folder_path = surface_folder_path, upper_folder_path = upper_folder_path):
    # 计算surface数据的所有均值和方差
    # 获取文件夹中所有的.nc文件
    # /data/lbk/pangu_data/cli_download_data/nc_data/years_surface_data/2010-01-surface.nc
    file_list = [file for file in os.listdir(surface_folder_path) if file.endswith(extension)]

    # 均值的计算
    surface_mean_list = torch.zeros(len(file_list), 4)
    # 逐个读取文件并将数据存入列表
    for i, file in enumerate(file_list):
        file_path = os.path.join(surface_folder_path, file)
        data_tensor = nc_surface_loader(file_path)      #  (4, day, 1440, 721)      day 是一个月的天数

        surface_mean_list[i] = torch.mean(data_tensor, dim = (1, 2, 3))
        
    all_surface_mean = torch.mean(surface_mean_list, dim = 0)                   # (4, )的张量

    # 标准差的计算
    surface_std_list = torch.zeros(len(file_list), 4)
    for i, file in enumerate(file_list):
        file_path = os.path.join(surface_folder_path, file)
        data_tensor = nc_surface_loader(file_path)      #  (4, day, 1440, 721)
        
        surface_std_list[i] = torch.sqrt(torch.mean(((data_tensor.permute(1, 2, 3, 0) - all_surface_mean) ** 2), dim = (0, 1, 2)))

    all_surface_std = torch.mean(surface_std_list, dim = 0)  


    # 计算upper数据的所有均值和方差

    file_list = [file for file in os.listdir(upper_folder_path) if file.endswith(extension)]

    # 均值的计算
    upper_mean_list = torch.zeros(len(file_list), 5, 13)            # 注意到它5个变量随着高度变化挺大，因此分别求出相应的均值和标准差

    for i, file in enumerate(file_list):
        file_path = os.path.join(upper_folder_path, file)
        data_tensor = nc_upper_loader(file_path)                    # (5, 13, 1440, 721)
        
        upper_mean_list[i] = torch.mean(data_tensor, dim = (2, 3))
        
    all_upper_mean = torch.mean(upper_mean_list, dim = 0)       # (5, 13)
        
    # 标准差的计算
    upper_std_list = torch.zeros(len(file_list), 5, 13)            # 注意到它5个变量随着高度变化挺大，因此分别求出相应的均值和标准差
    for i, file in enumerate(file_list):
        file_path = os.path.join(upper_folder_path, file)
        data_tensor = nc_upper_loader(file_path)                    # (5, 13, 1440, 721)
        
        upper_std_list[i] = torch.sqrt(torch.mean(((data_tensor.permute(2, 3, 0, 1) - all_upper_mean) ** 2), dim = (0, 1)))
        
    all_upper_std = torch.mean(upper_std_list, dim = 0)       # (5, 13)
    
    return all_surface_mean, all_surface_std, all_upper_mean, all_upper_std

if __name__ == "__main__":
    all_surface_mean, all_surface_std, all_upper_mean, all_upper_std = cal_mean_std()
    print(all_surface_mean)
    print(all_upper_mean)
    

import torch
import xarray as xr
import numpy as np
from typing import Callable, Optional, Tuple, Any
import netCDF4 as nc
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, Dataset
import time
import lmdb
from torch.profiler import profile, record_function, ProfilerActivity

def get_year_month_day(index):
    start_date = datetime(year=2010, month=1, day=1)  # 设置起始日期
    target_date = start_date + timedelta(days=index)  # 计算目标日期

    year = target_date.year
    month = target_date.month
    day = target_date.day

    return year, month, day

class LmdbDataset(Dataset):
    def __init__(self, 
                 surface_data_path = "/data/lbk/pangu_data/cli_download_data/tensored_years_surface_data/surface_data_720_361.pt", 
                 upper_data_lmdb_path = "/data1/lbk/upper_lmdb"):
        # https://github.com/chainer/chainermn/issues/129
	    # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self.env = None
        self.db_path = upper_data_lmdb_path
        self.samples_surface = torch.load(surface_data_path)
        
        print("samples_surface loaded over")
	
	# Workaround to have length from the start for ImageNet since we don't have LMDB at initialization time

	
    def _init_db(self):
        self.env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=True, meminit=False)
        self.txn = self.env.begin()

    def __getitem__(self, index):
        # time1 = time.time()
        self._init_db()
        # time2 = time.time()
        # print("***************", time2 - time1)
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        year, month, day = get_year_month_day(index)
        if year == 2021:
            sample_length = min(7, len(self.samples_surface) - index)
        else:
            sample_length = min(torch.randint(low = 2 , high = 8, size = (1, )),  len(self.samples_surface) - index - 365)

        surface_sample = self.samples_surface[index: index + sample_length]
    
        # 取upper数据, (sample = self.loader(path)的shape为[5, 13, 1440, 721])
        upper_sample = torch.zeros(sample_length, 5, 13, 720, 361)


        for i in range(sample_length):
            # time1 = time.time()
            upper_sample[i] = torch.tensor(np.frombuffer(self.txn.get(str(index + i).encode()), dtype=np.float32)).reshape(5, 13, 720, 361)
            # time2 = time.time()
            # print(time2 - time1)
            
        self.env.close()            
        # env = lmdb.open(self.db_path, map_size = 1099511627776, readonly=True, lock=False, readahead=True, meminit=False)
        # with env.begin(write=False) as txn:
        #     for i in range(sample_length):
        #         time1 = time.time()
        #         upper_sample[i] = torch.tensor(np.frombuffer(txn.get(str(index + i).encode()), dtype=np.float32)).reshape(5, 13, 720, 361)
        #         time2 = time.time()
        #         print(time2 - time1)
                
        
        return surface_sample, upper_sample

    def __len__(self) -> int:
        return len(self.samples_surface)

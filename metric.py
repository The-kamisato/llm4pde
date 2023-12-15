import torch

def cal_rqe(logits, target):
  q = -torch.linspace(1, 4, steps=50)
  q = 1-10**q
  q = q.cuda()
  if len(logits.shape)==6:
    logits_quantile = torch.quantile(logits.view(logits.shape[1], logits.shape[3], logits.shape[4]*logits.shape[5]), q, dim=2)
    target_quantile = torch.quantile(target.view(logits.shape[1], logits.shape[3], logits.shape[4]*logits.shape[5]), q, dim=2)
  elif len(logits.shape)==5:
    logits_quantile = torch.quantile(logits.view(logits.shape[1], logits.shape[3]*logits.shape[4]), q, dim=1)
    target_quantile = torch.quantile(target.view(logits.shape[1], logits.shape[3]*logits.shape[4]), q, dim=1)
  RQE = (logits_quantile-target_quantile)/target_quantile
  RQE = RQE.mean(dim=0)
  return RQE, logits_quantile, target_quantile

def cal_acc(logits, target, logits_mean, lat_weight):
  logits_long_time = logits - logits_mean
  target_long_time = target - logits_mean
  acc_up = lat_weight * target_long_time * logits_long_time
  if len(logits.shape)==6:
    acc_up = torch.mean(acc_up, dim=(0, 2, 4, 5))
    acc_down = torch.mean(lat_weight * logits_long_time * logits_long_time, dim=(0, 2, 4, 5)) * torch.mean(lat_weight * target_long_time * target_long_time, dim=(0, 2, 4, 5))
  elif len(logits.shape)==5:
    acc_up = torch.mean(acc_up, dim=(0, 2, 3, 4))
    acc_down = torch.mean(lat_weight * logits_long_time * logits_long_time, dim=(0, 2, 3, 4)) * torch.mean(lat_weight * target_long_time * target_long_time, dim=(0, 2, 3, 4))
  acc_down = torch.sqrt(acc_down)
  acc = acc_up / acc_down
  return acc

# def cal_mae_mse(logits, target, lat_weight):
#   residual2 = (logits-target) * (logits-target) * lat_weight    # 残差平方
#   residual = (logits-target) * lat_weight             
#   residual = torch.abs(residual)                                # 绝对残差
#   if len(logits.shape)==6:      
#     mae_all = torch.mean(residual, dim=(0, 2, 3, 4, 5))   # (batch, T, C, p, H, W)需要转为 (batch, C, T, p, H, W), 在外面使用.transpose(1, 2)
#     mae_height = torch.mean(residual, dim=(0, 2, 4, 5))
#     mse_all = torch.sqrt(torch.mean(residual2, dim=(0, 2, 3, 4, 5)))    
#     mse_height = torch.sqrt(torch.mean(residual2, dim=(0, 2, 4, 5)))      # mse_height是(5, 13)的张量, [2, 2]对应的是T850, [0, 5]对应的是Z500
#     return mae_all, mse_all, mae_height, mse_height
#   elif len(logits.shape)==5:                              # (batch, T, C, H, W) 需要转为 (batch, C, T, H, W) 在外面使用.transpose(1, 2)
#     mae_all = torch.mean(residual, dim=(0, 2, 3, 4))
#     mse_all = torch.sqrt(torch.mean(residual2, dim=(0, 2, 3, 4)))
#     return mae_all, mse_all

def cal_mae_mse(logits, target, lat_weight):
  residual2 = (logits-target) * (logits-target) * lat_weight    # 残差平方
  residual = (logits-target) * lat_weight             
  residual = torch.abs(residual)                                # 绝对残差
  if len(logits.shape)==6:      
    mae_all = torch.mean(residual, dim=(0, 2, 3, 4, 5))    # (batch, T, C, p, H, W)需要转为 (batch, C, T, p, H, W), 在外面使用.transpose(1, 2)
    mae_height = torch.mean(residual, dim=(0, 2, 4, 5))
    
    mse_all = torch.mean(torch.sqrt(torch.mean(residual2, dim=(3, 4, 5))), dim = (0, 2))    
    mse_height = torch.mean(torch.sqrt(torch.mean(residual2, dim=(4, 5))), dim = (0, 2))     # mse_height是(5, 13)的张量, [2, 2]对应的是T850, [0, 5]对应的是Z500
    return mae_all, mse_all, mae_height, mse_height
  elif len(logits.shape)==5:                              # (batch, T, C, H, W) 需要转为 (batch, C, T, H, W) 在外面使用.transpose(1, 2)
    mae_all = torch.mean(residual, dim=(0, 2, 3, 4))
    mse_all = torch.mean(torch.sqrt(torch.mean(residual2, dim=(3, 4))), dim = (0, 2))
    return mae_all, mse_all

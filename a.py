import os
import pandas as pd
import torch

center = torch.tensor([[-1.3139e+00],
         [ 1.0176e+00],
         [ 1.0200e+00],
         [-3.7354e+00],
         [ 1.0356e+01],
         [-1.1160e+00],
         [-2.3052e+00],
         [-1.2784e+00],
         [ 4.4933e+00],
         [-2.2510e+00],
         [ 9.0434e+00],
         [ 1.0298e+00],
         [ 7.1134e-01],
         [-4.2999e-01],
         [ 7.2536e-01],
         [ 2.6529e+00],
         [ 8.7143e+00],
         [-2.8185e-01],
         [-2.1479e-02],
         [ 3.7487e-02],
         [ 6.2032e+00],
         [-2.7908e+01],
         [-1.0620e-01],
         [-3.0666e+01],
         [-1.3483e-01]])

scale = torch.tensor([[6.4029e+01],
         [7.1535e+01],
         [1.4170e+02],
         [8.8454e+00],
         [6.6249e+00],
         [4.4105e+00],
         [3.8370e+01],
         [6.8555e+01],
         [1.2289e+02],
         [5.9683e+00],
         [4.9365e+00],
         [2.6096e+00],
         [2.1315e+01],
         [4.5228e+01],
         [8.1414e+01],
         [3.9808e+00],
         [4.4334e+00],
         [1.8336e+00],
         [1.9111e-01],
         [7.6550e-02],
         [5.3170e+00],
         [2.7279e+01],
         [6.0311e+01],
         [2.7828e+01],
         [1.0717e+02]])

# # 加载预训练模型
# device = torch.device('cpu')
# model_path = os.path.join("logs", "trained_tcn.tar")
# # model_path = os.path.join("logs", "trained_tcn_encoders_only.tar")
# model_info = torch.load(model_path, map_location = device)
# print(model_info.keys())
# del model_info["state_dict"]
# print(model_info)

# # 读取CSV文件
# file_path = 'data/BT23/dynamic_walk_1_1/Joint_Moments_Filt.csv'
# label_names = ["hip_flexion_r_moment", "knee_angle_r_moment"]
# device = torch.device('cpu')
# df = pd.read_csv(file_path)
# label_data = torch.tensor(df[label_names].values, device = device)
# print(label_data.size())
# label_data = label_data.transpose(0, 1).unsqueeze(0).float()

# # 验证zip的功能
# data = [[1,'a'], [2,'b']]
# nums, strs = zip(*data)
# print(nums)
# print(strs)

# # 验证RMSE、R²和Normalized MAE的计算过程以及复现RMSE非常小时R²和Normalized MAE值异常的情况
# scale = 10
# # est = torch.rand(10, 50) * scale
# # lbl = torch.rand(10, 50)
# # lbl = est.clone()
# est = torch.tensor([[0.5,0.5001,0.5002,0.5003,0.5004,0.5005,0.5006,0.5007,0.5008,0.5009,0.5010,0.5011,0.5012,0.5013,0.5014,0.5015,0.5016,0.5017,0.5018,0.5019,0.5020]])
# lbl = torch.tensor([[0.55,0.5501,0.5502,0.5503,0.5504,0.5505,0.5506,0.5507,0.5508,0.5509,0.5510,0.5511,0.5512,0.5513,0.5514,0.5515,0.5516,0.5517,0.5518,0.5519,0.5520]])
#
# # est[0,:] = float('nan')
#
# # 创建有效数据掩码
# valid_mask = ~torch.isnan(est) & ~torch.isnan(lbl)  # [batch_size, seq_len]
#
# # ============ 1. RMSE: 在所有有效数据点上计算 ============
# est_flat = est[valid_mask]
# lbl_flat = lbl[valid_mask]
# rmse = torch.sqrt(torch.mean((est_flat - lbl_flat) ** 2))
#
# print(f'rmse:{rmse}')
#
# # ============ 2. R²: 向量化计算（每个序列） ============
# # 计算每个序列的有效元素数量
# valid_counts = valid_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
#
# # 用0填充NaN位置（配合mask使用，不影响计算）
# est_filled = torch.where(valid_mask, est, torch.zeros_like(est))
# lbl_filled = torch.where(valid_mask, lbl, torch.zeros_like(lbl))
#
# # 计算每个序列的label均值（只对有效元素）
# lbl_sum = (lbl_filled * valid_mask.float()).sum(dim=1, keepdim=True)  # [batch_size, 1]
# lbl_mean = lbl_sum / (valid_counts + 1e-8)  # [batch_size, 1]
#
# # 计算ss_res（残差平方和）
# diff = lbl_filled - est_filled  # [batch_size, seq_len]
# diff_sq = diff ** 2
# ss_res = (diff_sq * valid_mask.float()).sum(dim=1)  # [batch_size]
# print(f'ss_res:{ss_res}')
#
# # 计算ss_tot（总平方和）
# lbl_centered = lbl_filled - lbl_mean  # [batch_size, seq_len]
# lbl_centered_sq = lbl_centered ** 2
# ss_tot = (lbl_centered_sq * valid_mask.float()).sum(dim=1)  # [batch_size]
# print(f'ss_tot:{ss_tot}')
#
# # 计算R²
# r2 = 1 - (ss_res / (ss_tot + 1e-8))  # [batch_size]
#
# # 只保留有效序列（至少有1个有效点）
# valid_sequences = valid_counts.squeeze() > 0  # [batch_size]
# r2_valid = r2[valid_sequences]
# r2_mean = r2_valid.mean() if len(r2_valid) > 0 else torch.tensor(0.0, device=r2.device)
#
# print(f'r2_mean:{r2_mean}')
#
# # ============ 3. Normalized MAE: 向量化计算 ============
# # 计算每个序列的label range（使用masked操作）
# # 对于min，将无效位置设为inf；对于max，将无效位置设为-inf
# lbl_for_max = torch.where(valid_mask, lbl_filled,
#                           torch.full_like(lbl_filled, -float('inf')))
# lbl_for_min = torch.where(valid_mask, lbl_filled,
#                           torch.full_like(lbl_filled, float('inf')))
#
# lbl_max = lbl_for_max.max(dim=1, keepdim=True)[0]  # [batch_size, 1]
# lbl_min = lbl_for_min.min(dim=1, keepdim=True)[0]  # [batch_size, 1]
# label_range = lbl_max - lbl_min  # [batch_size, 1]
# print(f'label_range:{label_range}')
#
# # 计算MAE
# abs_diff = torch.abs(lbl_filled - est_filled)  # [batch_size, seq_len]
# mae_sum = (abs_diff * valid_mask.float()).sum(dim=1, keepdim=True)  # [batch_size, 1]
# mae = mae_sum / (valid_counts + 1e-8)  # [batch_size, 1]
# print(f'mae:{mae}')
#
# # 归一化为百分数
# mae_percent = (mae / (label_range + 1e-12)) * 100.0  # [batch_size, 1]
#
# # 只保留有效序列
# mae_percent_valid = mae_percent.squeeze()[valid_sequences]
# mae_percent_mean = mae_percent_valid.mean() if len(mae_percent_valid) > 0 else torch.tensor(0.0,
#                                                                                             device=mae_percent.device)
# print(f'mae_percent_mean:{mae_percent_mean}')
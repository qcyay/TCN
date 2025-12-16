import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import OrderedDict

from models.predictor_model import PredictorTransformer
from models.generative_model import GenerativeTransformer
from models.tcn import TCN

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

# # 验证csv文件读取功能
# def read_csv_first_row_to_tensor(csv_file_path, skip_header=True, dtype=torch.float32):
#     """
#     读取CSV文件的第一行数据并转换为PyTorch张量
#
#     参数:
#         csv_file_path (str): CSV文件的路径
#         skip_header (bool): 是否跳过标题行，默认为True
#         dtype: 目标张量的数据类型，默认为torch.float32
#
#     返回:
#         torch.Tensor: 包含第一行数据的PyTorch张量
#     """
#     try:
#         # 检查文件是否存在
#         if not Path(csv_file_path).exists():
#             raise FileNotFoundError(f"文件未找到: {csv_file_path}")
#
#         # 使用pandas读取CSV文件
#         # header=None表示不将第一行作为列名，skiprows跳过标题行（如果skip_header为True）
#         skip_rows = 1 if skip_header else 0
#         df = pd.read_csv(csv_file_path, header=None, skiprows=skip_rows, nrows=1)
#
#         # 检查是否成功读取到数据
#         if df.empty:
#             raise ValueError("CSV文件为空或没有数据行")
#
#         # 将第一行数据转换为numpy数组
#         first_row_np = df.iloc[0].values
#
#         # 将numpy数组转换为PyTorch张量
#         tensor = torch.tensor(first_row_np, dtype=dtype)
#
#         print(f"成功读取CSV文件: {csv_file_path}")
#         print(f"第一行数据形状: {tensor.shape}")
#         print(f"数据类型: {tensor.dtype}")
#         print(f"数据值: {tensor}")
#
#         return tensor
#
#     except FileNotFoundError as e:
#         print(f"错误: {e}")
#         return None
#     except pd.errors.EmptyDataError:
#         print("错误: CSV文件为空")
#         return None
#     except Exception as e:
#         print(f"读取CSV文件时发生错误: {e}")
#         return None
#
# csv_path = "a.csv"  # 替换为你的CSV文件路径
# result_tensor = read_csv_first_row_to_tensor(csv_path, skip_header=True)
#
# # 检查NaN
# if torch.isnan(result_tensor).any():
#     print("警告：张量包含NaN值")
#     print(f"NaN数量: {torch.isnan(result_tensor).sum().item()}")
# else:
#     print("张量数据正常")

# # 验证加载模型功能
# def load_model(model_path: str, device: torch.device):
#     """加载训练好的模型"""
#     print(f"加载模型: {model_path}")
#     checkpoint = torch.load(model_path, map_location=device)
#
#     model_type = checkpoint.get("model_type", "TCN")
#     print(f"模型类型: {model_type}")
#
#     if model_type == "GenerativeTransformer":
#         model = GenerativeTransformer(
#             input_size=checkpoint["input_size"],
#             output_size=checkpoint["output_size"],
#             d_model=checkpoint["d_model"],
#             nhead=checkpoint["nhead"],
#             num_encoder_layers=checkpoint["num_encoder_layers"],
#             num_decoder_layers=checkpoint["num_decoder_layers"],
#             dim_feedforward=checkpoint["dim_feedforward"],
#             dropout=checkpoint["dropout"],
#             sequence_length=checkpoint["sequence_length"],
#             encoder_type=checkpoint["encoder_type"],
#             use_positional_encoding=checkpoint["use_positional_encoding"],
#             center=checkpoint["center"],
#             scale=checkpoint["scale"]
#         ).to(device)
#     elif model_type == "Transformer":
#         model = PredictorTransformer(
#             input_size=checkpoint["input_size"],
#             output_size=checkpoint["output_size"],
#             d_model=checkpoint["d_model"],
#             nhead=checkpoint["nhead"],
#             num_encoder_layers=checkpoint["num_encoder_layers"],
#             dim_feedforward=checkpoint["dim_feedforward"],
#             dropout=checkpoint["dropout"],
#             sequence_length=checkpoint["sequence_length"],
#             output_sequence_length=checkpoint["output_sequence_length"],
#             use_positional_encoding=checkpoint["use_positional_encoding"],
#             center=checkpoint["center"],
#             scale=checkpoint["scale"]
#         ).to(device)
#     else:  # TCN
#         model = TCN(
#             input_size=checkpoint["input_size"],
#             output_size=checkpoint["output_size"],
#             num_channels=checkpoint["num_channels"],
#             ksize=checkpoint["ksize"],
#             dropout=checkpoint["dropout"],
#             eff_hist=checkpoint["eff_hist"],
#             spatial_dropout=checkpoint["spatial_dropout"],
#             activation=checkpoint["activation"],
#             norm=checkpoint["norm"],
#             center=checkpoint["center"],
#             scale=checkpoint["scale"]
#         ).to(device)
#
#     state_dict = checkpoint['state_dict']
#     # 打印模型参数信息
#     total_params = sum(p.numel() for p in state_dict.values())
#     print(f"✓ 模型参数总数: {total_params:,}")
#     print("✓ 参数形状:")
#     for key, value in list(state_dict.items())[:5]:  # 只显示前5个
#         print(f"  {key}: {value.shape}")
#         print(f"  {key}: {value}")
#     if len(state_dict) > 5:
#         print(f"  ... 还有 {len(state_dict) - 5} 个参数")
#
#     model.load_state_dict(checkpoint["state_dict"])
#     print(f"模型加载成功! Epoch: {checkpoint.get('epoch', 'N/A')}")
#
#     return model, model_type
#
# # 加载模型
# model, model_type = load_model('logs/trained_tcn_default_config/2/best_model.tar', 'cpu')
# # model, model_type = load_model('logs/trained_tcn_default_config/2/model_epoch_1.tar', 'cpu')

positions = torch.arange(i, min(i + output_seq_len, full_len), device=trial_est.device)
valid_length = len(positions)  # 直接获取有效长度

# 使用有效的子序列部分
est_full.index_add_(1, positions, trial_est[i, :, :valid_length])
lbl_full.index_add_(1, positions, trial_lbl[i, :, :valid_length])
count.index_add_(1, positions, torch.ones(num_outputs, valid_length, device=trial_est.device))
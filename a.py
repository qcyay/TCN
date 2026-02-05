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

# 加载预训练模型
device = torch.device('cpu')
model_path = os.path.join("logs", "trained_tcn.tar")
# model_path = os.path.join("logs", "trained_tcn_encoders_only.tar")
model_info = torch.load(model_path, map_location = device)
print(model_info.keys())
del model_info["state_dict"]
print(model_info)

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

a=torch.rand(10,25,3)
a=(a-center)/scale
print(a.size())
import os
import torch

# ==================== 模型选择 ====================
# 模型类型: 'TCN', 'Transformer', 或 'GenerativeTransformer'
# model_type = 'TCN'
model_type = 'Transformer'
# model_type = 'GenerativeTransformer'

# ==================== 数据配置 ====================

# 相对路径:训练好的模型保存位置
model_path = os.path.join("logs", "trained_model.tar")

# 相对路径:数据目录
data_dir = 'data/example'

# 使用的身体侧别 (l: 左侧, r: 右侧)
side = "r"

# ==================== Activity Flag 配置 ====================
# 是否启用activity_flag掩码功能
# True: 读取activity_flag.csv文件，只在掩码为1的位置计算损失和指标
# False: 不使用activity_flag，在所有位置计算损失和指标
activity_flag = False

# 模型输入特征名称(* 会被替换为 side)
input_names = [
    "foot_imu_*_gyro_x", "foot_imu_*_gyro_y", "foot_imu_*_gyro_z",
    "foot_imu_*_accel_x", "foot_imu_*_accel_y", "foot_imu_*_accel_z",
    "shank_imu_*_gyro_x", "shank_imu_*_gyro_y", "shank_imu_*_gyro_z",
    "shank_imu_*_accel_x", "shank_imu_*_accel_y", "shank_imu_*_accel_z",
    "thigh_imu_*_gyro_x", "thigh_imu_*_gyro_y", "thigh_imu_*_gyro_z",
    "thigh_imu_*_accel_x", "thigh_imu_*_accel_y", "thigh_imu_*_accel_z",
    "insole_*_cop_x", "insole_*_cop_z", "insole_*_force_y",
    "hip_angle_*", "hip_angle_*_velocity_filt",
    "knee_angle_*", "knee_angle_*_velocity_filt"
]

# 模型输出(预测)的标签名称
label_names = ["hip_flexion_*_moment", "knee_angle_*_moment"]
# label_names = ["knee_angle_*_moment"]

# 模型预测的延迟(单位:数据点)
# 数据采样率为200Hz,每个点代表5ms
model_delays = [10, 0]

# 参与者体重字典(单位:kg)
participant_masses = {
    "BT01": 80.59, "BT02": 72.24, "BT03": 95.29, "BT04": 98.23,
    "BT06": 79.33, "BT07": 64.49, "BT08": 69.13, "BT09": 82.31,
    "BT10": 93.45, "BT11": 50.39, "BT12": 78.15, "BT13": 89.85,
    "BT14": 67.30, "BT15": 58.40, "BT16": 64.33, "BT17": 60.03,
    "BT18": 67.96, "BT19": 69.95, "BT20": 55.44, "BT21": 58.85,
    "BT22": 76.79, "BT23": 67.23, "BT24": 77.79
}

# ==================== 通用模型配置 ====================

# 输入特征数量(如果启用特征选择，会自动更新)
input_size = len(input_names)

# 输出特征数量(自动计算)
output_size = len(label_names)

# 输入特征归一化参数（对应所有25个原始特征）
# 注意：如果启用特征选择，这些参数会自动根据selected_feature_indices进行调整
center = torch.tensor([
    [-1.3139e+00], [1.0176e+00], [1.0200e+00], [-3.7354e+00], [1.0356e+01],
    [-1.1160e+00], [-2.3052e+00], [-1.2784e+00], [4.4933e+00], [-2.2510e+00],
    [9.0434e+00], [1.0298e+00], [7.1134e-01], [-4.2999e-01], [7.2536e-01],
    [2.6529e+00], [8.7143e+00], [-2.8185e-01], [-2.1479e-02], [3.7487e-02],
    [6.2032e+00], [-2.7908e+01], [-1.0620e-01], [-3.0666e+01], [-1.3483e-01]
], dtype=torch.float32)

scale = torch.tensor([
    [6.4029e+01], [7.1535e+01], [1.4170e+02], [8.8454e+00], [6.6249e+00],
    [4.4105e+00], [3.8370e+01], [6.8555e+01], [1.2289e+02], [5.9683e+00],
    [4.9365e+00], [2.6096e+00], [2.1315e+01], [4.5228e+01], [8.1414e+01],
    [3.9808e+00], [4.4334e+00], [1.8336e+00], [1.9111e-01], [7.6550e-02],
    [5.3170e+00], [2.7279e+01], [6.0311e+01], [2.7828e+01], [1.0717e+02]
], dtype=torch.float32)

# ==================== TCN专用配置 ====================

# TCN每层的通道数
num_channels = [64, 64]

# 卷积核大小
ksize = 3

# TCN Dropout比率
dropout = 0.2

# 是否使用空间dropout
spatial_dropout = False

# 激活函数类型
activation = 'ReLU'

# 归一化方法
norm = 'weight_norm'

# 模型的有效历史长度(感受野)
eff_hist = 248

# ==================== Transformer预测模型配置 ====================

# 输入序列长度（滑动窗口大小）
sequence_length = 200

# 输出序列长度（预测或生成的序列长度）
# 对于预测模型：决定预测的时间步数
# 对于生成式模型：决定生成的时间步数
output_sequence_length = 1

# Transformer隐藏维度
d_model = 128

# 注意力头数
nhead = 8

# Encoder层数
num_encoder_layers = 3

# 前馈网络维度
dim_feedforward = 256

# Transformer的dropout
transformer_dropout = 0.1

# 是否使用位置编码
use_positional_encoding = True

# ==================== 生成式Transformer配置 ====================

# 编码器类型: 'transformer' 或 'linear'
encoder_type = 'transformer'  # 'transformer' 使用Transformer Encoder, 'linear' 使用线性层

# 生成式模型专用参数
gen_d_model = 128              # 生成模型的隐藏维度
gen_nhead = 8                  # 注意力头数
gen_num_encoder_layers = 3     # 编码器层数（如果使用transformer encoder）
gen_num_decoder_layers = 3     # 解码器层数
gen_dim_feedforward = 512      # 前馈网络维度
gen_dropout = 0.1              # Dropout比率
gen_sequence_length = 100      # 编码器输入序列长度

# 自回归生成时的起始token值（可以是0或其他合理值）
start_token_value = 0.0

# Teacher forcing比率（训练时使用真实值的概率，1.0表示总是使用真实值）
teacher_forcing_ratio = 1.0

# ==================== 评估配置 ====================

# 序列重组方法（仅用于Transformer和GenerativeTransformer）
# 'only_first': 每个短序列只取第一个预测值（推荐，速度快，接近实际使用）
# 'average': 对每个位置的所有预测值取平均（结果更平滑，但计算慢）
# 注意：TCN模型不使用此参数
reconstruction_method = 'only_first'

# ==================== 训练配置 ====================

# 训练轮数
num_epochs = 500

# 训练批次大小
batch_size = 128

# 测试批次大小（可以与训练批次大小不同）
test_batch_size = 1024

# 学习率
learning_rate = 0.0001

# 权重衰减(L2正则化系数)
weight_decay = 1e-5

# 验证间隔(每隔多少轮在测试集上验证一次)
val_interval = 1

# 模型保存间隔(每隔多少轮保存一次模型)
save_interval = 20

# ==================== 训练批次比例配置 ====================
# 每个epoch使用的批次比例（0.0-1.0）
# 1.0表示使用全部批次，0.4表示每个epoch只训练前40%的批次
# 这个参数对于大数据集的Transformer训练特别有用，可以显著加快训练速度
# 推荐值：
#   - 小数据集(总批次<1000): 1.0 (使用全部数据)
#   - 中等数据集(总批次1000-5000): 0.5-0.8
#   - 大数据集(总批次>5000): 0.3-0.5
# 注意：DataLoader本身是shuffle的，所以每个epoch训练的batch都是随机的
train_batch_ratio = 0.4

# ==================== 学习率调度器配置 ====================

# ReduceLROnPlateau 学习率衰减因子
scheduler_factor = 0.9

# ReduceLROnPlateau 耐心值
scheduler_patience = 10

# 最小学习率
min_lr = 1e-6

# ==================== 早停配置 ====================

# 早停耐心值
early_stopping_patience = 50

# 早停最小改善幅度
early_stopping_min_delta = 0.0

# ==================== 其他配置 ====================

# 随机种子(保证结果可复现)
random_seed = 42

# 梯度裁剪阈值(防止梯度爆炸)
grad_clip = 1.0
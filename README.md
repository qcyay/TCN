# TCN关节力矩预测项目

基于时间卷积网络(TCN)的人体关节力矩预测系统，用于从可穿戴传感器数据预测指定延迟时刻的关节力矩。

## 项目结构

```
project/
├── configs/
│   └── TCN/
│       └── default_config.py          # 配置文件
├── dataset_loaders/
│   └── dataloader.py                  # 数据加载器
├── models/
│   └── tcn.py                         # TCN网络实现
├── utils/
│   └── config_utils.py                # 配置加载工具
├── data/
│   └── example/
│       ├── train/                     # 训练数据
│       │   └── 参与者名/
│       │       └── 运动类型/
│       │           ├── 参与者_运动_exo.csv          # 传感器数据
│       │           └── 参与者_运动_moment_filt.csv  # 力矩真值
│       └── test/                      # 测试数据
│           └── (同上结构)
├── logs/                              # 训练日志和模型保存目录
│   └── trained_tcn_配置名/
│       └── 序号/
│           ├── config.py              # 训练时使用的配置文件副本
│           ├── train_log.txt          # 训练日志
│           ├── validation_log.txt     # 验证日志
│           └── model_epoch_*.tar      # 模型检查点
├── train.py                           # 训练脚本
├── test.py                            # 测试脚本
└── README.md                          # 本文件
```

## 数据格式

### 数据采样率
- 200 Hz（每个数据点代表 5ms）

### 传感器数据文件 (*_exo.csv)
包含以下类型的传感器数据：
- **IMU数据**（足部、小腿、大腿）：
  - 陀螺仪：gyro_x, gyro_y, gyro_z
  - 加速度计：accel_x, accel_y, accel_z
- **压力鞋垫数据**：
  - cop_x, cop_z（压力中心位置）
  - force_y（垂直力）
- **关节角度**：
  - hip_angle, knee_angle
  - 角度变化速率（已滤波）

### 力矩真值文件 (*_moment_filt.csv)
包含关节力矩的真实值：
- hip_flexion_moment（髋关节屈曲力矩）
- knee_angle_moment（膝关节力矩）

## 配置说明

配置文件位于 `configs/TCN/default_config.py`，主要参数包括：

### 数据配置
- `data_dir`: 数据目录路径
- `side`: 使用的身体侧别（'l' 或 'r'）
- `input_names`: 输入特征名称列表
- `label_names`: 输出标签名称列表
- `model_delays`: 预测延迟（单位：数据点）
- `participant_masses`: 参与者体重字典

### 模型配置
- `num_channels`: TCN各层通道数，例如 [64, 64, 64, 64, 64, 64, 64, 64]
- `ksize`: 卷积核大小
- `dropout`: Dropout比率
- `eff_hist`: 有效历史长度（感受野）
- `center` 和 `scale`: 输入归一化参数

### 训练配置
- `num_epochs`: 训练轮数
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `val_interval`: 验证间隔
- `save_interval`: 模型保存间隔

## 使用方法

### 1. 准备数据

确保数据按照以下结构组织：
```
data/example/
├── train/
│   ├── BT23/
│   │   ├── walking/
│   │   │   ├── BT23_walking_exo.csv
│   │   │   └── BT23_walking_moment_filt.csv
│   │   └── running/
│   │       └── ...
│   └── BT24/
│       └── ...
└── test/
    └── (同上结构)
```

### 2. 计算输入归一化参数（重要！）

在训练前，需要计算训练集的统计信息并更新配置文件中的 `center` 和 `scale`：

```python
# 计算训练集输入特征的均值和标准差
import torch
from dataset_loaders.dataloader import TcnDataset
from configs.TCN import default_config as config

# 加载训练数据
input_names = [name.replace("*", config.side) for name in config.input_names]
train_dataset = TcnDataset(
    data_dir=config.data_dir,
    input_names=input_names,
    label_names=[name.replace("*", config.side) for name in config.label_names],
    side=config.side,
    participant_masses=config.participant_masses,
    device=torch.device("cpu"),
    mode='train'
)

# 收集所有训练数据
all_inputs = []
for i in range(len(train_dataset)):
    input_data, _, _ = train_dataset[i]
    all_inputs.append(input_data)

all_inputs = torch.cat(all_inputs, dim=2)  # 合并所有序列

# 计算均值和标准差
center = torch.mean(all_inputs, dim=(0, 2), keepdim=True)
scale = torch.std(all_inputs, dim=(0, 2), keepdim=True)

print("Center shape:", center.shape)  # [1, num_features, 1]
print("Scale shape:", scale.shape)    # [1, num_features, 1]

# 将这些值填入配置文件的 center 和 scale
```

### 3. 训练模型

```bash
# 使用默认配置训练
python train.py

# 指定配置文件
python train.py --config_path configs.TCN.default_config

# 指定GPU设备
python train.py --device cuda

# 从检查点恢复训练
python train.py --resume logs/trained_tcn_default_config/0/model_epoch_50.tar
```

### 4. 监控训练过程

训练日志会实时保存到 `logs/trained_tcn_配置名/序号/` 目录：
- `train_log.txt`: 记录每轮的训练损失
- `validation_log.txt`: 记录定期验证的评估指标

### 5. 测试模型

```bash
# 测试训练好的模型
python test.py --config_path configs.TCN.default_config --device cuda
```

## 评估指标

模型在测试集上计算以下指标：

1. **RMSE (Root Mean Square Error)**：均方根误差
   - 单位：Nm/kg
   - 衡量预测值与真值的整体偏差

2. **R² (Coefficient of Determination)**：决定系数
   - 范围：(-∞, 1]，越接近1越好
   - 衡量模型对数据变化的解释能力

3. **归一化MAE (Normalized Mean Absolute Error)**：归一化平均绝对误差
   - 使用真值序列的范围（max - min）进行归一化
   - 衡量相对误差大小

## 训练输出示例

### 训练过程
```
Epoch [1/100] - 训练损失: 0.125436
Epoch [2/100] - 训练损失: 0.098723
...
Epoch [5/100] - 训练损失: 0.067845

在测试集上进行验证 (Epoch 5)...
=== Epoch 5 验证结果 ===
hip_flexion_r_moment:
  RMSE: 0.2345 Nm/kg
  R²: 0.8567
  归一化MAE: 0.1234

knee_angle_r_moment:
  RMSE: 0.1876 Nm/kg
  R²: 0.9012
  归一化MAE: 0.0987
```

## 目录自动管理

训练脚本会自动管理保存目录：
1. 在 `logs/` 下创建 `trained_tcn_配置名/` 目录
2. 自动检测已有的训练序号，创建新序号目录
3. 复制当前使用的配置文件到保存目录
4. 保存所有日志和模型检查点

例如：
```
logs/
└── trained_tcn_default_config/
    ├── 0/  # 第一次训练
    ├── 1/  # 第二次训练
    └── 2/  # 第三次训练
```

## 注意事项

1. **输入归一化**：训练前必须计算并设置 `center` 和 `scale` 参数
2. **延迟设置**：`model_delays` 参数需根据实际预测目标设置
3. **有效历史长度**：`eff_hist` 应与TCN架构匹配，计算公式：
   ```
   eff_hist = (ksize - 1) × (2^num_layers - 1)
   ```
4. **批次大小**：根据GPU内存调整 `batch_size`
5. **学习率调整**：如果训练不稳定，可以降低 `learning_rate`

## 模型检查点格式

保存的 `.tar` 文件包含：
- `state_dict`: 模型权重
- `epoch`: 训练轮数
- 所有模型超参数（input_size, output_size, num_channels等）
- `optimizer_state`: 优化器状态（可选）

## 故障排除

### 问题1：CUDA内存不足
- 减小 `batch_size`
- 减少 `num_channels` 中的通道数

### 问题2：训练损失不下降
- 检查 `center` 和 `scale` 是否正确计算
- 降低学习率
- 增加训练轮数

### 问题3：过拟合
- 增加 `dropout` 值
- 减少模型复杂度（减少层数或通道数）
- 增加训练数据

### 问题4：数据加载失败
- 检查数据目录结构是否正确
- 确认文件命名格式：`参与者_运动类型_exo.csv` 和 `参与者_运动类型_moment_filt.csv`
- 检查参与者体重是否在 `participant_masses` 中定义

## 引用

如果使用本代码，请引用原始论文：
```
Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation
```

## 许可证

本项目基于MIT许可证。TCN实现部分修改自 [CMU Locus Lab的TCN实现](https://github.com/locuslab/TCN)。

## 联系方式

如有问题或建议，请联系项目维护者或提交Issue。
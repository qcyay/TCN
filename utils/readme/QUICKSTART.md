# 快速开始指南

## 🎯 5分钟上手

### 步骤1: 检查数据结构

确保你的数据按以下方式组织：

```
data/
├── train/
│   └── BT01/
│       └── walk/
│           ├── BT01_walk_exo.csv          # 传感器数据
│           └── BT01_walk_moment_filt.csv  # 力矩真值
└── test/
    └── （相同结构）
```

### 步骤2: 选择模型

打开 `configs/default_config.py`，设置模型类型：

```python
# 使用Transformer（推荐用于快速训练）
model_type = 'Transformer'

# 或使用TCN（推荐用于实时预测）
model_type = 'TCN'
```

### 步骤3: 开始训练

```bash
# 使用GPU训练（推荐）
python train_transformer.py --device cuda

# 使用CPU训练
python train_transformer.py --device cpu
```

### 步骤4: 测试模型

```bash
python test_transformer.py \
    --model_path logs/trained_transformer_default_config/0/best_model.tar \
    --device cuda
```

## 📊 查看结果

训练结束后，在以下位置查看结果：

- **训练日志:** `logs/trained_*/*/train_log.txt`
- **验证结果:** `logs/trained_*/*/validation_log.txt`
- **最佳模型:** `logs/trained_*/*/best_model.tar`

## 🔧 常用命令

### 训练相关

```bash
# 基础训练
python train_transformer.py

# 指定配置文件
python train_transformer.py --config_path configs.my_config

# 使用4个数据加载进程
python train_transformer.py --num_workers 4

# 从检查点恢复
python train_transformer.py --resume path/to/checkpoint.tar
```

### 测试相关

```bash
# 基础测试
python test_transformer.py --model_path path/to/model.tar

# 指定批次大小
python test_transformer.py --model_path path/to/model.tar --batch_size 64
```

## ⚙️ 核心配置参数

### Transformer模型

```python
# configs/partial_motion_knee_config.py

model_type = 'Transformer'
sequence_length = 100    # 序列长度：50-200
d_model = 128            # 模型维度：64-256
nhead = 8                # 注意力头数：4-16
num_encoder_layers = 4   # Encoder层数：2-8
batch_size = 32          # 批次大小：16-64
learning_rate = 0.001    # 学习率：1e-4 to 1e-2
```

### TCN模型

```python
# configs/partial_motion_knee_config.py

model_type = 'TCN'
num_channels = [64, 64]  # 通道数
ksize = 3                # 卷积核大小：2-7
eff_hist = 248           # 有效历史
batch_size = 4           # 批次大小：2-8（变长序列）
learning_rate = 0.001
```

## 📈 性能优化建议

### 🚀 提升训练速度

1. **使用GPU**
   ```bash
   python train_transformer.py --device cuda
   ```

2. **增加数据加载进程**
   ```bash
   python train_transformer.py --num_workers 8
   ```

3. **使用Transformer而非TCN**
   - Transformer使用固定长度窗口，训练更快
   - TCN处理变长序列，训练较慢

4. **增加批次大小**（如果内存允许）
   ```python
   batch_size = 64  # Transformer
   ```

### 🎯 提升模型精度

1. **增加模型容量**
   ```python
   # Transformer
   d_model = 256
   num_encoder_layers = 6
   
   # TCN
   num_channels = [128, 128, 128]
   ```

2. **调整学习率调度**
   ```python
   scheduler_factor = 0.5      # 更激进的衰减
   scheduler_patience = 5      # 更早触发衰减
   ```

3. **增加训练轮数**
   ```python
   num_epochs = 2000
   ```

4. **使用更长的序列**（Transformer）
   ```python
   sequence_length = 200
   ```

### 💾 降低内存使用

1. **减小批次大小**
   ```python
   batch_size = 16  # Transformer
   batch_size = 2   # TCN
   ```

2. **减小模型大小**
   ```python
   # Transformer
   d_model = 64
   num_encoder_layers = 2
   
   # TCN
   num_channels = [32, 32]
   ```

3. **减小序列长度**（Transformer）
   ```python
   sequence_length = 50
   ```

## 🐛 问题排查

### ❌ 训练损失为NaN

**原因：** 数据中有NaN值或学习率过大

**解决：**
1. 检查 `center` 和 `scale` 参数是否正确填写
2. 降低学习率：`learning_rate = 0.0001`
3. 数据中的NaN会自动处理，但确保center/scale是基于干净数据计算的

### ❌ 内存不足 (CUDA OOM)

**解决：**
1. 减小 `batch_size`
2. 减小 `sequence_length`（Transformer）
3. 减小 `d_model` 或 `num_channels`
4. 使用更少的 `num_workers`

### ❌ 验证指标不提升

**解决：**
1. **过拟合：** 增加dropout、减小模型容量
2. **欠拟合：** 增加模型容量、训练更久
3. 调整学习率调度器参数

## 📝 结果解读

### 训练日志示例

```
Epoch [100/1000] - 训练损失: 0.002345, 学习率: 1.00e-03

=== Epoch 100 验证结果 ===
验证损失: 0.001987

hip_flexion_r_moment:
  RMSE: 0.0423 Nm/kg
  R²: 0.9234
  归一化MAE: 0.0156

knee_angle_r_moment:
  RMSE: 0.0387 Nm/kg
  R²: 0.9456
  归一化MAE: 0.0143
```

### 指标含义

- **RMSE < 0.05:** 优秀
- **R² > 0.90:** 优秀
- **归一化MAE < 0.02:** 优秀

## 🎓 下一步

1. **调整超参数：** 根据你的数据特点调整配置
2. **尝试两种模型：** 比较Transformer和TCN的性能
3. **数据增强：** 如果数据不足，考虑数据增强策略
4. **集成学习：** 训练多个模型并集成预测结果

## 🐛 已知问题和解决方案

### TCN序列长度问题

**问题**：TCN使用变长序列，不同batch的序列长度可能不同

**解决方案**：已在代码中修复，通过逐batch计算指标而非concatenate所有结果

**影响**：无，用户正常使用即可

详细说明请查看 `BUGFIX.md`

1. **首次训练：** 使用默认参数先跑一遍，了解基准性能
2. **快速实验：** 设置小的 `num_epochs` (如50) 快速测试不同配置
3. **保存最佳模型：** 系统会自动保存验证损失最低的模型为 `best_model.tar`
4. **监控训练：** 定期查看日志文件，及时发现问题
5. **GPU利用率：** 使用 `nvidia-smi` 监控GPU使用情况

## 📞 获取帮助

遇到问题？查看完整文档：`README_Transformer.md`

祝训练顺利！ 🎉
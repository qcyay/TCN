# 时序数据预测系统 - 完整文档

本项目实现了三种深度学习模型用于基于传感器数据预测人体关节力矩：
1. **TCN** - 时间卷积网络
2. **Transformer预测模型** - 基于Encoder的直接预测
3. **Transformer生成模型** - 基于Encoder-Decoder的自回归生成

## 📁 项目结构

```
project/
├── configs/
│   └── default_config.py          # 统一配置文件（支持所有模型）
├── models/
│   ├── tcn.py                      # TCN模型
│   ├── predictor_model.py          # Transformer预测模型
│   ├── generative_model.py         # Transformer生成模型
│   └── positional_encoding.py     # 位置编码
├── dataset_loaders/
│   ├── dataloader.py               # TCN数据加载器
│   └── sequence_dataloader.py     # Transformer数据加载器
├── utils/
│   └── config_utils.py             # 配置加载工具
├── data/
│   ├── train/                      # 训练数据
│   └── test/                       # 测试数据
├── logs/                           # 训练日志和模型保存
├── train.py                        # 统一训练脚本
└── test.py                         # 统一测试脚本
```

## 🚀 快速开始

### 1. 环境配置

```bash
pip install torch pandas numpy
```

### 2. 准备数据

数据结构：
```
data/
├── train/
│   └── 参与者ID/
│       └── 运动类型/
│           ├── 参与者ID_运动类型_exo.csv          # 传感器数据
│           └── 参与者ID_运动类型_moment_filt.csv  # 力矩真值
└── test/
    └── （相同结构）
```

### 3. 配置模型

编辑 `configs/default_config.py`：

```python
# 选择模型类型
model_type = 'GenerativeTransformer'  # 'TCN', 'Transformer', 或 'GenerativeTransformer'

# 根据模型类型配置相应参数...
```

### 4. 训练

```bash
# 基础训练
python train.py --device cuda

# 指定配置
python train.py --config_path configs.my_config --device cuda --num_workers 4

# 恢复训练
python train.py --resume logs/trained_*/*/model_epoch_100.tar
```

### 5. 测试

```bash
# 基础测试
python test.py --model_path logs/trained_*/*/best_model.tar --device cuda

# 生成式模型使用自回归生成
python test.py --model_path path/to/model.tar --use_generation
```

## 🔬 三种模型详解

### 1. TCN (Temporal Convolutional Network)

**架构特点：**
- 因果卷积，使用完整序列历史
- 膨胀卷积扩大感受野
- 残差连接

**优点：**
- 适合实时在线预测
- 内存效率高
- 使用完整上下文信息

**缺点：**
- 感受野受限
- 训练速度较慢
- 需要padding处理变长序列

**配置示例：**
```python
model_type = 'TCN'
num_channels = [64, 64]
ksize = 3
eff_hist = 248
batch_size = 4
```

**使用场景：**
- 实时预测系统
- 序列长度变化大
- 需要完整历史信息

---

### 2. Transformer预测模型

**架构特点：**
```
输入 [B,C,N] → 归一化 → 投影到d_model → 位置编码 
→ Transformer Encoder → 输出投影 → 预测 [B,output_size,N]
```

**优点：**
- 并行计算效率高
- 全局注意力机制
- 训练速度快

**缺点：**
- 固定长度窗口
- 内存占用较大

**配置示例：**
```python
model_type = 'Transformer'
sequence_length = 100
d_model = 128
nhead = 8
num_encoder_layers = 4
batch_size = 32
```

**使用场景：**
- 离线批量预测
- 数据量充足
- 需要快速训练

---

### 3. Transformer生成模型 ⭐ 新增

**架构特点：**
```
传感器数据 [B,C,N]:
  → 归一化 → 投影 → Encoder特征 [B,N,d_model]
                          ↓ (作为memory)
力矩数据 [B,2,N]:              ↓
  → Shifted → 投影 → Decoder → Cross-Attention
                          ↓
                    输出 [B,2,N]
```

**编码器选项：**
1. **Transformer Encoder** (`encoder_type='transformer'`)
   - 使用多层Transformer Encoder处理传感器数据
   - 捕捉输入序列的复杂时序关系
   - 参数更多，表达能力更强

2. **Linear Encoder** (`encoder_type='linear'`)
   - 使用简单的线性层处理传感器数据
   - 轻量级，训练更快
   - 适合输入特征相对简单的场景

**训练模式：**
- 使用Teacher Forcing
- 解码器输入：右移的真实力矩值
- 使用因果掩码防止看到未来信息

**测试模式：**
- 自回归生成
- 从起始token开始逐步预测
- 每步的预测作为下一步的输入

**配置示例：**
```python
model_type = 'GenerativeTransformer'
encoder_type = 'transformer'  # 或 'linear'
gen_sequence_length = 100
gen_d_model = 128
gen_nhead = 8
gen_num_encoder_layers = 3
gen_num_decoder_layers = 3
start_token_value = 0.0
teacher_forcing_ratio = 1.0
```

**使用场景：**
- 需要序列到序列建模
- 想要利用解码器的自回归特性
- 对生成质量有较高要求

**测试选项：**
```bash
# Teacher Forcing模式（快速，用于评估）
python test.py --model_path path/to/model.tar

# 自回归生成模式（真实场景）
python test.py --model_path path/to/model.tar --use_generation
```

---

## 📊 模型对比

| 特性 | TCN | Transformer预测 | Transformer生成 |
|------|-----|----------------|----------------|
| 架构 | 卷积 | Encoder | Encoder-Decoder |
| 序列长度 | 变长 | 固定窗口 | 固定窗口 |
| 计算模式 | 序列 | 并行 | 部分并行 |
| 训练速度 | 慢 | 快 | 中等 |
| 内存使用 | 低 | 中 | 高 |
| 批次大小 | 2-8 | 16-64 | 16-32 |
| 实时性 | 优秀 | 良好 | 良好 |
| 预测方式 | 直接 | 直接 | 自回归 |
| 参数量 | 中 | 中 | 大 |

## ⚙️ 配置参数详解

### 通用参数

```python
# 数据配置
data_dir = 'data'
side = "r"  # 'l' 或 'r'
model_delays = [10, 0]  # 每个输出的延迟

# 训练配置
num_epochs = 1000
batch_size = 32  # 根据模型调整
learning_rate = 0.001
weight_decay = 1e-5
random_seed = 42

# 归一化参数（需要从训练数据计算）
center = torch.tensor([...])
scale = torch.tensor([...])
```

### TCN专用参数

```python
num_channels = [64, 64]      # 各层通道数
ksize = 3                     # 卷积核大小
eff_hist = 248                # 有效历史长度
dropout = 0.2                 # Dropout比率
spatial_dropout = False       # 是否使用空间dropout
activation = 'ReLU'           # 激活函数
norm = 'weight_norm'          # 归一化方法
```

### Transformer预测模型参数

```python
sequence_length = 100         # 序列窗口长度
d_model = 128                 # 模型维度
nhead = 8                     # 注意力头数
num_encoder_layers = 4        # Encoder层数
dim_feedforward = 512         # FFN维度
transformer_dropout = 0.1     # Dropout比率
use_positional_encoding = True # 是否使用位置编码
```

### Transformer生成模型参数

```python
encoder_type = 'transformer'  # 编码器类型
gen_sequence_length = 100     # 序列长度
gen_d_model = 128             # 模型维度
gen_nhead = 8                 # 注意力头数
gen_num_encoder_layers = 3    # Encoder层数
gen_num_decoder_layers = 3    # Decoder层数
gen_dim_feedforward = 512     # FFN维度
gen_dropout = 0.1             # Dropout比率
start_token_value = 0.0       # 起始token值
teacher_forcing_ratio = 1.0   # Teacher forcing比率
```

## 🎯 超参数调优建议

### TCN调优

| 参数 | 推荐范围 | 影响 |
|------|---------|------|
| num_channels | [32,32] ~ [128,128] | 模型容量 |
| ksize | 3-7 | 感受野大小 |
| dropout | 0.1-0.3 | 过拟合控制 |
| batch_size | 2-8 | 内存/稳定性 |

### Transformer预测模型调优

| 参数 | 推荐范围 | 影响 |
|------|---------|------|
| sequence_length | 50-200 | 上下文长度 |
| d_model | 64-256 | 表达能力 |
| nhead | 4-16 | 注意力多样性 |
| num_encoder_layers | 2-8 | 模型深度 |
| batch_size | 16-64 | 训练效率 |

### Transformer生成模型调优

| 参数 | 推荐范围 | 影响 |
|------|---------|------|
| encoder_type | transformer/linear | 编码器复杂度 |
| gen_num_encoder_layers | 2-6 | 编码深度 |
| gen_num_decoder_layers | 2-6 | 解码深度 |
| gen_d_model | 64-256 | 模型容量 |
| batch_size | 16-32 | 内存平衡 |

## 📈 训练技巧

### 1. 数据预处理

```python
# 计算归一化参数（从训练数据）
# 确保center和scale是基于干净数据（无NaN）计算的
center = train_data.mean(axis=0, keepdims=True)
scale = train_data.std(axis=0, keepdims=True)
```

### 2. 学习率调度

```python
# 配置ReduceLROnPlateau
scheduler_factor = 0.9        # 衰减因子
scheduler_patience = 10       # 耐心值
min_lr = 1e-6                 # 最小学习率
```

### 3. 早停策略

```python
early_stopping_patience = 50  # 耐心值
early_stopping_min_delta = 0.0 # 最小改善
```

### 4. 梯度裁剪

```python
grad_clip = 1.0  # 防止梯度爆炸
```

## 🐛 常见问题

### 1. 训练损失为NaN

**原因：**
- 数据中有NaN值
- 学习率过大
- center/scale参数错误

**解决：**
```python
# 确保center和scale正确
# 降低学习率
learning_rate = 0.0001
# NaN会自动过滤，但确保归一化参数正确
```

### 2. 内存不足

**解决：**
```python
# 减小batch_size
batch_size = 16  # 或更小

# 减小序列长度（Transformer）
sequence_length = 50

# 减小模型大小
d_model = 64
num_encoder_layers = 2
```

### 3. 生成模型预测质量差

**可能原因：**
- Teacher forcing ratio设置不当
- 起始token选择不合理
- 解码器层数不足

**解决：**
```python
# 调整teacher forcing
teacher_forcing_ratio = 0.8  # 逐渐降低

# 选择合适的起始token
start_token_value = 0.0  # 或数据的均值

# 增加解码器深度
gen_num_decoder_layers = 4
```

### 4. 模型选择建议

**选择TCN：**
- ✅ 需要实时预测
- ✅ 序列长度变化大
- ✅ 内存受限

**选择Transformer预测：**
- ✅ 离线批量处理
- ✅ 数据量充足
- ✅ 需要快速训练

**选择Transformer生成：**
- ✅ 需要序列建模
- ✅ 对生成质量要求高
- ✅ 想探索自回归方法

## 📝 性能指标

### 评估指标

1. **RMSE (Root Mean Square Error)**
   - 单位：Nm/kg
   - 越小越好
   - < 0.05 为优秀

2. **R² (R-squared)**
   - 范围：-∞ to 1
   - 越接近1越好
   - \> 0.90 为优秀

3. **归一化MAE (Normalized Mean Absolute Error)**
   - 范围：0 to 1
   - 越小越好
   - < 0.02 为优秀

## 🔬 进阶使用

### 自定义编码器类型

```python
# 在generative_model.py中
# 尝试不同的编码器架构

# 1. 纯线性编码器（最快）
encoder_type = 'linear'

# 2. Transformer编码器（最强）
encoder_type = 'transformer'
gen_num_encoder_layers = 4
```

### 混合训练策略

```python
# 可以尝试课程学习
# 开始时使用teacher_forcing_ratio=1.0
# 逐渐降低到0.5
```

### 集成学习

```python
# 训练多个不同配置的模型
# 使用加权平均进行预测
predictions = 0.4 * model1_pred + 0.3 * model2_pred + 0.3 * model3_pred
```

## 📚 引用

如果使用本项目，请引用：
- **Transformer:** "Attention Is All You Need" (Vaswani et al., 2017)
- **TCN:** "An Empirical Evaluation of Generic Convolutional and Recurrent Networks" (Bai et al., 2018)
- **应用背景:** "Task-Agnostic Exoskeleton Control via Biological Joint Moment Estimation"

## 📧 联系方式

如有问题或建议，请提交Issue。

## 📄 许可证

MIT License
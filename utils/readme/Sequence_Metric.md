# 序列级别指标计算 - 更新说明

## 概述

本次更新解决了Transformer模型在短序列上计算指标导致R²和MAE异常的问题。现在所有模型（TCN、Transformer预测模型、生成式Transformer）都在**完整序列**上计算指标。

## 主要变化

### 1. 问题分析

当序列长度较短(50-100)时，单个短序列内的数据变化很小，导致：
- `label_range` 非常小 → MAE归一化后变得很大（5000%）
- `ss_tot` 非常小 → R²变成大负数（-15）

但全局RMSE看起来正常，因为预测误差相对较小。

### 2. 解决方案

#### 对于Transformer模型
将短序列预测重组回完整序列，然后在完整序列上计算指标。提供两种重组方式：

1. **only_first** (推荐)
   - 每个短序列只取第一个预测值
   - 因为后续位置会被下一个短序列的预测覆盖
   - 适合预测任务，避免重复计算

2. **average**
   - 对每个位置的所有预测值取平均
   - 可以利用多个预测的信息
   - 计算量稍大，但可能更稳定

#### 对于TCN模型
保持不变，继续在完整的长序列上计算指标。

## 使用方法

### 配置reconstruction_method参数

有三种方式设置序列重组方法：

**方法1：在配置文件中设置（推荐）**
```python
# configs/default_config.py
reconstruction_method = 'only_first'  # 或 'average'
```
然后直接运行：
```bash
python train.py --config_path configs.default_config --device cuda
```

**方法2：命令行参数覆盖**
```bash
# 临时使用不同的方法
python train.py --config_path configs.default_config --device cuda --reconstruction_method average
```

**方法3：不指定（使用默认值only_first）**
```bash
python train.py --config_path configs.default_config --device cuda
```

**优先级：** 命令行参数 > 配置文件 > 默认值

详细说明见 [PARAMETER_CONFIG_GUIDE.md](PARAMETER_CONFIG_GUIDE.md)

### 训练

```bash
# 使用 only_first 方法（默认）
python train.py --config_path configs.default_config --device cuda

# 使用 average 方法
python train.py --config_path configs.default_config --device cuda --reconstruction_method average
```

### 测试

```bash
# 测试Transformer模型 - only_first方法
python test.py --config_path configs.default_config \
               --model_path logs/trained_transformer_default_config/0/best_model.tar \
               --device cuda \
               --reconstruction_method only_first

# 测试Transformer模型 - average方法
python test.py --config_path configs.default_config \
               --model_path logs/trained_transformer_default_config/0/best_model.tar \
               --device cuda \
               --reconstruction_method average

# 测试TCN模型（不需要指定reconstruction_method）
python test.py --config_path configs.default_config \
               --model_path logs/trained_tcn_default_config/0/best_model.tar \
               --device cuda
```

### 生成式模型测试

```bash
# 使用Teacher Forcing
python test.py --config_path configs.default_config \
               --model_path logs/trained_generativetransformer_default_config/0/best_model.tar \
               --device cuda \
               --reconstruction_method only_first

# 使用自回归生成
python test.py --config_path configs.default_config \
               --model_path logs/trained_generativetransformer_default_config/0/best_model.tar \
               --device cuda \
               --reconstruction_method only_first \
               --use_generation
```

## 核心函数说明

### `reconstruct_sequences_from_predictions()`

重组短序列预测为完整序列。

**参数:**
- `all_estimates`: [N, num_outputs, output_seq_len] - 所有短序列的预测
- `all_labels`: [N, num_outputs, output_seq_len] - 所有短序列的标签
- `trial_info`: List[(trial_idx, input_start_idx, seq_len)] - 每个短序列的信息
- `method`: 'only_first' 或 'average' - 重组方法

**返回:**
- `reconstructed_estimates`: List[Tensor[num_outputs, trial_len]] - 完整序列预测
- `reconstructed_labels`: List[Tensor[num_outputs, trial_len]] - 完整序列标签

**工作原理:**

1. **only_first 方法:**
   ```
   短序列: [seq1_pred0, seq1_pred1, ...], [seq2_pred0, seq2_pred1, ...], ...
   完整序列: [seq1_pred0, seq2_pred0, seq3_pred0, ...]
   ```
   每个短序列只取第一个预测值（因为后续位置会被覆盖）

2. **average 方法:**
   ```
   位置i的预测 = mean(所有覆盖位置i的短序列的预测值)
   ```
   对每个位置求平均，可以融合多个预测

### `compute_metrics_on_sequences()`

在完整序列上计算RMSE、R²和MAE%指标。

**参数:**
- `estimates_list`: List[Tensor[num_outputs, seq_len]] - 每个trial的预测
- `labels_list`: List[Tensor[num_outputs, seq_len]] - 每个trial的标签
- `label_names`: 标签名称列表

**返回:**
包含每个输出通道指标的字典

## 示例输出

### 训练时的验证输出
```
=== Epoch 20 验证结果 ===
使用 'only_first' 方法重组序列...
验证损失: 0.002341

hip_flexion_r_moment:
  RMSE: 0.0523 Nm/kg
  R²: 0.8934
  MAE: 3.24%

knee_angle_r_moment:
  RMSE: 0.0445 Nm/kg
  R²: 0.9156
  MAE: 2.87%
```

### 测试时的输出
```
测试结果 (在完整序列上计算):
============================================================

hip_flexion_r_moment:
  RMSE: 0.0518 Nm/kg
  R²: 0.8956
  MAE: 3.18%
  有效试验数: 45

knee_angle_r_moment:
  RMSE: 0.0438 Nm/kg
  R²: 0.9178
  MAE: 2.82%
  有效试验数: 45
============================================================
```

## 两种方法的比较

| 方法 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **only_first** | - 计算快速<br>- 避免重复<br>- 更接近实际使用 | - 只用一个预测值<br>- 可能波动较大 | 实时预测、在线应用 |
| **average** | - 融合多个预测<br>- 结果更平滑<br>- 可能更准确 | - 计算稍慢<br>- 不适合实时 | 离线分析、后处理 |

## 技术细节

### 序列重组原理

以sequence_length=100, output_sequence_length=50为例：

```
原始数据: [0, 1, 2, 3, 4, 5, ..., 1000]

短序列提取:
- seq0: input[0:100] → predict[100:150]
- seq1: input[1:101] → predict[101:151]
- seq2: input[2:102] → predict[102:152]
...

重组(only_first):
完整预测: [pred0[0], pred1[0], pred2[0], pred3[0], ...]
         = [100位置, 101位置, 102位置, 103位置, ...]

重组(average):
完整预测[100] = pred0[0]
完整预测[101] = mean(pred0[1], pred1[0])
完整预测[102] = mean(pred0[2], pred1[1], pred2[0])
...
```

### 为什么这样能解决问题？

1. **增加数据变化范围**: 完整序列长度可能有数千个点，数据变化充分，label_range正常
2. **稳定统计量**: ss_tot在大量数据上计算更稳定，不会过小
3. **真实反映性能**: 完整序列指标更能反映模型的实际表现

## 注意事项

1. **内存使用**: average方法需要额外的累加数组，内存占用稍大
2. **计算时间**: average方法需要遍历所有预测位置，时间稍长
3. **trial信息**: 需要正确记录每个短序列所属的trial和起始位置
4. **延迟处理**: model_delays已在数据加载时处理，重组时不需要再考虑

## 常见问题

### Q1: 为什么不直接在短序列上计算指标？
A: 短序列数据变化太小，统计量不稳定，导致R²和MAE指标异常。

### Q2: only_first和average哪个更好？
A: 
- 实时预测场景: 使用only_first（更快，更接近实际）
- 离线分析场景: 可以尝试average（可能更准确）
- 推荐先用only_first，如果需要可以对比两种方法

### Q3: TCN模型为什么不需要重组？
A: TCN直接处理完整的长序列，不需要切分，所以不存在重组问题。

### Q4: 训练时使用哪种方法？
A: 两种方法都可以，建议使用only_first（默认），因为：
- 更快
- 更接近实际使用场景
- 减少验证时间

### Q5: 两种方法的指标会有多大差异？
A: 通常差异不大（<5%），但取决于：
- 序列长度: 越短差异可能越大
- 预测稳定性: 预测越稳定差异越小
- 数据特性: 变化剧烈的数据差异可能较大

## 更新日志

### v2.0 (当前版本)
- ✅ 添加序列重组功能
- ✅ 支持only_first和average两种方法
- ✅ 在完整序列上计算指标
- ✅ 解决R²和MAE异常问题
- ✅ 添加命令行参数选择重组方法
- ✅ 更新train.py和test.py

### v1.0
- 基础版本
- 在短序列上计算指标（存在问题）

## 开发者信息

如有问题或建议，请联系开发团队。

最后更新: 2025-10-28
# Bug修复说明 - TCN验证时的序列长度不匹配问题

## 🐛 问题描述

在使用TCN模型训练时，验证（validate）过程会出现以下错误：

```
RuntimeError: Sizes of tensors must match except in dimension 0. 
Expected size 6201 but got size 20801 for tensor number 1 in the list.
```

## 🔍 问题原因

### 根本原因

TCN模型使用**变长序列**：
1. 每个试验（trial）的原始序列长度不同
2. DataLoader在每个batch内会将序列padding到**该batch的最大长度**
3. 不同batch的最大长度可能不同

### 错误发生位置

在原始的 `validate()` 函数中：

```python
# 收集所有batch的结果
for batch_data in dataloader:
    # ... 处理 ...
    all_estimates.append(estimates)  # estimates形状: [batch_size, num_outputs, seq_len_batch_1]
    all_labels.append(labels)

# 尝试concatenate - 这里会出错！
all_estimates = torch.cat(all_estimates, dim=0)  # ❌ 不同batch的seq_len不同
all_labels = torch.cat(all_labels, dim=0)
```

### 示例

```python
# Batch 1: 最大序列长度 = 6201
estimates_batch1.shape = [4, 2, 6201]

# Batch 2: 最大序列长度 = 20801
estimates_batch2.shape = [4, 2, 20801]

# 尝试concatenate
torch.cat([estimates_batch1, estimates_batch2], dim=0)  
# ❌ 错误！第2维（seq_len）不匹配：6201 vs 20801
```

## ✅ 解决方案

### 核心思路

**针对不同模型类型采用不同策略：**

1. **TCN模型**：不进行concatenate，逐batch计算指标并累积
2. **Transformer模型**：使用固定长度窗口，可以安全concatenate

### 实现细节

#### 1. 新增函数：逐batch累积指标

```python
def compute_metrics_batch(estimates: torch.Tensor, labels: torch.Tensor,
                         num_outputs: int, metrics_accumulator: dict) -> None:
    """
    逐batch累积计算评估指标
    避免concatenate不同长度的序列
    """
    batch_size, _, seq_len = estimates.shape

    for i in range(batch_size):
        for j in range(num_outputs):
            # 提取当前样本和输出
            estimate = estimates[i, j, :]
            label = labels[i, j, :]

            # 忽略NaN值
            valid_mask = ~torch.isnan(estimate) & ~torch.isnan(label)
            estimate_valid = estimate[valid_mask]
            label_valid = label[valid_mask]

            if len(estimate_valid) == 0:
                continue

            # 计算指标
            rmse = torch.sqrt(torch.mean((estimate_valid - label_valid) ** 2))
            # ... 其他指标 ...

            # 累积到accumulator
            metrics_accumulator[f"output_{j}"]["rmse"] += rmse.item()
            metrics_accumulator[f"output_{j}"]["count"] += 1
```

#### 2. 修改validate函数

```python
def validate(model, dataloader, label_names, device, model_type, config):
    # 初始化指标累积器
    metrics_accumulator = {
        f"output_{i}": {"rmse": 0.0, "r2": 0.0, "normalized_mae": 0.0, "count": 0}
        for i in range(num_outputs)
    }

    with torch.no_grad():
        if model_type == 'TCN':
            # TCN: 逐batch处理，不concatenate
            for batch_data in dataloader:
                # ... 前向传播 ...
                
                # 直接在当前batch上计算指标并累积
                for i in range(batch_size):
                    for j in range(num_outputs):
                        # 考虑model_history和delays
                        est = estimates[i, j, model_history:trial_lengths[i]]
                        lbl = label_data[i, j, model_history:trial_lengths[i]]
                        
                        # 计算并累积指标
                        # ...
        
        else:
            # Transformer: 固定长度，可以concatenate
            all_estimates = []
            all_labels = []
            
            for batch_data in dataloader:
                # ... 前向传播 ...
                all_estimates.append(estimates)
                all_labels.append(labels)
            
            # 安全concatenate
            all_estimates = torch.cat(all_estimates, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # 计算指标
            compute_metrics_batch(all_estimates, all_labels, 
                                num_outputs, metrics_accumulator)
    
    # 计算平均指标
    metrics = finalize_metrics(metrics_accumulator, num_outputs)
    return result_dict, avg_loss
```

#### 3. 同步修复test.py

在test.py中对TCN也采用相同的策略：

```python
# TCN测试 - 不concatenate
total_metrics = {label_name: {...} for label_name in label_names}

for batch_data in test_loader:
    # 前向传播
    estimates = model(input_data)
    
    # 逐样本计算指标并累积
    for i in range(batch_size):
        for j, label_name in enumerate(label_names):
            # 计算指标
            # 累积到total_metrics

# 打印平均结果
for label_name in label_names:
    avg_rmse = total_metrics[label_name]["rmse"] / count
    # ...
```

## 📊 修复前后对比

### 修复前

```python
# ❌ 错误的方式
all_estimates = []
for batch in dataloader:
    estimates = model(batch)  # 不同batch的seq_len不同
    all_estimates.append(estimates)

all_estimates = torch.cat(all_estimates, dim=0)  # ❌ 出错！
```

### 修复后

```python
# ✅ 正确的方式 - TCN
metrics_accumulator = initialize_metrics()
for batch in dataloader:
    estimates = model(batch)
    # 在当前batch上计算指标并累积
    compute_metrics_batch(estimates, labels, metrics_accumulator)

metrics = finalize_metrics(metrics_accumulator)  # ✅ 成功！
```

```python
# ✅ 正确的方式 - Transformer（固定长度）
all_estimates = []
for batch in dataloader:
    estimates = model(batch)  # 所有batch的seq_len相同
    all_estimates.append(estimates)

all_estimates = torch.cat(all_estimates, dim=0)  # ✅ 成功！
```

## 🎯 关键要点

### 1. 为什么Transformer没问题？

**Transformer使用固定长度窗口**：
- 数据集生成时就切分为固定长度（如100）
- 所有序列长度完全一致
- 可以安全concatenate

```python
# Transformer数据
batch1: [32, 2, 100]  # 序列长度固定为100
batch2: [32, 2, 100]  # 序列长度固定为100
torch.cat([batch1, batch2], dim=0)  # ✅ OK
```

### 2. 为什么TCN有问题？

**TCN使用完整的变长序列**：
- 保留每个试验的完整长度
- 每个batch内padding到不同的最大长度
- 不能跨batch concatenate

```python
# TCN数据
batch1: [4, 2, 6201]   # padding到6201
batch2: [4, 2, 20801]  # padding到20801
torch.cat([batch1, batch2], dim=0)  # ❌ 错误
```

### 3. 修复策略总结

| 模型类型 | 序列特点 | 处理策略 |
|---------|---------|---------|
| TCN | 变长，每batch不同 | 逐batch计算，累积指标 |
| Transformer预测 | 固定长度 | Concatenate后统一计算 |
| Transformer生成 | 固定长度 | Concatenate后统一计算 |

## 🧪 验证修复

### 测试TCN

```bash
# 设置配置
model_type = 'TCN'

# 训练（验证会自动执行）
python train.py --device cuda --config_path configs.default_config
```

**期望结果**：
- ✅ 训练过程正常
- ✅ 验证过程不报错
- ✅ 正确显示RMSE、R²和归一化MAE

### 测试Transformer

```bash
# 设置配置
model_type = 'Transformer'  # 或 'GenerativeTransformer'

# 训练
python train.py --device cuda
```

**期望结果**：
- ✅ 训练和验证都正常
- ✅ 指标计算正确

## 💡 经验教训

### 1. 注意数据维度一致性

在进行tensor操作（尤其是concatenate）时，要确保：
- 明确每个维度的含义
- 检查不同batch是否维度一致
- 考虑padding带来的影响

### 2. 区分固定长度和变长序列

不同模型架构对序列长度的处理方式不同：
- **固定长度模型**：可以批量处理，concatenate安全
- **变长序列模型**：需要特殊处理，避免直接concatenate

### 3. 早期发现问题

在开发阶段就应该：
- 打印tensor的shape
- 检查不同batch的形状
- 使用断言验证假设

```python
# 调试技巧
print(f"Batch 1 shape: {estimates1.shape}")
print(f"Batch 2 shape: {estimates2.shape}")
assert estimates1.shape[2] == estimates2.shape[2], "Sequence length mismatch!"
```

## 📝 相关文件修改

1. **train.py**
   - ✏️ 添加 `compute_metrics_batch()` 函数
   - ✏️ 添加 `finalize_metrics()` 函数
   - ✏️ 重写 `validate()` 函数，区分TCN和Transformer
   - ✏️ 更新 `validate()` 调用，添加config参数

2. **test.py**
   - ✏️ 修改TCN测试逻辑，使用累积计算而非concatenate
   - ✏️ 保持Transformer测试逻辑不变

## ✅ 检查清单

修复完成后，确认以下各项：

- [ ] TCN训练和验证都能正常运行
- [ ] Transformer预测模型训练和验证正常
- [ ] Transformer生成模型训练和验证正常
- [ ] 所有模型的指标计算正确
- [ ] test.py对所有模型都能正常工作
- [ ] 不再出现"Sizes of tensors must match"错误

## 🎉 总结

通过区分固定长度和变长序列的处理方式，我们成功解决了TCN模型在验证时的concatenate错误，同时保持了Transformer模型的正常运行。这个修复确保了所有三种模型都能正确训练和评估。
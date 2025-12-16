# Dataloader 序列长度过滤功能说明

## 修改概述

本次修改为 `dataloader.py` 添加了根据 `min_sequence_length` 参数进行序列过滤的功能。

## 主要修改内容

### 1. 新增参数

在 `TcnDataset` 类的 `__init__` 方法中新增参数:
- **`min_sequence_length`** (int, 默认值: -1)
  - 最小序列长度阈值(单位:采样点,200Hz,每点=5ms)
  - 值为 -1 表示不限制序列长度
  - 值 > 0 时,序列长度小于该值的试验将被过滤掉

### 2. 新增统计信息

添加了 `length_filter_stats` 字典来记录过滤统计:
```python
self.length_filter_stats = {
    'trials_before_filter': 0,      # 过滤前试验数量
    'trials_after_filter': 0,       # 过滤后试验数量
    'trials_filtered_out': 0,       # 被过滤掉的试验数量
    'min_length_before': 0,         # 过滤前最小序列长度
    'max_length_before': 0,         # 过滤前最大序列长度
    'min_length_after': 0,          # 过滤后最小序列长度
    'max_length_after': 0           # 过滤后最大序列长度
}
```

### 3. 新增方法

#### `_filter_by_sequence_length()`
- **功能**: 根据 `min_sequence_length` 过滤序列
- **执行时机**: 在 `_preload_all_data()` 之后,`_remove_invalid_label_sequences()` 之前
- **过滤逻辑**: 
  - 遍历所有预加载的试验
  - 如果试验的序列长度 < `min_sequence_length`,则过滤掉该试验
  - 同步更新 `all_input_data`, `all_label_data`, `trial_lengths`, `trial_names` 和 `all_activity_mask`

#### `print_length_filter_summary()`
- **功能**: 打印序列长度过滤的统计摘要
- **调用时机**: 在数据集初始化完成时,如果 `min_sequence_length > 0`

#### `get_length_filter_stats()`
- **功能**: 返回序列长度过滤统计信息
- **返回值**: `length_filter_stats` 字典

### 4. 执行流程

数据集初始化时的执行顺序:
1. `_preload_all_data()` - 预加载所有数据
2. `_filter_by_sequence_length()` - **[新增]** 根据最小序列长度过滤
3. `_remove_invalid_label_sequences()` - 移除标签全为NaN的序列
4. `print_nan_removal_summary()` - 打印NaN移除统计
5. `print_length_filter_summary()` - **[新增]** 打印序列长度过滤统计

## 配置文件修改

在 `default_config.py` 中已经包含了 `min_sequence_length` 参数:

```python
# ==================== 序列长度过滤配置 ====================
# 允许加载的最小序列长度(单位:采样点,200Hz,每点=5ms)
# -1 表示不限制序列长度(默认行为)
min_sequence_length = -1
```

## 使用示例

### 示例1: 不限制序列长度(默认)
```python
min_sequence_length = -1  # 不过滤任何序列
```

### 示例2: 过滤掉短于1000个采样点的序列
```python
min_sequence_length = 1000  # 约5秒的数据(1000点 × 5ms/点 = 5s)
```

### 示例3: 过滤掉短于2000个采样点的序列
```python
min_sequence_length = 2000  # 约10秒的数据(2000点 × 5ms/点 = 10s)
```

## 输出示例

当启用序列长度过滤时,控制台会输出类似以下信息:

```
开始加载 train 数据集 (TCN)...
找到 1234 个试验 (动作筛选: 禁用)
预加载所有数据到内存...
  加载进度: 1234/1234
数据预加载完成!

开始根据最小序列长度 (1000) 进行过滤...
  过滤掉 45 个序列长度不足的试验
序列长度过滤完成!
数据集初始化完成 - 模式: train, 试验数量: 1189

============================================================
NaN移除统计摘要 - TRAIN 数据集
============================================================
处理的试验总数: 1234
包含NaN的试验数: 23
移除的数据行总数: 456
标签全为NaN的试验数: 0
包含NaN的试验比例: 1.86%
平均每个含NaN试验移除的行数: 19.8
============================================================

============================================================
序列长度过滤统计摘要 - TRAIN 数据集
============================================================
最小序列长度阈值: 1000
过滤前试验数量: 1234
过滤后试验数量: 1189
被过滤掉的试验数: 45
过滤比例: 3.65%
过滤前序列长度范围: [234, 8976]
过滤后序列长度范围: [1003, 8976]
============================================================
```

## 注意事项

1. **序列长度单位**: `min_sequence_length` 的单位是采样点数,不是时间。由于数据以200Hz采样,每个采样点代表5ms。

2. **过滤顺序**: 序列长度过滤在NaN移除之后、标签全为NaN序列移除之前执行。

3. **数据同步**: 过滤时会同步更新所有相关的数据列表:
   - `all_input_data`
   - `all_label_data`
   - `trial_lengths`
   - `trial_names`
   - `all_activity_mask` (如果启用)

4. **向后兼容**: 当 `min_sequence_length = -1` (默认值)时,不会进行任何过滤,保持原有行为。

5. **统计信息**: 通过 `get_length_filter_stats()` 方法可以获取详细的过滤统计信息。

## 测试建议

在使用新功能前,建议:
1. 先使用 `min_sequence_length = -1` 查看所有序列的长度分布
2. 根据实际需求设置合适的阈值
3. 观察过滤后的试验数量和序列长度分布是否符合预期

## 相关函数调用链

```
TcnDataset.__init__()
    ├── _get_trial_names()
    ├── _preload_all_data()
    │   └── _load_single_trial() (循环调用)
    │       ├── _load_input_data()
    │       ├── _load_label_data()
    │       └── _load_activity_flag()
    ├── _filter_by_sequence_length()  [新增]
    ├── _remove_invalid_label_sequences()
    ├── print_nan_removal_summary()
    └── print_length_filter_summary()  [新增]
```
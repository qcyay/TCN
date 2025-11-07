# 🎮 GPU 选择快速参考

## 最简使用方式

直接用数字指定GPU，超级简单！

```bash
# 使用 GPU 0
python train.py --device 0

# 使用 GPU 1
python train.py --device 1

# 使用 GPU 2
python train.py --device 2

# 使用 GPU 2
python train.py --device 3
```

## 常用命令

### 训练
```bash
# 查看GPU状态
nvidia-smi

# 选择空闲的GPU训练
python train.py --config_path configs.default_config --device 2
```

### 测试
```bash
python test.py --config_path configs.default_config \
               --model_path logs/model.tar \
               --device 1
```

### 并行训练（4块GPU同时工作）
```bash
python train.py --config configs.exp1 --device 0 &
python train.py --config configs.exp2 --device 1 &
python train.py --config configs.exp3 --device 2 &
python train.py --config configs.exp4 --device 3 &
```

## 完整参数列表

| 参数 | 说明 | 示例 |
|------|------|------|
| `--device cpu` | CPU | `python train.py --device cpu` |
| `--device cuda` | GPU 0 | `python train.py --device cuda` |
| `--device 0` | GPU 0 | `python train.py --device 0` ⭐ |
| `--device 1` | GPU 1 | `python train.py --device 1` ⭐ |
| `--device 2` | GPU 2 | `python train.py --device 2` ⭐ |
| `--device 3` | GPU 3 | `python train.py --device 3` ⭐ |

⭐ = 推荐格式

## 启动显示

```
可用GPU数量: 4
  GPU 0: NVIDIA GeForce RTX 3090 (24.0 GB)
  GPU 1: NVIDIA GeForce RTX 3090 (24.0 GB)
  GPU 2: NVIDIA GeForce RTX 3080 (10.0 GB)
  GPU 3: NVIDIA GeForce RTX 3080 (10.0 GB)

使用设备: GPU 2 - NVIDIA GeForce RTX 3080
```

## 实用技巧

### 1. 批量训练脚本
```bash
#!/bin/bash
# 保存为 train_all.sh

python train.py --config configs.exp1 --device 0 &
python train.py --config configs.exp2 --device 1 &
python train.py --config configs.exp3 --device 2 &
python train.py --config configs.exp4 --device 3 &
wait

echo "所有训练完成！"
```

运行：
```bash
chmod +x train_all.sh
./train_all.sh
```

### 2. 查看GPU使用
```bash
# 简单查看
nvidia-smi

# 持续监控（推荐）
pip install gpustat
gpustat -i 2  # 每2秒刷新
```

### 3. tmux多任务
```bash
# 启动4个会话
tmux new -s gpu0 -d "python train.py --device 0"
tmux new -s gpu1 -d "python train.py --device 1"
tmux new -s gpu2 -d "python train.py --device 2"
tmux new -s gpu3 -d "python train.py --device 3"

# 查看
tmux ls

# 连接
tmux attach -t gpu0
```

## 常见场景

### 场景1: 单任务训练
```bash
nvidia-smi  # 看哪块GPU空闲
python train.py --device 1  # 用空闲的
```

### 场景2: 4任务并行
```bash
python train.py --config configs.exp1 --device 0 &
python train.py --config configs.exp2 --device 1 &
python train.py --config configs.exp3 --device 2 &
python train.py --config configs.exp4 --device 3 &
```

### 场景3: 训练+测试
```bash
python train.py --device 0 &  # GPU 0 训练
python test.py --model_path logs/model.tar --device 1 &  # GPU 1 测试
```

## 错误处理

### GPU不存在
```bash
python train.py --device 5
# 警告: GPU 5 不存在，使用 GPU 0
```

### 无GPU时
```bash
python train.py --device 0
# 警告: CUDA不可用，回退到CPU
```

## 性能对比

**单GPU串行 vs 4GPU并行**

| 场景 | 单GPU | 4GPU | 提升 |
|------|-------|------|------|
| 4个模型 | 8小时 | 2小时 | 4x ⚡ |
| 8个模型 | 16小时 | 4小时 | 4x ⚡ |

## 记住这个

```bash
# 最简单的方式 - 直接用数字！
python train.py --device 0  # GPU 0
python train.py --device 1  # GPU 1
python train.py --device 2  # GPU 2
python train.py --device 3  # GPU 2
```

**就是这么简单！** 🎯

---

详细文档：
- [GPU_UPDATE.md](computer:///mnt/user-data/outputs/GPU_UPDATE.md) - 功能介绍
- [GPU_SELECTION_GUIDE.md](computer:///mnt/user-data/outputs/GPU_SELECTION_GUIDE.md) - 完整指南
"""
计算训练数据的归一化参数（center和scale）

使用方法:
    # 从项目根目录执行
    python compute_normalization_params.py --config_path configs.TCN.default_config

    # 或从utils目录执行
    cd utils
    python compute_normalization_params.py --config_path configs.TCN.default_config

输出:
    将输出center和scale的PyTorch张量值，可以直接复制到配置文件中
"""

import sys
import os

# 添加项目根目录到Python路径
# 这样无论从哪里执行脚本都能正确导入模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = parent_dir if os.path.basename(current_dir) == 'utils' else current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"项目根目录: {project_root}")
print(f"当前工作目录: {os.getcwd()}")

# 修改工作目录到项目根目录（重要！用于正确加载数据）
if os.getcwd() != project_root:
    os.chdir(project_root)
    print(f"已切换工作目录到: {os.getcwd()}\n")

import argparse
import torch
from utils.config_utils import load_config
from dataset_loaders.dataloader import TcnDataset
import numpy as np

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs.TCN.default_config",
                    help="配置文件路径")
parser.add_argument("--device", type=str, default="cpu",
                    help="计算设备")
args = parser.parse_args()

# 加载配置
config = load_config(args.config_path)


def compute_normalization_params():
    """计算训练集的归一化参数"""

    device = torch.device(args.device)
    print(f"使用设备: {device}")
    print(f"配置文件: {args.config_path}\n")

    # 准备输入和标签名称
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    print(f"输入特征数量: {len(input_names)}")
    print(f"输出标签数量: {len(label_names)}\n")

    # 创建训练数据集
    print("加载训练数据集...")
    train_dataset = TcnDataset(
        data_dir=config.data_dir,
        input_names=input_names,
        label_names=label_names,
        side=config.side,
        participant_masses=config.participant_masses,
        device=device,
        mode='train'
    )

    print(f"训练集包含 {len(train_dataset)} 个trials\n")

    # 收集所有训练数据
    print("收集所有训练数据...")
    all_inputs = []

    for i in range(len(train_dataset)):
        input_data, _, trial_lengths = train_dataset[i]
        # 只取有效数据部分（不包括zero padding）
        valid_data = input_data[:, :, :trial_lengths[0]]
        all_inputs.append(valid_data)

        if (i + 1) % 10 == 0 or (i + 1) == len(train_dataset):
            print(f"  已处理: {i+1}/{len(train_dataset)} trials")

    # 合并所有数据: [num_trials, num_features, variable_lengths] -> [1, num_features, total_length]
    print("\n合并数据...")
    all_inputs = torch.cat(all_inputs, dim=2)

    print(f"总数据形状: {all_inputs.shape}")
    print(f"  批次维度: {all_inputs.shape[0]}")
    print(f"  特征维度: {all_inputs.shape[1]}")
    print(f"  时间维度: {all_inputs.shape[2]}\n")

    # 计算每个特征的均值和标准差
    print("计算归一化参数...")

    # breakpoint()
    # 在批次和时间维度上计算统计量
    center = torch.mean(all_inputs, dim=(0, 2), keepdim=True)  # [1, num_features, 1]
    scale = torch.std(all_inputs, dim=(0, 2), keepdim=True)    # [1, num_features, 1]

    # 避免除以零
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))

    print(f"Center形状: {center.shape}")
    print(f"Scale形状: {scale.shape}\n")

    # 输出详细统计信息
    print("=" * 70)
    print("每个特征的统计信息:")
    print("=" * 70)

    for i, feature_name in enumerate(input_names):
        print(f"{i:2d}. {feature_name:40s} | "
              f"Mean: {center[0, i, 0].item():8.4f} | "
              f"Std: {scale[0, i, 0].item():8.4f}")

    print("=" * 70)

    # 将张量转换为可以复制到配置文件的格式
    print("\n" + "=" * 70)
    print("归一化参数（复制到配置文件中）:")
    print("=" * 70)

    # 转换为numpy数组以便于输出
    center_np = center.squeeze().cpu().numpy()
    scale_np = scale.squeeze().cpu().numpy()

    print("\n# 方式1: 使用PyTorch张量（推荐）")
    print("import torch")
    print(f"center = torch.tensor({center_np.tolist()}, dtype=torch.float32).view({center.shape[1]}, 1)")
    print(f"scale = torch.tensor({scale_np.tolist()}, dtype=torch.float32).view({scale.shape[1]}, 1)")

    print("\n# 方式2: 使用NumPy数组")
    print(f"import numpy as np")
    print(f"import torch")
    print(f"center = torch.from_numpy(np.array({center_np.tolist()}, dtype=np.float32).reshape({center.shape[1]}, 1))")
    print(f"scale = torch.from_numpy(np.array({scale_np.tolist()}, dtype=np.float32).reshape({scale.shape[1]}, 1))")

    print("\n# 方式3: 简化版本（如果所有特征使用相同归一化）")
    print(f"# 注意：这种方式会丢失每个特征的独立统计信息")
    global_mean = center_np.mean()
    global_std = scale_np.mean()
    print(f"center = {global_mean:.6f}")
    print(f"scale = {global_std:.6f}")

    print("\n" + "=" * 70)

    # 保存到文件
    output_file = "normalization_params.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("归一化参数计算结果\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"配置文件: {args.config_path}\n")
        f.write(f"训练集大小: {len(train_dataset)} trials\n")
        f.write(f"总数据点: {all_inputs.shape[2]}\n")
        f.write(f"特征数量: {all_inputs.shape[1]}\n\n")

        f.write("每个特征的统计信息:\n")
        f.write("-" * 70 + "\n")
        for i, feature_name in enumerate(input_names):
            f.write(f"{i:2d}. {feature_name:40s} | "
                   f"Mean: {center[0, i, 0].item():8.4f} | "
                   f"Std: {scale[0, i, 0].item():8.4f}\n")
        f.write("-" * 70 + "\n\n")

        f.write("PyTorch张量格式:\n")
        f.write("import torch\n")
        f.write(f"center = torch.tensor({center_np.tolist()}, dtype=torch.float32).view({center.shape[1]}, 1)\n")
        f.write(f"scale = torch.tensor({scale_np.tolist()}, dtype=torch.float32).view({scale.shape[1]}, 1)\n\n")

        f.write("简化版本:\n")
        f.write(f"center = {global_mean:.6f}\n")
        f.write(f"scale = {global_std:.6f}\n")

    print(f"\n归一化参数已保存到: {os.path.join(project_root, output_file)}")
    
    # 验证归一化效果
    print("\n" + "=" * 70)
    print("验证归一化效果:")
    print("=" * 70)
    
    normalized_data = (all_inputs - center) / scale
    normalized_mean = torch.mean(normalized_data, dim=(0, 2))
    normalized_std = torch.std(normalized_data, dim=(0, 2))
    
    print(f"归一化后的全局均值: {normalized_mean.mean().item():.6f} (应接近0)")
    print(f"归一化后的全局标准差: {normalized_std.mean().item():.6f} (应接近1)")
    print(f"归一化后的最小值: {normalized_data.min().item():.4f}")
    print(f"归一化后的最大值: {normalized_data.max().item():.4f}")
    
    print("\n完成！")


if __name__ == "__main__":
    compute_normalization_params()
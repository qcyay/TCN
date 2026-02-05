"""
计算训练数据的归一化参数（center和scale）
包括输入特征和输出标签的归一化参数

使用方法:
    # 从项目根目录执行
    python utils/compute_normalization_params.py --config_path configs.TCN.default_config

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
from utils.config_utils import load_config, apply_feature_selection
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
config = apply_feature_selection(config)


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
    all_labels = []

    for i in range(len(train_dataset)):
        # 新格式返回:
        # input_data: [num_input_features, sequence_length]
        # label_data: [num_label_features, sequence_length]
        # trial_length: 列表，包含一个整数值
        input_data, label_data, trial_length, _ = train_dataset[i]

        # 数据已经是正确的长度，不需要手动截取
        # input_data: [C, N], label_data: [N_label, N]
        all_inputs.append(input_data)  # [C, N]
        all_labels.append(label_data)  # [N_label, N]

        if (i + 1) % 10 == 0 or (i + 1) == len(train_dataset):
            print(f"  已处理: {i+1}/{len(train_dataset)} trials")

    # 合并所有数据: [num_trials, num_features, variable_lengths] -> [1, num_features, total_length]
    print("\n合并数据...")
    all_inputs = torch.cat(all_inputs, dim=1)  # 在时间维度拼接
    all_labels = torch.cat(all_labels, dim=1)  # 在时间维度拼接

    print(f"输入数据形状: {all_inputs.shape}")
    print(f"  特征维度: {all_inputs.shape[0]}")
    print(f"  时间维度: {all_inputs.shape[1]}")

    print(f"\n输出数据形状: {all_labels.shape}")
    print(f"  标签维度: {all_labels.shape[0]}")
    print(f"  时间维度: {all_labels.shape[1]}")

    # 检查NaN值
    nan_count = torch.isnan(all_labels).sum().item()
    total_count = all_labels.numel()
    print(f"\n标签数据中的NaN数量: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)\n")

    # 计算输入特征的均值和标准差
    print("计算输入特征的归一化参数...")

    # 在批次和时间维度上计算统计量
    input_center = torch.mean(all_inputs, dim=1, keepdim=True)  # [num_features, 1]
    input_scale = torch.std(all_inputs, dim=1, keepdim=True)    # [num_features, 1]

    # 避免除以零
    input_scale = torch.where(input_scale > 1e-8, input_scale, torch.ones_like(input_scale))

    print(f"Input Center形状: {input_center.shape}")
    print(f"Input Scale形状: {input_scale.shape}\n")

    # 计算输出标签的均值和标准差（忽略NaN值）
    print("计算输出标签的归一化参数（忽略NaN值）...")

    # 为每个标签特征分别计算统计量
    num_labels = all_labels.shape[0]
    label_center = torch.zeros(num_labels, 1, device=device)
    label_scale = torch.zeros(num_labels, 1, device=device)

    for i in range(num_labels):
        # 获取当前标签的所有数据: [time_steps]
        label_data = all_labels[i, :]  # [time_steps]

        # 创建非NaN的掩码
        valid_mask = ~torch.isnan(label_data)
        valid_count = valid_mask.sum().item()

        print(f"  标签 {i}: 有效数据点 {valid_count} / {label_data.numel()} ({valid_count/label_data.numel()*100:.2f}%)")

        # 只使用非NaN的值计算均值
        valid_data = label_data[valid_mask]
        label_center[i, 0] = torch.mean(valid_data)

        # 计算标准差
        label_scale[i, 0] = torch.std(valid_data)

    # 避免除以零
    label_scale = torch.where(label_scale > 1e-8, label_scale, torch.ones_like(label_scale))

    print(f"\nLabel Center形状: {label_center.shape}")
    print(f"Label Scale形状: {label_scale.shape}\n")

    # 输出详细统计信息
    print("=" * 70)
    print("输入特征的统计信息:")
    print("=" * 70)

    for i, feature_name in enumerate(input_names):
        print(f"{i:2d}. {feature_name:40s} | "
              f"Mean: {input_center[i, 0].item():8.4f} | "
              f"Std: {input_scale[i, 0].item():8.4f}")

    print("=" * 70)

    print("\n" + "=" * 70)
    print("输出标签的统计信息:")
    print("=" * 70)

    joint_names = ['髋关节力矩 (Hip)', '膝关节力矩 (Knee)']
    for i, label_name in enumerate(label_names):
        joint_desc = joint_names[i] if i < len(joint_names) else f"标签 {i}"
        print(f"{i:2d}. {label_name:40s} ({joint_desc:20s}) | "
              f"Mean: {label_center[i, 0].item():8.4f} | "
              f"Std: {label_scale[i, 0].item():8.4f}")

    print("=" * 70)

    # 将张量转换为可以复制到配置文件的格式
    print("\n" + "=" * 70)
    print("归一化参数（复制到配置文件中）:")
    print("=" * 70)

    # 转换为numpy数组以便于输出
    input_center_np = input_center.cpu().numpy()
    input_scale_np = input_scale.cpu().numpy()
    label_center_np = label_center.cpu().numpy()
    label_scale_np = label_scale.cpu().numpy()

    print("\n# ========== 输入特征归一化参数 ==========")
    print("\n# 方式1: 使用PyTorch张量（推荐）")
    print("import torch")
    print(f"center = torch.tensor({input_center_np.tolist()}, dtype=torch.float32).view({input_center.shape[0]}, 1)")
    print(f"scale = torch.tensor({input_scale_np.tolist()}, dtype=torch.float32).view({input_scale.shape[0]}, 1)")

    print("\n# 方式2: 使用NumPy数组")
    print(f"import numpy as np")
    print(f"import torch")
    print(f"center = torch.from_numpy(np.array({input_center_np.tolist()}, dtype=np.float32).reshape({input_center.shape[0]}, 1))")
    print(f"scale = torch.from_numpy(np.array({input_scale_np.tolist()}, dtype=np.float32).reshape({input_scale.shape[0]}, 1))")

    print("\n# 方式3: 简化版本（如果所有特征使用相同归一化）")
    print(f"# 注意：这种方式会丢失每个特征的独立统计信息")
    global_mean = input_center_np.mean()
    global_std = input_scale_np.mean()
    print(f"center = {global_mean:.6f}")
    print(f"scale = {global_std:.6f}")

    print("\n" + "=" * 70)
    print("\n# ========== 输出标签归一化参数 ==========")
    print("\n# 方式1: 使用PyTorch张量（推荐）")
    print("import torch")
    print(f"label_center = torch.tensor({label_center_np.tolist()}, dtype=torch.float32).view({label_center.shape[0]}, 1)")
    print(f"label_scale = torch.tensor({label_scale_np.tolist()}, dtype=torch.float32).view({label_scale.shape[0]}, 1)")

    print("\n# 方式2: 使用NumPy数组")
    print(f"import numpy as np")
    print(f"import torch")
    print(f"label_center = torch.from_numpy(np.array({label_center_np.tolist()}, dtype=np.float32).reshape({label_center.shape[0]}, 1))")
    print(f"label_scale = torch.from_numpy(np.array({label_scale_np.tolist()}, dtype=np.float32).reshape({label_scale.shape[0]}, 1))")

    print("\n# 标签统计摘要:")
    print(f"# 髋关节力矩: Mean = {label_center_np[0].mean():.4f} Nm/kg, Std = {label_scale_np[0].mean():.4f} Nm/kg")
    print(f"# 膝关节力矩: Mean = {label_center_np[1].mean():.4f} Nm/kg, Std = {label_scale_np[1].mean():.4f} Nm/kg")

    print("\n" + "=" * 70)

    # 保存到文件 - 确保保存在utils目录下
    # 如果当前在主目录执行，输出到utils目录
    # 如果当前在utils目录执行，输出到当前目录
    if os.path.basename(current_dir) == 'utils':
        output_dir = current_dir
    else:
        output_dir = os.path.join(project_root, 'utils')

    output_file = os.path.join(output_dir, "normalization_params.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("归一化参数计算结果\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"配置文件: {args.config_path}\n")
        f.write(f"训练集大小: {len(train_dataset)} trials\n")
        f.write(f"输入总数据点: {all_inputs.shape[1]}\n")
        f.write(f"输入特征数量: {all_inputs.shape[0]}\n")
        f.write(f"输出标签数量: {all_labels.shape[0]}\n")
        f.write(f"标签NaN数量: {nan_count} / {total_count} ({nan_count/total_count*100:.2f}%)\n\n")

        f.write("=" * 70 + "\n")
        f.write("输入特征的统计信息:\n")
        f.write("=" * 70 + "\n")
        for i, feature_name in enumerate(input_names):
            f.write(f"{i:2d}. {feature_name:40s} | "
                   f"Mean: {input_center[i, 0].item():8.4f} | "
                   f"Std: {input_scale[i, 0].item():8.4f}\n")
        f.write("=" * 70 + "\n\n")

        f.write("=" * 70 + "\n")
        f.write("输出标签的统计信息 (忽略NaN值):\n")
        f.write("=" * 70 + "\n")
        for i, label_name in enumerate(label_names):
            joint_desc = joint_names[i] if i < len(joint_names) else f"标签 {i}"
            # 计算每个标签的有效数据比例
            label_data = all_labels[i, :]
            valid_mask = ~torch.isnan(label_data)
            valid_ratio = valid_mask.sum().item() / label_data.numel() * 100
            f.write(f"{i:2d}. {label_name:40s} ({joint_desc:20s}) | "
                   f"Mean: {label_center[i, 0].item():8.4f} | "
                   f"Std: {label_scale[i, 0].item():8.4f} | "
                   f"Valid: {valid_ratio:.1f}%\n")
        f.write("=" * 70 + "\n\n")

        f.write("# ========== 输入特征归一化参数 ==========\n\n")
        f.write("PyTorch张量格式:\n")
        f.write("import torch\n")
        f.write(f"center = torch.tensor({input_center_np.tolist()}, dtype=torch.float32).view({input_center.shape[0]}, 1)\n")
        f.write(f"scale = torch.tensor({input_scale_np.tolist()}, dtype=torch.float32).view({input_scale.shape[0]}, 1)\n\n")

        f.write("简化版本:\n")
        f.write(f"center = {global_mean:.6f}\n")
        f.write(f"scale = {global_std:.6f}\n\n")

        f.write("# ========== 输出标签归一化参数 ==========\n\n")
        f.write("PyTorch张量格式:\n")
        f.write("import torch\n")
        f.write(f"label_center = torch.tensor({label_center_np.tolist()}, dtype=torch.float32).view({label_center.shape[0]}, 1)\n")
        f.write(f"label_scale = torch.tensor({label_scale_np.tolist()}, dtype=torch.float32).view({label_scale.shape[0]}, 1)\n\n")

        f.write("标签统计摘要:\n")
        f.write(f"髋关节力矩: Mean = {label_center_np[0].mean():.4f} Nm/kg, Std = {label_scale_np[0].mean():.4f} Nm/kg\n")
        f.write(f"膝关节力矩: Mean = {label_center_np[1].mean():.4f} Nm/kg, Std = {label_scale_np[1].mean():.4f} Nm/kg\n")

    print(f"\n归一化参数已保存到: {output_file}")

    # 验证归一化效果
    print("\n" + "=" * 70)
    print("验证归一化效果:")
    print("=" * 70)

    # 验证输入特征归一化
    print("\n输入特征:")
    normalized_inputs = (all_inputs - input_center) / input_scale
    input_normalized_mean = torch.mean(normalized_inputs, dim=1)
    input_normalized_std = torch.std(normalized_inputs, dim=1)

    print(f"  归一化后的全局均值: {input_normalized_mean.mean().item():.6f} (应接近0)")
    print(f"  归一化后的全局标准差: {input_normalized_std.mean().item():.6f} (应接近1)")
    print(f"  归一化后的最小值: {normalized_inputs.min().item():.4f}")
    print(f"  归一化后的最大值: {normalized_inputs.max().item():.4f}")

    # 验证输出标签归一化（忽略NaN值）
    print("\n输出标签 (忽略NaN值):")
    normalized_labels = (all_labels - label_center) / label_scale

    # 为每个标签分别计算统计量
    label_normalized_means = []
    label_normalized_stds = []

    for i in range(num_labels):
        label_data = normalized_labels[i, :]
        valid_mask = ~torch.isnan(label_data)

        valid_data = label_data[valid_mask]
        label_normalized_means.append(torch.mean(valid_data).item())
        label_normalized_stds.append(torch.std(valid_data).item())

    avg_mean = np.mean(label_normalized_means)
    avg_std = np.mean(label_normalized_stds)

    print(f"  归一化后的全局均值: {avg_mean:.6f} (应接近0)")
    print(f"  归一化后的全局标准差: {avg_std:.6f} (应接近1)")

    # 计算最小最大值（忽略NaN）
    valid_normalized = normalized_labels[~torch.isnan(normalized_labels)]
    print(f"  归一化后的最小值: {valid_normalized.min().item():.4f}")
    print(f"  归一化后的最大值: {valid_normalized.max().item():.4f}")

    # 输出每个标签的详细归一化效果
    print("\n各标签归一化后的统计:")
    for i, label_name in enumerate(label_names):
        joint_desc = joint_names[i] if i < len(joint_names) else f"标签 {i}"
        print(f"  {joint_desc}: Mean = {label_normalized_means[i]:.6f}, "
              f"Std = {label_normalized_stds[i]:.6f}")
    
    print("\n完成！")


if __name__ == "__main__":
    compute_normalization_params()
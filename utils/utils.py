import os
import shutil
from typing import List, Tuple, Dict
import random
import numpy as np
import torch

def setup_device(device_str: str) -> torch.device:
    """
    设置并验证训练设备

    参数:
        device_str: 设备字符串
            - 'cpu': 使用CPU
            - 'cuda': 使用默认GPU (GPU 0)
            - '0', '1', '2', '3', ...: 直接指定GPU编号
            - 'cuda:0', 'cuda:1', ...: 也支持这种格式

    返回:
        torch.device: 验证后的设备对象
    """
    # 显示可用的GPU信息
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n可用GPU数量: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024 ** 3:.1f} GB)")
    else:
        print("\n未检测到可用的GPU")

    # 解析设备字符串
    device_str = device_str.lower().strip()

    if device_str == "cpu":
        print(f"\n使用设备: CPU")
        return torch.device("cpu")

    # 处理纯数字（如 '0', '1', '2', '3'）
    if device_str.isdigit():
        if not torch.cuda.is_available():
            print(f"警告: CUDA不可用，回退到CPU")
            return torch.device("cpu")

        gpu_id = int(device_str)
        num_gpus = torch.cuda.device_count()

        if gpu_id >= num_gpus or gpu_id < 0:
            print(f"警告: GPU {gpu_id} 不存在（可用GPU: 0-{num_gpus - 1}），使用 GPU 0")
            gpu_id = 0

        device = torch.device(f"cuda:{gpu_id}")
        print(f"\n使用设备: GPU {gpu_id} - {torch.cuda.get_device_properties(gpu_id).name}")

        # 设置当前设备
        torch.cuda.set_device(gpu_id)

        return device

    # 处理 cuda 相关设备
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"警告: CUDA不可用，回退到CPU")
            return torch.device("cpu")

        # 解析GPU编号
        if device_str == "cuda":
            gpu_id = 0
        else:
            try:
                # 提取 cuda:X 中的 X
                gpu_id = int(device_str.split(':')[1])
            except (IndexError, ValueError):
                print(f"警告: 无效的设备字符串 '{device_str}'，使用 GPU 0")
                gpu_id = 0

        # 验证GPU编号
        num_gpus = torch.cuda.device_count()
        if gpu_id >= num_gpus or gpu_id < 0:
            print(f"警告: GPU {gpu_id} 不存在（可用GPU: 0-{num_gpus - 1}），使用 GPU 0")
            gpu_id = 0

        device = torch.device(f"cuda:{gpu_id}")
        print(f"\n使用设备: GPU {gpu_id} - {torch.cuda.get_device_properties(gpu_id).name}")

        # 设置当前设备
        torch.cuda.set_device(gpu_id)

        return device

    # 未知设备类型
    print(f"警告: 未知的设备类型 '{device_str}'，使用CPU")
    return torch.device("cpu")

def set_seed(seed):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn_tcn(batch):
    """TCN的collate函数"""
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [item[2][0] for item in batch]

    max_len = max(lengths)
    padded_inputs = []
    padded_labels = []

    for inp, lbl, length in zip(inputs, labels, lengths):
        if inp.shape[-1] < max_len:
            pad_len = max_len - inp.shape[-1]
            inp_pad = torch.zeros((1, inp.shape[1], pad_len), device=inp.device)
            lbl_pad = torch.zeros((1, lbl.shape[1], pad_len), device=lbl.device)
            inp = torch.cat([inp, inp_pad], dim=2)
            lbl = torch.cat([lbl, lbl_pad], dim=2)
        padded_inputs.append(inp)
        padded_labels.append(lbl)

    batch_inputs = torch.cat(padded_inputs, dim=0)
    batch_labels = torch.cat(padded_labels, dim=0)

    return batch_inputs, batch_labels, lengths


def collate_fn_predictor(batch):
    """预测模型的collate函数"""
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return inputs, labels


def collate_fn_generative(batch):
    """生成式模型的collate函数"""
    inputs = torch.stack([item[0] for item in batch])
    shifted_labels = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return inputs, shifted_labels, labels

def create_save_directory(config_path: str, model_type: str) -> str:
    """在logs文件夹下创建保存目录结构"""
    config_name = config_path.split(".")[-1]
    if config_name == "py":
        config_name = config_path.split(".")[-2]

    base_dir = os.path.join("logs", f"trained_{model_type.lower()}_{config_name}")
    os.makedirs(base_dir, exist_ok=True)

    existing_runs = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)) and item.isdigit():
                existing_runs.append(int(item))

    new_run_number = max(existing_runs) + 1 if existing_runs else 0
    save_dir = os.path.join(base_dir, str(new_run_number))
    os.makedirs(save_dir, exist_ok=True)

    print(f"创建保存目录: {save_dir}")
    return save_dir


def copy_config_file(config_path: str, save_dir: str):
    """
    复制配置文件到保存目录

    支持的路径格式:
    - configs.TCN.default_config.py -> configs/TCN/default_config.py
    - configs.TCN.default_config -> configs/TCN/default_config.py
    - configs/TCN/default_config.py -> configs/TCN/default_config.py
    - configs/TCN/default_config -> configs/TCN/default_config.py
    """
    possible_paths = []

    # 处理已经带.py后缀的情况
    if config_path.endswith(".py"):
        # 情况1: configs.TCN.default_config.py
        # 移除.py后缀，将点替换为路径分隔符，再加上.py
        path_without_py = config_path[:-3]  # 移除.py
        possible_paths.append(path_without_py.replace(".", os.sep) + ".py")
        possible_paths.append(path_without_py.replace(".", "/") + ".py")

        # 情况2: configs/TCN/default_config.py (已经是正确路径)
        possible_paths.append(config_path)

        # 情况3: 如果包含点，尝试将.py前的最后一个点替换为/
        if "." in path_without_py:
            parts = path_without_py.split(".")
            possible_paths.append(os.path.join(*parts) + ".py")
    else:
        # 没有.py后缀的情况
        # configs.TCN.default_config 或 configs/TCN/default_config

        # 情况1: 点分隔格式 -> 路径格式
        possible_paths.append(config_path.replace(".", os.sep) + ".py")
        possible_paths.append(config_path.replace(".", "/") + ".py")

        # 情况2: 已经是路径格式
        possible_paths.append(config_path + ".py")

        # 情况3: 混合格式处理
        if "." in config_path:
            parts = config_path.split(".")
            possible_paths.append(os.path.join(*parts) + ".py")

    # 去重
    possible_paths = list(dict.fromkeys(possible_paths))

    # 尝试找到存在的文件
    config_file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_file_path = path
            break

    if config_file_path:
        dest_path = os.path.join(save_dir, "config.py")
        shutil.copy(config_file_path, dest_path)
        print(f"✓ 配置文件已复制: {config_file_path} -> {dest_path}")
    else:
        print(f"⚠ 警告: 配置文件未找到，尝试过以下路径:")
        for path in possible_paths:
            print(f"    - {path}")

def reconstruct_sequences(
        all_estimates: torch.Tensor,
        all_labels: torch.Tensor,
        trial_sequence_counts: List[int],
        method: str = "only_first"
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    优化版本：利用测试数据不shuffle的特性，高效重组序列

    参数:
        all_estimates: [N, num_outputs, output_seq_len] 所有短序列的预测（按trial顺序）
        all_labels: [N, num_outputs, output_seq_len] 所有短序列的标签（按trial顺序）
        trial_sequence_counts: List[int] 每个trial的子序列数量
        method: 重组方法 'only_first' 或 'average'

    返回:
        reconstructed_estimates: List[Tensor[num_outputs, trial_len]] 每个trial的完整序列预测
        reconstructed_labels: List[Tensor[num_outputs, trial_len]] 每个trial的完整序列标签
    """
    reconstructed_estimates = []
    reconstructed_labels = []

    # 按trial_sequence_counts分割预测和标签
    estimates_splits = torch.split(all_estimates, trial_sequence_counts, dim=0)
    labels_splits = torch.split(all_labels, trial_sequence_counts, dim=0)

    if method == "only_first":
        # 每组只取第一个时间步（索引0），然后沿着时间维度拼接
        for trial_est, trial_lbl in zip(estimates_splits, labels_splits):
            # trial_est: [num_subsequences, num_outputs, output_seq_len]
            # 取每个子序列的第一个值: [:, :, 0] -> [num_subsequences, num_outputs]
            # 转置得到: [num_outputs, num_subsequences]
            reconstructed_estimates.append(trial_est[:, :, 0].t().contiguous())
            reconstructed_labels.append(trial_lbl[:, :, 0].t().contiguous())

    elif method == "average":
        # 使用unfold展开然后平均的方式
        for trial_est, trial_lbl in zip(estimates_splits, labels_splits):
            # trial_est: [num_subsequences, num_outputs, output_seq_len]
            num_subseqs, num_outputs, output_seq_len = trial_est.shape

            # 完整序列长度 = 子序列数量 + output_seq_len - 1
            # （因为最后一个子序列也会预测output_seq_len个点）
            full_len = num_subseqs + output_seq_len - 1

            # 为每个输出通道分别处理
            est_full = torch.zeros(num_outputs, full_len, device=trial_est.device)
            lbl_full = torch.zeros(num_outputs, full_len, device=trial_lbl.device)
            count = torch.zeros(num_outputs, full_len, device=trial_est.device)

            # 生成索引矩阵用于累加
            # 每个子序列i对应的位置是 i, i+1, ..., i+output_seq_len-1
            for i in range(num_subseqs):
                # 对应的位置索引
                positions = torch.arange(i, i + output_seq_len, device=trial_est.device)
                # 使用scatter_add累加
                est_full.index_add_(1, positions, trial_est[i])
                lbl_full.index_add_(1, positions, trial_lbl[i])
                count.index_add_(1, positions, torch.ones(num_outputs, output_seq_len, device=trial_est.device))

            # 计算平均值
            est_avg = est_full / count
            lbl_avg = lbl_full / count

            reconstructed_estimates.append(est_avg)
            reconstructed_labels.append(lbl_avg)

    else:
        raise ValueError(f"未知的重组方法: {method}")

    return reconstructed_estimates, reconstructed_labels


def reconstruct_sequences_from_predictions(
        all_estimates: torch.Tensor,
        all_labels: torch.Tensor,
        trial_info: List[Tuple[int, int, int]],
        method: str = "only_first"
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    将短序列预测重组回完整序列

    参数:
        all_estimates: [N, num_outputs, output_seq_len] 所有短序列的预测
        all_labels: [N, num_outputs, output_seq_len] 所有短序列的标签
        trial_info: List[(trial_idx, input_start_idx, seq_len)] 每个短序列的信息
        method: 重组方法 'only_first' 或 'average'

    返回:
        reconstructed_estimates: List[Tensor[num_outputs, trial_len]] 每个trial的完整序列预测
        reconstructed_labels: List[Tensor[num_outputs, trial_len]] 每个trial的完整序列标签
    """
    # 按trial分组
    trial_groups = {}
    for idx, (trial_idx, input_start_idx, seq_len) in enumerate(trial_info):
        if trial_idx not in trial_groups:
            trial_groups[trial_idx] = []
        trial_groups[trial_idx].append((input_start_idx, idx, seq_len))

    reconstructed_estimates = []
    reconstructed_labels = []

    for trial_idx in sorted(trial_groups.keys()):
        group = sorted(trial_groups[trial_idx], key=lambda x: x[0])  # 按input_start_idx排序

        # 确定完整序列长度
        max_end_pos = max(start_idx + seq_len for start_idx, _, seq_len in group)
        num_outputs = all_estimates.size(1)
        output_seq_len = all_estimates.size(2)

        if method == "only_first":
            # 只使用每个短序列的第一个预测值
            trial_estimate = torch.full((num_outputs, max_end_pos), float('nan'), device=all_estimates.device)
            trial_label = torch.full((num_outputs, max_end_pos), float('nan'), device=all_labels.device)

            for start_idx, pred_idx, seq_len in group:
                # 只取第一个预测值
                end_pos = start_idx + seq_len
                trial_estimate[:, end_pos:end_pos + 1] = all_estimates[pred_idx, :, 0:1]
                trial_label[:, end_pos:end_pos + 1] = all_labels[pred_idx, :, 0:1]

        elif method == "average":
            # 对每个位置的所有预测值取平均
            # 使用累加和计数来实现平均
            trial_estimate_sum = torch.zeros((num_outputs, max_end_pos), device=all_estimates.device)
            trial_estimate_count = torch.zeros((num_outputs, max_end_pos), device=all_estimates.device)
            trial_label_sum = torch.zeros((num_outputs, max_end_pos), device=all_labels.device)
            trial_label_count = torch.zeros((num_outputs, max_end_pos), device=all_labels.device)

            for start_idx, pred_idx, seq_len in group:
                pred = all_estimates[pred_idx]  # [num_outputs, output_seq_len]
                lbl = all_labels[pred_idx]  # [num_outputs, output_seq_len]

                end_pos = start_idx + seq_len
                # 将预测值添加到对应位置
                for i in range(min(output_seq_len, max_end_pos - end_pos)):
                    pos = end_pos + i
                    # 只累加非NaN值
                    valid_mask = ~torch.isnan(pred[:, i])
                    trial_estimate_sum[valid_mask, pos] += pred[valid_mask, i]
                    trial_estimate_count[valid_mask, pos] += 1

                    valid_mask = ~torch.isnan(lbl[:, i])
                    trial_label_sum[valid_mask, pos] += lbl[valid_mask, i]
                    trial_label_count[valid_mask, pos] += 1

            # 计算平均
            trial_estimate = torch.where(
                trial_estimate_count > 0,
                trial_estimate_sum / trial_estimate_count,
                torch.full_like(trial_estimate_sum, float('nan'))
            )
            trial_label = torch.where(
                trial_label_count > 0,
                trial_label_sum / trial_label_count,
                torch.full_like(trial_label_sum, float('nan'))
            )

        else:
            raise ValueError(f"未知的重组方法: {method}")

        reconstructed_estimates.append(trial_estimate)
        reconstructed_labels.append(trial_label)

    return reconstructed_estimates, reconstructed_labels
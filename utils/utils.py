import sys
sys.path.insert(0, '.')
import os
import re
import shutil
from typing import List, Tuple, Dict
from collections import defaultdict
from itertools import repeat
import random
import numpy as np
import inspect
# 设置Matplotlib使用Agg后端（无图形界面）
import matplotlib
matplotlib.use('Agg')  # 关键：使用非交互式后端
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.predictor_model import PredictorTransformer
from models.generative_model import GenerativeTransformer
from models.tcn import TCN

# 记录“模型 __init__ 参数名” 到 “config 中字段名”的映射
# 没写在这里的默认认为两边同名
CONFIG_ATTRIBUTES = {
    "GenerativeTransformer": {
        "d_model": "gen_d_model",
        "nhead": "gen_nhead",
        "num_encoder_layers": "gen_num_encoder_layers",
        "num_decoder_layers": "gen_num_decoder_layers",
        "dim_feedforward": "gen_dim_feedforward",
        "dropout": "gen_dropout",
        "sequence_length": "gen_sequence_length",
        # encoder_type / use_positional_encoding / center / scale 同名就不用写了
    },
    "Transformer": {
        "dropout": "transformer_dropout",
        # 其它：d_model / nhead / sequence_length / output_sequence_length 等同名
    },
    "TCN": {
        # TCN 这边假设基本同名，就不用写映射了
    },
}

MODEL_SPECIFIC_FIELDS: Dict[str, Dict[str, str]] = {
    # checkpoint_key : config中对应的属性名
    "GenerativeTransformer": {
        "d_model": "gen_d_model",
        "nhead": "gen_nhead",
        "num_encoder_layers": "gen_num_encoder_layers",
        "num_decoder_layers": "gen_num_decoder_layers",
        "dim_feedforward": "gen_dim_feedforward",
        "dropout": "gen_dropout",
        "sequence_length": "gen_sequence_length",
        "encoder_type": "encoder_type",
        "use_positional_encoding": "use_positional_encoding",
    },
    "Transformer": {
        "d_model": "d_model",
        "nhead": "nhead",
        "num_encoder_layers": "num_encoder_layers",
        "dim_feedforward": "dim_feedforward",
        "dropout": "transformer_dropout",
        "sequence_length": "sequence_length",
        "output_sequence_length": "output_sequence_length",
        "use_positional_encoding": "use_positional_encoding",
    },
    # 默认当作 TCN
    "TCN": {
        "num_channels": "num_channels",
        "ksize": "ksize",
        "dropout": "dropout",
        "eff_hist": "eff_hist",
        "spatial_dropout": "spatial_dropout",
        "activation": "activation",
        "norm": "norm",
    },
}

# ==================== 运动类型分类配置 ====================

# 运动类型到大类的映射（用于test.py分类评估）
# 键为运动类型的正则表达式，值为大类名称
ACTION_TO_CATEGORY = {
    # 周期性运动 (Cyclic)
    r"^normal_walk_.*_(shuffle|0-6|1-2|1-8).*": "Cyclic",  # Level ground walk
    r"^walk_backward_.*": "Cyclic",  # Backwards walk
    r"^weighted_walk_.*": "Cyclic",  # 25 lb Loaded walk
    r"^normal_walk_.*_(2-0|2-5).*": "Cyclic",  # Run
    r"^dynamic_walk_.*(toe-walk|heel-walk).*": "Cyclic",  # Toe and heel walk
    r"^incline_walk_.*up.*": "Cyclic",  # Inclined walk
    r"^stairs_.*down.*": "Cyclic",  # Stair descent
    r"^stairs_.*up.*": "Cyclic",  # Stair ascent
    r"^incline_walk_.*down.*": "Cyclic",  # Declined walk

    # 阻抗性运动 (Impedance-like)
    r"^poses_.*": "Impedance-like",  # Standing poses
    r"^jump_.*_(hop|vertical|180|90-f|90-s).*": "Impedance-like",  # Jump in place
    r"^sit_to_stand_.*": "Impedance-like",  # Sit and stand
    r"^lift_weight_.*": "Impedance-like",  # Lift and place weight
    r"^tug_of_war_.*": "Impedance-like",  # Tug of war
    r"^jump_.*_(fb|lateral).*": "Impedance-like",  # Jump across (part)
    r"^side_shuffle_.*": "Impedance-like",  # Jump across (part)
    r"^lunges_.*": "Impedance-like",  # Lunge
    r"^ball_toss_.*": "Impedance-like",  # Medicine ball toss
    r"^squats_.*": "Impedance-like",  # Squat
    r"^step_ups_.*": "Impedance-like",  # Step up

    # 非结构化运动 (Unstructured)
    r"^dynamic_walk_.*(high-knees|butt-kicks).*": "Unstructured",  # Calisthenics (part)
    r"^normal_walk_.*skip.*": "Unstructured",  # Calisthenics (part)
    r"^tire_run_.*": "Unstructured",  # Calisthenics (part)
    r"^push_.*": "Unstructured",  # Push and pull recovery
    r"^turn_and_step_.*": "Unstructured",  # Turns
    r"^cutting_.*": "Unstructured",  # Cut
    r"^twister_.*": "Unstructured",  # Twister
    r"^meander_.*": "Unstructured",  # Meander
    r"^start_stop_.*": "Unstructured",  # Start and stop
    r"^obstacle_walk_.*": "Unstructured",  # Step over
    r"^curb_.*": "Unstructured",  # Curb
}

# 训练集中未见过的任务（用于单独评估）
UNSEEN_ACTION_PATTERNS = [
    r"^lunges_.*",  # Lunge
    r"^stairs_.*up.*",  # Stair ascent
    r"^incline_walk_.*down.*",  # Declined walk
    r"^start_stop_.*",  # Start and stop
    r"^ball_toss_.*",  # Medicine ball toss
    r"^obstacle_walk_.*",  # Step over
    r"^squats_.*",  # Squat
    r"^curb_.*",  # Curb
    r"^step_ups_.*",  # Step up
]

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value: float) -> bool:
        if self.best_value is None:
            self.best_value = current_value
            return False

        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def create_model(config, device: torch.device, resume_path: str = None) -> Tuple[nn.Module, int]:
    """创建或加载模型"""
    model_type = config.model_type

    if resume_path and os.path.exists(resume_path):
        print(f"从检查点恢复训练: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        start_epoch = checkpoint.get("epoch", 0) + 1

        # 根据模型类型创建模型
        model = create_model_from_config(model_type, checkpoint, config, device)

        model.load_state_dict(checkpoint["state_dict"])
        print(f"模型已加载,从第 {start_epoch} 轮继续训练")
    else:
        print(f"创建新的{model_type}模型")
        model = create_model_from_config(model_type, None, config, device)
        start_epoch = 0
        print(f"新模型已创建，参数数量: {sum(p.numel() for p in model.parameters()):,}")

    return model, start_epoch

def create_model_from_config(model_type, checkpoint=None, config=None, device=None):
    """
    根据 model_type + config (+ 可选 checkpoint) 创建模型。
    - checkpoint 为 None：所有超参数来自 config（带别名映射）
    - checkpoint 不为 None：优先用 checkpoint[name]，否则用 config 对应字段
    """

    model_classes = {
        "GenerativeTransformer": GenerativeTransformer,
        "Transformer": PredictorTransformer,
        "TCN": TCN,
    }
    model_class = model_classes[model_type]
    attributes_map = CONFIG_ATTRIBUTES.get(model_type, {})

    sig = inspect.signature(model_class.__init__)
    params = {}
    sources = {}

    # 分两种情况写清楚一点
    if checkpoint is None:
        # 完全新建模型，所有参数来自 config
        for name, p in sig.parameters.items():
            if name == "self":
                continue

            cfg_attr = attributes_map.get(name, name)  # 映射到 config 字段名
            if hasattr(config, cfg_attr):
                params[name] = getattr(config, cfg_attr)
                sources[name] = f"config.{cfg_attr}"
    else:
        # 从 checkpoint 恢复，优先 checkpoint，其次 config
        for name, p in sig.parameters.items():
            if name == "self":
                continue

            if name in checkpoint:
                params[name] = checkpoint[name]
                sources[name] = "checkpoint"
            else:
                cfg_attr = attributes_map.get(name, name)
                if hasattr(config, cfg_attr):
                    params[name] = getattr(config, cfg_attr)
                    sources[name] = f"config.{cfg_attr}"
                # 两边都没有就跳过，让 __init__ 用默认值（如果有）

    print(f"Creating {model_type} with parameters:")
    for k, v in params.items():
        print(f"  {k} ({sources.get(k, 'unknown')}): {v}")

    model = model_class(**params).to(device)
    return model

def save_model(model: nn.Module, save_dir: str, epoch: int, config,
               optimizer: optim.Optimizer = None,
               scheduler: ReduceLROnPlateau = None):
    """保存模型检查点"""

    checkpoint = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "model_type": config.model_type,
        "input_size": config.input_size,
        "output_size": config.output_size,
        "center": config.center,
        "scale": config.scale
    }

    # 利用MODEL_SPECIFIC_FIELDS动态添加模型特定参数
    model_fields = MODEL_SPECIFIC_FIELDS.get(config.model_type, {})
    for checkpoint_key, config_attr in model_fields.items():
        checkpoint[checkpoint_key] = getattr(config, config_attr)

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.tar")
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")

def setup_device(device_str: str) -> torch.device:
    """
    设置并验证训练设备

    参数:
        device_str: 设备字符串
            - 'cpu': 使用CPU
            - 'cuda': 使用默认GPU (GPU 0)
            - '0', '1', '2', '3', ...: 直接指定GPU编号

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

    if device_str == "cpu":
        print(f"\n使用设备: CPU")
        return torch.device("cpu")

    # 处理纯数字（如 '0', '1', '2', '3'）
    if device_str.isdigit() or device_str.startswith("cuda"):

        # 解析GPU编号
        if device_str == "cuda":
            gpu_id = 0

        gpu_id = int(device_str)
        num_gpus = torch.cuda.device_count()

        device = torch.device(f"cuda:{gpu_id}")
        print(f"\n使用设备: GPU {gpu_id} - {torch.cuda.get_device_properties(gpu_id).name}")

        # 设置当前设备
        torch.cuda.set_device(gpu_id)

        return device

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
    masks = [item[3] for item in batch]

    # breakpoint()

    max_len = max(lengths)
    padded_inputs = []
    padded_labels = []

    for inp, lbl, length in zip(inputs, labels, lengths):
        if inp.shape[-1] < max_len:
            pad_len = max_len - inp.shape[-1]
            inp_pad = torch.zeros((inp.shape[0], pad_len), device=inp.device)
            lbl_pad = torch.zeros((lbl.shape[0], pad_len), device=lbl.device)
            inp = torch.cat([inp, inp_pad], dim=1)
            lbl = torch.cat([lbl, lbl_pad], dim=1)
        padded_inputs.append(inp)
        padded_labels.append(lbl)

    batch_inputs = torch.stack(padded_inputs, dim=0)
    batch_labels = torch.stack(padded_labels, dim=0)

    return batch_inputs, batch_labels, lengths, masks


def collate_fn_predictor(batch):
    """预测模型的collate函数"""
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch]) if batch[0][2] is not None else None
    return inputs, labels, masks


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
    - configs/TCN/default_config.py -> configs/TCN/default_config.py
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
        all_masks: torch.Tensor,
        trial_sequence_counts: List[int],
        method: str = "only_first"
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    优化版本：利用测试数据不shuffle的特性，高效重组序列

    参数:
        all_estimates: [N, num_outputs, output_seq_len] 所有短序列的预测（按trial顺序）
        all_labels: [N, num_outputs, output_seq_len] 所有短序列的标签（按trial顺序）
        all_masks: [N, num_outputs, output_seq_len] 所有短序列的掩码（按trial顺序）
        trial_sequence_counts: List[int] 每个trial的子序列数量
        method: 重组方法 'only_first' 或 'average'

    返回:
        reconstructed_estimates: List[Tensor[num_outputs, trial_len]] 每个trial的完整序列预测
        reconstructed_labels: List[Tensor[num_outputs, trial_len]] 每个trial的完整序列标签
        reconstructed_masks: List[Tensor[num_outputs, trial_len]] 每个trial的完整序列掩码
    """
    reconstructed_estimates = []
    reconstructed_labels = []
    reconstructed_masks = [] if all_masks is not None else None

    # 按trial_sequence_counts分割预测和标签
    estimates_splits = torch.split(all_estimates, trial_sequence_counts, dim=0)
    labels_splits = torch.split(all_labels, trial_sequence_counts, dim=0)
    masks_splits = torch.split(all_masks, trial_sequence_counts, dim=0) if all_masks is not None else repeat(None)

    # 对于标签和掩码，每组只取第一个时间步（索引0），然后沿着时间维度拼接
    for trial_lbl, trial_mask in zip(labels_splits, masks_splits):
        # trial_lbl: [num_subsequences, num_outputs, output_seq_len]
        # 取每个子序列的第一个值: [:, :, 0] -> [num_subsequences, num_outputs]
        # 转置得到: [num_outputs, num_subsequences]
        reconstructed_labels.append(trial_lbl[:, :, 0].t().contiguous())
        if trial_mask is not None:
            reconstructed_masks.append(trial_mask[:, :, 0].t().contiguous())

    if method == "only_first":
        # 每组只取第一个时间步（索引0），然后沿着时间维度拼接
        for trial_est in estimates_splits:
            # trial_est: [num_subsequences, num_outputs, output_seq_len]
            # 取每个子序列的第一个值: [:, :, 0] -> [num_subsequences, num_outputs]
            # 转置得到: [num_outputs, num_subsequences]
            reconstructed_estimates.append(trial_est[:, :, 0].t().contiguous())

    elif method == "average":
        # 使用unfold展开然后平均的方式
        for trial_est in estimates_splits:
            # trial_est: [num_subsequences, num_outputs, output_seq_len]
            num_subseqs, num_outputs, output_seq_len = trial_est.shape

            # # 完整序列长度 = 子序列数量 + output_seq_len - 1
            # # （因为最后一个子序列也会预测output_seq_len个点）
            # full_len = num_subseqs + output_seq_len - 1

            # 为每个输出通道分别处理
            est_full = torch.zeros(num_outputs, num_subseqs, device=trial_est.device)
            count = torch.zeros(num_outputs, num_subseqs, device=trial_est.device)

            # 生成索引矩阵用于累加
            # 每个子序列i对应的位置是 i, i+1, ..., i+output_seq_len-1
            for i in range(num_subseqs):
                # 对应的位置索引
                positions = torch.arange(i, min(i + output_seq_len, num_subseqs), device=trial_est.device)
                # 使用scatter_add累加
                est_full.index_add_(1, positions, trial_est[i])
                count.index_add_(1, positions, torch.ones(num_outputs, positions.size(0), device=trial_est.device))

            # 计算平均值
            est_avg = est_full / count

            reconstructed_estimates.append(est_avg)

    else:
        raise ValueError(f"未知的重组方法: {method}")

    return reconstructed_estimates, reconstructed_labels, reconstructed_masks

def categorize_trial(trial_name: str, action_to_category: Dict[str, str]) -> str:
    """
    根据trial名称判断所属类别

    参数:
        trial_name: 试验名称，格式为 "参与者/运动类型"
        action_to_category: 运动类型到类别的映射字典

    返回:
        类别名称，如果未匹配返回 "Unknown"
    """
    # 提取运动类型部分
    action_name = trial_name.split("/")[-1].split("\\")[-1]

    # 尝试匹配每个正则表达式
    for pattern, category in action_to_category.items():
        if re.match(pattern, action_name):
            return category

    return "Unknown"


def check_unseen_task(trial_name: str, unseen_patterns: List[str]) -> bool:
    """
    判断trial是否属于未见过的任务

    参数:
        trial_name: 试验名称
        unseen_patterns: 未见过任务的正则表达式列表

    返回:
        True如果属于未见过的任务，否则False
    """
    action_name = trial_name.split("/")[-1].split("\\")[-1]

    for pattern in unseen_patterns:
        if re.match(pattern, action_name):
            return True

    return False

def compute_metrics_on_sequences(
        estimates_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        masks_list: List[torch.Tensor],
        num_outputs: int
) -> Dict:
    """
    在完整序列上计算指标

    参数:
        estimates_list: List[Tensor[num_outputs, seq_len]] 每个trial的预测序列
        labels_list: List[Tensor[num_outputs, seq_len]] 每个trial的标签序列
        masks_list: List[Tensor[num_outputs, seq_len]] 每个trial的动作掩码序列
        num_outputs: 输出通道数

    返回:
        metrics: 包含每个输出通道指标的字典
    """
    metrics = {}
    masks_list = masks_list if masks_list is not None else repeat(None)

    for j in range(num_outputs):
        rmse_sum = 0.0
        r2_sum = 0.0
        mae_percent_sum = 0.0
        count = 0

        for est_seq, lbl_seq, mask_seq in zip(estimates_list, labels_list, masks_list):
            est = est_seq[j]  # [seq_len]
            lbl = lbl_seq[j]  # [seq_len]
            if mask_seq is not None:
                mask = mask_seq[j] # [seq_len]

            # 创建有效数据掩码
            valid_mask = ~torch.isnan(est) & ~torch.isnan(lbl)
            if mask_seq is not None:
                valid_mask = valid_mask & (mask == 1)

            if valid_mask.sum() == 0:
                continue

            est_valid = est[valid_mask]
            lbl_valid = lbl[valid_mask]

            # RMSE
            rmse = torch.sqrt(torch.mean((est_valid - lbl_valid) ** 2))

            # R²
            ss_res = torch.sum((lbl_valid - est_valid) ** 2)
            ss_tot = torch.sum((lbl_valid - torch.mean(lbl_valid)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))

            # MAE as percentage
            label_range = torch.max(lbl_valid) - torch.min(lbl_valid)
            mae = torch.mean(torch.abs(est_valid - lbl_valid))
            mae_percent = (mae / (label_range + 1e-8)) * 100.0

            rmse_sum += rmse.item()
            r2_sum += r2.item()
            mae_percent_sum += mae_percent.item()
            count += 1

        if count > 0:
            metrics[f"output_{j}"] = {
                "rmse": rmse_sum / count,
                "r2": r2_sum / count,
                "mae_percent": mae_percent_sum / count,
                "count": count
            }
        else:
            metrics[f"output_{j}"] = {
                "rmse": 0.0,
                "r2": 0.0,
                "mae_percent": 0.0,
                "count": 0
            }

    return metrics

def compute_metrics_tcn_trial(estimates: torch.Tensor,
                              labels: torch.Tensor,
                              valid_mask: torch.Tensor,
                              num_outputs: int) -> dict:
    """
    TCN模型的指标计算（针对单个长序列试验）

    策略：所有指标在整个长序列上计算
    """
    metrics = {}

    for j in range(num_outputs):
        est = estimates[j]
        lbl = labels[j]
        mask = valid_mask[j]

        if mask.sum() == 0:
            metrics[f"output_{j}"] = {
                "rmse": 0.0,
                "r2": 0.0,
                "mae_percent": 0.0,
                "count": 0
            }
            continue

        est_valid = est[mask]
        lbl_valid = lbl[mask]

        # RMSE
        rmse = torch.sqrt(torch.mean((est_valid - lbl_valid) ** 2))

        # R²
        ss_res = torch.sum((lbl_valid - est_valid) ** 2)
        ss_tot = torch.sum((lbl_valid - torch.mean(lbl_valid)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # MAE as percentage
        label_range = torch.max(lbl_valid) - torch.min(lbl_valid)
        mae = torch.mean(torch.abs(est_valid - lbl_valid))
        mae_percent = (mae / (label_range + 1e-8)) * 100.0

        metrics[f"output_{j}"] = {
            "rmse": rmse.item(),
            "r2": r2.item(),
            "mae_percent": mae_percent.item(),
            "count": 1
        }

    return metrics

def compute_metrics_per_sequence(
        estimates: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    计算单个序列的指标

    参数:
        estimates: [num_outputs, seq_len]
        labels: [num_outputs, seq_len]

    返回:
        metrics: 包含rmse, r2, mae_percent的字典
    """
    # 创建有效数据掩码
    valid_mask = ~torch.isnan(estimates) & ~torch.isnan(labels)
    if mask is not None:
        valid_mask = valid_mask & (mask == 1)

    if valid_mask.sum() == 0:
        return None

    est_valid = estimates[valid_mask]
    lbl_valid = labels[valid_mask]

    # RMSE
    rmse = torch.sqrt(torch.mean((est_valid - lbl_valid) ** 2)).item()

    # R²
    ss_res = torch.sum((lbl_valid - est_valid) ** 2)
    ss_tot = torch.sum((lbl_valid - torch.mean(lbl_valid)) ** 2)
    r2 = (1 - (ss_res / (ss_tot + 1e-8))).item()

    # Normalized MAE
    label_range = torch.max(lbl_valid) - torch.min(lbl_valid)
    mae = torch.mean(torch.abs(est_valid - lbl_valid))
    mae_percent = ((mae / (label_range + 1e-8)) * 100.0).item()

    return {'rmse': rmse, 'r2': r2, 'mae_percent': mae_percent}

def compute_category_metrics(
        estimates_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        masks_list: List[torch.Tensor],
        trial_names: List[str],
        label_names: List[str],
        action_to_category: Dict[str, str],
        unseen_patterns: List[str],
) -> Dict:
    """
    计算各类别的指标

    参数:
        estimates_list: List[Tensor[num_outputs, seq_len]] 每个trial的预测序列
        labels_list: List[Tensor[num_outputs, seq_len]] 每个trial的标签序列
        trial_names: 试验名称列表
        label_names: 标签名称列表
        action_to_category: 运动类型到类别的映射
        unseen_patterns: 未见过任务的正则表达式列表

    返回:
        category_metrics: 嵌套字典 {category: {label_name: {metric: [values]}}}
    """
    num_outputs = len(label_names)

    # 初始化存储结构
    # category_metrics[category][label_name][metric] = [value1, value2, ...]
    category_metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # 根据masks_list是否为None创建迭代器
    masks_list = masks_list if masks_list is not None else repeat(None)

    for est_seq, lbl_seq, mask_seq, trial_name in zip(estimates_list, labels_list, masks_list, trial_names):
        # 判断类别
        category = categorize_trial(trial_name, action_to_category)

        # 判断是否为未见过的任务
        is_unseen = check_unseen_task(trial_name, unseen_patterns)

        # 对每个输出通道计算指标
        for j, label_name in enumerate(label_names):
            est = est_seq[j]  # [seq_len]
            lbl = lbl_seq[j]  # [seq_len]
            if mask_seq is not None:
                mask = mask_seq[j] # [seq_len]

            metrics = compute_metrics_per_sequence(est, lbl, mask)

            if metrics is not None:
                # 添加到对应类别
                category_metrics[category][label_name]['rmse'].append(metrics['rmse'])
                category_metrics[category][label_name]['r2'].append(metrics['r2'])
                category_metrics[category][label_name]['mae_percent'].append(metrics['mae_percent'])

                # 添加到"所有"类别
                category_metrics['All'][label_name]['rmse'].append(metrics['rmse'])
                category_metrics['All'][label_name]['r2'].append(metrics['r2'])
                category_metrics['All'][label_name]['mae_percent'].append(metrics['mae_percent'])

                # 如果是未见过的任务，额外添加到Unseen类别
                if is_unseen:
                    category_metrics['Unseen'][label_name]['rmse'].append(metrics['rmse'])
                    category_metrics['Unseen'][label_name]['r2'].append(metrics['r2'])
                    category_metrics['Unseen'][label_name]['mae_percent'].append(metrics['mae_percent'])

    return category_metrics


def print_category_results(category_metrics: Dict, label_names: List[str]):
    """打印各类别的测试结果"""
    print("\n" + "=" * 80)
    print("按类别统计的测试结果:")
    print("=" * 80)

    # 定义类别顺序
    category_order = ['All', 'Cyclic', 'Impedance-like', 'Unstructured', 'Unseen']

    for category in category_order:
        if category not in category_metrics:
            continue

        print(f"\n【{category}】")
        print("-" * 80)

        for label_name in label_names:
            if label_name not in category_metrics[category]:
                continue

            metrics = category_metrics[category][label_name]

            # 计算均值
            mean_rmse = sum(metrics['rmse']) / len(metrics['rmse']) if metrics['rmse'] else 0
            mean_r2 = sum(metrics['r2']) / len(metrics['r2']) if metrics['r2'] else 0
            mean_mae = sum(metrics['mae_percent']) / len(metrics['mae_percent']) if metrics['mae_percent'] else 0

            print(f"\n  {label_name}:")
            print(f"    RMSE: {mean_rmse:.4f} Nm/kg (n={len(metrics['rmse'])})")
            print(f"    R²: {mean_r2:.4f} (n={len(metrics['r2'])})")
            print(f"    MAE: {mean_mae:.2f}% (n={len(metrics['mae_percent'])})")

    print("\n" + "=" * 80 + "\n")

def create_boxplots(
        category_metrics: Dict,
        label_names: List[str],
        save_dir: str,
        categories_to_plot: List[str] = ['Cyclic', 'Impedance-like', 'Unstructured']
):
    """
    创建分面箱线图

    参数:
        category_metrics: 类别指标字典
        label_names: 标签名称列表
        save_dir: 保存目录
        categories_to_plot: 要绘制的类别列表
    """
    # 定义指标的显示名称和单位
    metric_info = {
        'rmse': {'name': 'RMSE', 'unit': 'Nm/kg', 'ylabel': 'RMSE (Nm/kg)'},
        'r2': {'name': 'R²', 'unit': '', 'ylabel': 'R²'},
        'mae_percent': {'name': 'MAE', 'unit': '%', 'ylabel': 'Normalized MAE (%)'}
    }

    # 简化标签名称用于显示
    label_display_names = {
        "hip_flexion_r_moment": "Hip",
        "knee_angle_r_moment": "Knee",
        "hip_flexion_l_moment": "Hip",
        "knee_angle_l_moment": "Knee"
    }

    for label_idx, label_name in enumerate(label_names):
        # 为每个输出通道创建一个图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{label_display_names.get(label_name, label_name)} Moment Prediction',
                     fontsize=16, fontweight='bold')

        for metric_idx, (metric_key, metric_data) in enumerate(metric_info.items()):
            ax = axes[metric_idx]

            # 收集数据
            data_to_plot = []
            positions = []
            labels = []

            for cat_idx, category in enumerate(categories_to_plot):
                if category in category_metrics and label_name in category_metrics[category]:
                    values = category_metrics[category][label_name].get(metric_key, [])
                    if values:
                        data_to_plot.append(values)
                        positions.append(cat_idx + 1)
                        labels.append(category)

            if not data_to_plot:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(metric_data['name'])
                continue

            # 创建箱线图
            bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                            patch_artist=True, showfliers=True,
                            boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                            medianprops=dict(color='black', linewidth=2),
                            whiskerprops=dict(color='black', linewidth=1.5),
                            capprops=dict(color='black', linewidth=1.5),
                            flierprops=dict(marker='o', markerfacecolor='gray', markersize=4,
                                            linestyle='none', markeredgecolor='gray'))

            # 添加均值标记（黑色方块）
            for i, values in enumerate(data_to_plot):
                mean_val = sum(values) / len(values)
                ax.plot(positions[i], mean_val, marker='s', markersize=6,
                        color='black', zorder=3)

            # 设置标题和标签
            ax.set_title(metric_data['name'], fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_data['ylabel'], fontsize=12)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=11)
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # 为R²添加灰色背景
            if metric_key == 'r2':
                ax.set_facecolor('#f0f0f0')

        plt.tight_layout()

        # 保存图片
        save_path = os.path.join(save_dir, f'boxplot_{label_name}_{"-".join(categories_to_plot)}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"箱线图已保存: {save_path}")
        plt.close()

def save_metrics_to_file(category_metrics: Dict, label_names: List[str], save_dir: str):
    """将指标保存到文本文件"""
    save_path = os.path.join(save_dir, 'test_results_by_category.txt')

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("按类别统计的测试结果\n")
        f.write("=" * 80 + "\n\n")

        # 定义类别顺序
        category_order = ['All', 'Cyclic', 'Impedance-like', 'Unstructured', 'Unseen']

        for category in category_order:
            if category not in category_metrics:
                continue

            f.write(f"【{category}】\n")
            f.write("-" * 80 + "\n")

            for label_name in label_names:
                if label_name not in category_metrics[category]:
                    continue

                metrics = category_metrics[category][label_name]

                # 计算均值和标准差
                mean_rmse = sum(metrics['rmse']) / len(metrics['rmse']) if metrics['rmse'] else 0
                std_rmse = (sum((x - mean_rmse) ** 2 for x in metrics['rmse']) / len(metrics['rmse'])) ** 0.5 if \
                metrics['rmse'] else 0

                mean_r2 = sum(metrics['r2']) / len(metrics['r2']) if metrics['r2'] else 0
                std_r2 = (sum((x - mean_r2) ** 2 for x in metrics['r2']) / len(metrics['r2'])) ** 0.5 if metrics[
                    'r2'] else 0

                mean_mae = sum(metrics['mae_percent']) / len(metrics['mae_percent']) if metrics['mae_percent'] else 0
                std_mae = (sum((x - mean_mae) ** 2 for x in metrics['mae_percent']) / len(
                    metrics['mae_percent'])) ** 0.5 if metrics['mae_percent'] else 0

                f.write(f"\n  {label_name}:\n")
                f.write(f"    RMSE: {mean_rmse:.4f} ± {std_rmse:.4f} Nm/kg (n={len(metrics['rmse'])})\n")
                f.write(f"    R²: {mean_r2:.4f} ± {std_r2:.4f} (n={len(metrics['r2'])})\n")
                f.write(f"    MAE: {mean_mae:.2f} ± {std_mae:.2f}% (n={len(metrics['mae_percent'])})\n")

            f.write("\n")

        f.write("=" * 80 + "\n")

    print(f"测试结果已保存到: {save_path}")

if __name__ == '__main__':

    # from types import SimpleNamespace
    #
    # config = SimpleNamespace()
    #
    # # config 对象
    # config.input_size = 10
    # config.output_size = 5
    # config.d_model = 512
    # config.transformer_dropout = 0.2
    #
    # # checkpoint 内容（可能包含部分覆盖参数）
    # checkpoint = {
    #     'input_size': 12,  # 覆盖config中的值
    #     # 没有output_size，使用config默认值
    #     'd_model': 256  # 覆盖config中的值
    # }
    #
    # # 调用函数
    # # model = create_model_from_config('Transformer', None, config, 'cpu')
    # model = create_model_from_config('Transformer', checkpoint, config, 'cpu')

    copy_config_file('configs.TCN.default_config.py', '.')
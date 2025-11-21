import os
import shutil
from typing import List, Tuple, Dict
import random
import numpy as np
import inspect
import torch
from models.predictor_model import PredictorTransformer
from models.generative_model import GenerativeTransformer
from models.tcn import TCN

# 记录“模型 __init__ 参数名” 到 “config 中字段名”的映射
# 没写在这里的默认认为两边同名
CONFIG_ALIAS = {
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
    alias_map = CONFIG_ALIAS.get(model_type, {})

    sig = inspect.signature(model_class.__init__)
    params = {}
    sources = {}

    # 分两种情况写清楚一点
    if checkpoint is None:
        # 完全新建模型，所有参数来自 config
        for name, p in sig.parameters.items():
            if name == "self":
                continue

            cfg_attr = alias_map.get(name, name)  # 映射到 config 字段名
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
                cfg_attr = alias_map.get(name, name)
                if hasattr(config, cfg_attr):
                    params[name] = getattr(config, cfg_attr)
                    sources[name] = f"config.{cfg_attr}"
                # 两边都没有就跳过，让 __init__ 用默认值（如果有）

    print(f"Creating {model_type} with parameters:")
    for k, v in params.items():
        print(f"  {k} ({sources.get(k, 'unknown')}): {v}")

    model = model_class(**params).to(device)
    return model

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

    # 解析设备字符串
    device_str = device_str.lower().strip()

    if device_str == "cpu":
        print(f"\n使用设备: CPU")
        return torch.device("cpu")

    # 处理纯数字（如 '0', '1', '2', '3'）
    if device_str.isdigit() or device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print(f"警告: CUDA不可用，回退到CPU")
            return torch.device("cpu")

        # 解析GPU编号
        if device_str == "cuda":
            gpu_id = 0

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

        # 情况3: 如果包含点，尝试将.py前的最后一个点替换为/
        if "." in path_without_py:
            parts = path_without_py.split(".")
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
    # breakpoint()

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
        num_outputs: int
) -> Dict:
    """
    在完整序列上计算指标

    参数:
        estimates_list: List[Tensor[num_outputs, seq_len]] 每个trial的预测序列
        labels_list: List[Tensor[num_outputs, seq_len]] 每个trial的标签序列
        num_outputs: 输出通道数

    返回:
        metrics: 包含每个输出通道指标的字典
    """
    metrics = {}

    for j in range(num_outputs):
        rmse_sum = 0.0
        r2_sum = 0.0
        mae_percent_sum = 0.0
        count = 0

        for est_seq, lbl_seq in zip(estimates_list, labels_list):
            est = est_seq[j]  # [seq_len]
            lbl = lbl_seq[j]  # [seq_len]

            # 创建有效数据掩码
            valid_mask = ~torch.isnan(est) & ~torch.isnan(lbl)

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
        labels: torch.Tensor
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
        trial_names: List[str],
        label_names: List[str],
        action_to_category: Dict[str, str],
        unseen_patterns: List[str]
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

    for est_seq, lbl_seq, trial_name in zip(estimates_list, labels_list, trial_names):
        # 判断类别
        category = categorize_trial(trial_name, action_to_category)

        # 判断是否为未见过的任务
        is_unseen = check_unseen_task(trial_name, unseen_patterns)

        # 对每个输出通道计算指标
        for j, label_name in enumerate(label_names):
            est = est_seq[j]  # [seq_len]
            lbl = lbl_seq[j]  # [seq_len]

            metrics = compute_metrics_per_sequence(est, lbl)

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

if __name__ == '__main__':

    from types import SimpleNamespace

    config = SimpleNamespace()

    # config 对象
    config.input_size = 10
    config.output_size = 5
    config.d_model = 512
    config.transformer_dropout = 0.2

    # checkpoint 内容（可能包含部分覆盖参数）
    checkpoint = {
        'input_size': 12,  # 覆盖config中的值
        # 没有output_size，使用config默认值
        'd_model': 256  # 覆盖config中的值
    }

    # 调用函数
    # model = create_model_from_config('Transformer', None, config, 'cpu')
    model = create_model_from_config('Transformer', checkpoint, config, 'cpu')
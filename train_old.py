import argparse
import os
import shutil
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.config_utils import load_config
from models.predictor_model import PredictorTransformer
from models.generative_model import GenerativeTransformer
from models.tcn import TCN
from dataset_loaders.sequence_dataloader import SequenceDataset
from dataset_loaders.dataloader import TcnDataset
from datetime import datetime
import random
import numpy as np

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs.default_config",
                    help="配置文件路径")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="训练设备 (cpu/cuda)")
parser.add_argument("--resume", type=str, default=None,
                    help="恢复训练的模型路径")
parser.add_argument("--num_workers", type=int, default=4,
                    help="DataLoader的工作进程数（0表示单进程）")
args = parser.parse_args()

# 加载配置
config = load_config(args.config_path)


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
    """复制配置文件到保存目录"""
    config_file_path = config_path.replace(".", os.sep) + ".py"

    if os.path.exists(config_file_path):
        dest_path = os.path.join(save_dir, "config.py")
        shutil.copy(config_file_path, dest_path)
        print(f"配置文件已复制到: {dest_path}")
    else:
        print(f"警告: 配置文件未找到 {config_file_path}")


def create_model(config, device: torch.device, resume_path: str = None) -> Tuple[nn.Module, int]:
    """创建或加载模型"""
    model_type = config.model_type

    if resume_path and os.path.exists(resume_path):
        print(f"从检查点恢复训练: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        start_epoch = checkpoint.get("epoch", 0) + 1

        # 根据模型类型创建模型
        if model_type == 'GenerativeTransformer':
            model = GenerativeTransformer(
                input_size=checkpoint.get("input_size", config.input_size),
                output_size=checkpoint.get("output_size", config.output_size),
                d_model=checkpoint.get("d_model", config.gen_d_model),
                nhead=checkpoint.get("nhead", config.gen_nhead),
                num_encoder_layers=checkpoint.get("num_encoder_layers", config.gen_num_encoder_layers),
                num_decoder_layers=checkpoint.get("num_decoder_layers", config.gen_num_decoder_layers),
                dim_feedforward=checkpoint.get("dim_feedforward", config.gen_dim_feedforward),
                dropout=checkpoint.get("dropout", config.gen_dropout),
                sequence_length=checkpoint.get("sequence_length", config.gen_sequence_length),
                encoder_type=checkpoint.get("encoder_type", config.encoder_type),
                use_positional_encoding=checkpoint.get("use_positional_encoding", config.use_positional_encoding),
                center=checkpoint.get("center", config.center),
                scale=checkpoint.get("scale", config.scale)
            ).to(device)
        elif model_type == 'Transformer':
            model = PredictorTransformer(
                input_size=checkpoint.get("input_size", config.input_size),
                output_size=checkpoint.get("output_size", config.output_size),
                d_model=checkpoint.get("d_model", config.d_model),
                nhead=checkpoint.get("nhead", config.nhead),
                num_encoder_layers=checkpoint.get("num_encoder_layers", config.num_encoder_layers),
                dim_feedforward=checkpoint.get("dim_feedforward", config.dim_feedforward),
                dropout=checkpoint.get("dropout", config.transformer_dropout),
                sequence_length=checkpoint.get("sequence_length", config.sequence_length),
                use_positional_encoding=checkpoint.get("use_positional_encoding", config.use_positional_encoding),
                center=checkpoint.get("center", config.center),
                scale=checkpoint.get("scale", config.scale)
            ).to(device)
        else:  # TCN
            model = TCN(
                input_size=checkpoint.get("input_size", config.input_size),
                output_size=checkpoint.get("output_size", config.output_size),
                num_channels=checkpoint.get("num_channels", config.num_channels),
                ksize=checkpoint.get("ksize", config.ksize),
                dropout=checkpoint.get("dropout", config.dropout),
                eff_hist=checkpoint.get("eff_hist", config.eff_hist),
                spatial_dropout=checkpoint.get("spatial_dropout", config.spatial_dropout),
                activation=checkpoint.get("activation", config.activation),
                norm=checkpoint.get("norm", config.norm),
                center=checkpoint.get("center", config.center),
                scale=checkpoint.get("scale", config.scale)
            ).to(device)

        model.load_state_dict(checkpoint["state_dict"])
        print(f"模型已加载,从第 {start_epoch} 轮继续训练")
    else:
        print(f"创建新的{model_type}模型")
        if model_type == 'GenerativeTransformer':
            model = GenerativeTransformer(
                input_size=config.input_size,
                output_size=config.output_size,
                d_model=config.gen_d_model,
                nhead=config.gen_nhead,
                num_encoder_layers=config.gen_num_encoder_layers,
                num_decoder_layers=config.gen_num_decoder_layers,
                dim_feedforward=config.gen_dim_feedforward,
                dropout=config.gen_dropout,
                sequence_length=config.gen_sequence_length,
                encoder_type=config.encoder_type,
                use_positional_encoding=config.use_positional_encoding,
                center=config.center,
                scale=config.scale
            ).to(device)
        elif model_type == 'Transformer':
            model = PredictorTransformer(
                input_size=config.input_size,
                output_size=config.output_size,
                d_model=config.d_model,
                nhead=config.nhead,
                num_encoder_layers=config.num_encoder_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.transformer_dropout,
                sequence_length=config.sequence_length,
                use_positional_encoding=config.use_positional_encoding,
                center=config.center,
                scale=config.scale
            ).to(device)
        else:  # TCN
            model = TCN(
                input_size=config.input_size,
                output_size=config.output_size,
                num_channels=config.num_channels,
                ksize=config.ksize,
                dropout=config.dropout,
                eff_hist=config.eff_hist,
                spatial_dropout=config.spatial_dropout,
                activation=config.activation,
                norm=config.norm,
                center=config.center,
                scale=config.scale
            ).to(device)
        start_epoch = 0
        print(f"新模型已创建，参数数量: {sum(p.numel() for p in model.parameters()):,}")

    return model, start_epoch


def compute_metrics_batch(estimates: torch.Tensor, labels: torch.Tensor,
                          num_outputs: int, metrics_accumulator: dict) -> None:
    """
    逐batch累积计算评估指标

    参数:
        estimates: [batch_size, num_outputs, seq_len]
        labels: [batch_size, num_outputs, seq_len]
        num_outputs: 输出特征数量
        metrics_accumulator: 累积指标的字典
    """
    batch_size, _, seq_len = estimates.shape

    for i in range(batch_size):
        for j in range(num_outputs):
            estimate = estimates[i, j, :]
            label = labels[i, j, :]

            valid_mask = ~torch.isnan(estimate) & ~torch.isnan(label)
            estimate_valid = estimate[valid_mask]
            label_valid = label[valid_mask]

            if len(estimate_valid) == 0:
                continue

            rmse = torch.sqrt(torch.mean((estimate_valid - label_valid) ** 2))
            ss_res = torch.sum((label_valid - estimate_valid) ** 2)
            ss_tot = torch.sum((label_valid - torch.mean(label_valid)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            label_range = torch.max(label_valid) - torch.min(label_valid)
            mae = torch.mean(torch.abs(estimate_valid - label_valid))
            normalized_mae = mae / (label_range + 1e-8)

            metrics_accumulator[f"output_{j}"]["rmse"] += rmse.item()
            metrics_accumulator[f"output_{j}"]["r2"] += r2.item()
            metrics_accumulator[f"output_{j}"]["normalized_mae"] += normalized_mae.item()
            metrics_accumulator[f"output_{j}"]["count"] += 1


def finalize_metrics(metrics_accumulator: dict, num_outputs: int) -> dict:
    """
    计算累积指标的平均值

    参数:
        metrics_accumulator: 累积的指标字典
        num_outputs: 输出特征数量

    返回:
        metrics: 平均后的指标字典
    """
    for j in range(num_outputs):
        count = metrics_accumulator[f"output_{j}"]["count"]
        if count > 0:
            metrics_accumulator[f"output_{j}"]["rmse"] /= count
            metrics_accumulator[f"output_{j}"]["r2"] /= count
            metrics_accumulator[f"output_{j}"]["normalized_mae"] /= count

    return metrics_accumulator


def validate(model: nn.Module,
             dataloader: DataLoader,
             label_names: List[str],
             device: torch.device,
             model_type: str,
             config) -> Tuple[dict, float]:
    """在验证/测试集上评估模型"""
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0
    num_outputs = len(label_names)

    # 初始化指标累积器
    metrics_accumulator = {
        f"output_{i}": {"rmse": 0.0, "r2": 0.0, "normalized_mae": 0.0, "count": 0}
        for i in range(num_outputs)
    }

    with torch.no_grad():
        if model_type == 'TCN':
            # TCN: 逐batch处理，不concatenate（因为序列长度不同）
            for batch_data in dataloader:
                input_data, label_data, trial_lengths = batch_data
                input_data = input_data.to(device)
                label_data = label_data.to(device)

                estimates = model(input_data)

                # 逐样本计算指标（考虑序列长度和延迟）
                model_history = model.get_effective_history()
                batch_size = estimates.size(0)

                for i in range(batch_size):
                    for j in range(num_outputs):
                        est = estimates[i, j, model_history:trial_lengths[i]]
                        lbl = label_data[i, j, model_history:trial_lengths[i]]

                        if config.model_delays[j] != 0:
                            est = est[config.model_delays[j]:]
                            lbl = lbl[:-config.model_delays[j]]

                        valid_mask = ~torch.isnan(est) & ~torch.isnan(lbl)
                        est_valid = est[valid_mask]
                        lbl_valid = lbl[valid_mask]

                        if len(est_valid) > 0:
                            # 计算RMSE
                            rmse = torch.sqrt(torch.mean((est_valid - lbl_valid) ** 2))

                            # 计算R²
                            ss_res = torch.sum((lbl_valid - est_valid) ** 2)
                            ss_tot = torch.sum((lbl_valid - torch.mean(lbl_valid)) ** 2)
                            r2 = 1 - (ss_res / (ss_tot + 1e-8))

                            # 计算归一化MAE
                            label_range = torch.max(lbl_valid) - torch.min(lbl_valid)
                            mae = torch.mean(torch.abs(est_valid - lbl_valid))
                            normalized_mae = mae / (label_range + 1e-8)

                            # 累积指标
                            metrics_accumulator[f"output_{j}"]["rmse"] += rmse.item()
                            metrics_accumulator[f"output_{j}"]["r2"] += r2.item()
                            metrics_accumulator[f"output_{j}"]["normalized_mae"] += normalized_mae.item()
                            metrics_accumulator[f"output_{j}"]["count"] += 1

                            # 累积损失
                            if j == 0:  # 只计算一次loss
                                loss = criterion(est_valid, lbl_valid)
                                total_loss += loss.item()

                num_batches += batch_size

        else:
            # Transformer模型: 可以concatenate（固定序列长度）
            all_estimates = []
            all_labels = []

            for batch_data in dataloader:
                if model_type == 'GenerativeTransformer':
                    input_data, shifted_label_data, label_data = batch_data
                    input_data = input_data.to(device)
                    shifted_label_data = shifted_label_data.to(device)
                    label_data = label_data.to(device)

                    # 创建因果掩码
                    seq_len = shifted_label_data.size(2)
                    tgt_mask = GenerativeTransformer._generate_square_subsequent_mask(seq_len).to(device)

                    # 使用teacher forcing
                    estimates = model(input_data, shifted_label_data, tgt_mask)

                else:  # Transformer预测模型
                    input_data, label_data = batch_data
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)

                    estimates = model(input_data)

                all_estimates.append(estimates)
                all_labels.append(label_data)

                # 计算损失
                valid_mask = ~torch.isnan(estimates) & ~torch.isnan(label_data)
                if valid_mask.sum() > 0:
                    loss = criterion(estimates[valid_mask], label_data[valid_mask])
                    total_loss += loss.item()
                    num_batches += 1

            # Concatenate并计算指标
            all_estimates = torch.cat(all_estimates, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 逐batch累积计算指标
            compute_metrics_batch(all_estimates, all_labels, num_outputs, metrics_accumulator)

    # 计算平均损失
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 计算平均指标
    metrics = finalize_metrics(metrics_accumulator, num_outputs)

    # 构建结果字典
    result_dict = {}
    for j, label_name in enumerate(label_names):
        result_dict[label_name] = metrics[f"output_{j}"]

    model.train()
    return result_dict, avg_loss


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

    if config.model_type == 'GenerativeTransformer':
        checkpoint.update({
            "d_model": config.gen_d_model,
            "nhead": config.gen_nhead,
            "num_encoder_layers": config.gen_num_encoder_layers,
            "num_decoder_layers": config.gen_num_decoder_layers,
            "dim_feedforward": config.gen_dim_feedforward,
            "dropout": config.gen_dropout,
            "sequence_length": config.gen_sequence_length,
            "encoder_type": config.encoder_type,
            "use_positional_encoding": config.use_positional_encoding
        })
    elif config.model_type == 'Transformer':
        checkpoint.update({
            "d_model": config.d_model,
            "nhead": config.nhead,
            "num_encoder_layers": config.num_encoder_layers,
            "dim_feedforward": config.dim_feedforward,
            "dropout": config.transformer_dropout,
            "sequence_length": config.sequence_length,
            "use_positional_encoding": config.use_positional_encoding
        })
    else:  # TCN
        checkpoint.update({
            "num_channels": config.num_channels,
            "ksize": config.ksize,
            "dropout": config.dropout,
            "eff_hist": config.eff_hist,
            "spatial_dropout": config.spatial_dropout,
            "activation": config.activation,
            "norm": config.norm
        })

    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.tar")
    torch.save(checkpoint, save_path)
    print(f"模型已保存: {save_path}")


def log_to_file(file_path: str, content: str):
    """将内容追加到日志文件"""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content + "\n")


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


def main():
    # 设置随机种子
    set_seed(config.random_seed)

    device = torch.device(args.device)
    print(f"使用设备: {device}")
    print(f"模型类型: {config.model_type}")
    print(f"DataLoader工作进程数: {args.num_workers}")

    save_dir = create_save_directory(args.config_path, config.model_type)
    copy_config_file(args.config_path, save_dir)

    train_log_path = os.path.join(save_dir, "train_log.txt")
    val_log_path = os.path.join(save_dir, "validation_log.txt")

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(train_log_path, f"训练开始时间: {start_time}")
    log_to_file(train_log_path, f"配置文件: {args.config_path}")
    log_to_file(train_log_path, f"模型类型: {config.model_type}")
    log_to_file(train_log_path, f"随机种子: {config.random_seed}")
    log_to_file(train_log_path, f"设备: {device}\n")

    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    # 根据模型类型选择数据加载器和collate函数
    if config.model_type in ['Transformer', 'GenerativeTransformer']:
        seq_len = config.gen_sequence_length if config.model_type == 'GenerativeTransformer' else config.sequence_length

        print(f"加载{config.model_type}训练数据集...")
        train_dataset = SequenceDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            sequence_length=seq_len,
            model_delays=config.model_delays,
            participant_masses=config.participant_masses,
            device=device,
            mode='train',
            model_type=config.model_type,
            start_token_value=config.start_token_value if config.model_type == 'GenerativeTransformer' else 0.0,
            remove_nan=True
        )

        print(f"加载{config.model_type}测试数据集...")
        test_dataset = SequenceDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            sequence_length=seq_len,
            model_delays=config.model_delays,
            participant_masses=config.participant_masses,
            device=device,
            mode='test',
            model_type=config.model_type,
            start_token_value=config.start_token_value if config.model_type == 'GenerativeTransformer' else 0.0,
            remove_nan=True
        )

        collate_fn = collate_fn_generative if config.model_type == 'GenerativeTransformer' else collate_fn_predictor
    else:  # TCN
        print("加载TCN训练数据集...")
        train_dataset = TcnDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            participant_masses=config.participant_masses,
            device=device,
            mode='train',
            load_to_device=False
        )

        print("加载TCN测试数据集...")
        test_dataset = TcnDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            participant_masses=config.participant_masses,
            device=device,
            mode='test',
            load_to_device=False
        )

        collate_fn = collate_fn_tcn

    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 and device.type == 'cuda' else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 and device.type == 'cuda' else False
    )

    model, start_epoch = create_model(config, device, args.resume)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        verbose=True,
        min_lr=config.min_lr
    )

    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        mode='min'
    )

    criterion = nn.MSELoss()

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])

    print(f"\n开始训练 (共 {config.num_epochs} 轮)...\n")
    best_val_loss = float('inf')

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch_data in enumerate(train_loader):
            if config.model_type == 'GenerativeTransformer':
                input_data, shifted_label_data, label_data = batch_data
                input_data = input_data.to(device)
                shifted_label_data = shifted_label_data.to(device)
                label_data = label_data.to(device)

                # 创建因果掩码
                seq_len = shifted_label_data.size(2)
                tgt_mask = GenerativeTransformer._generate_square_subsequent_mask(seq_len).to(device)

                # 前向传播
                estimates = model(input_data, shifted_label_data, tgt_mask)

                # 计算损失
                valid_mask = ~torch.isnan(estimates) & ~torch.isnan(label_data)
                if valid_mask.sum() > 0:
                    loss = criterion(estimates[valid_mask], label_data[valid_mask])

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

            elif config.model_type == 'Transformer':
                input_data, label_data = batch_data
                input_data = input_data.to(device)
                label_data = label_data.to(device)

                estimates = model(input_data)

                valid_mask = ~torch.isnan(estimates) & ~torch.isnan(label_data)
                if valid_mask.sum() > 0:
                    loss = criterion(estimates[valid_mask], label_data[valid_mask])

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

            else:  # TCN
                input_data, label_data, trial_lengths = batch_data
                input_data = input_data.to(device)
                label_data = label_data.to(device)

                estimates = model(input_data)
                batch_losses = []
                model_history = model.get_effective_history()

                for i in range(len(trial_lengths)):
                    for j in range(config.output_size):
                        est = estimates[i, j, model_history:trial_lengths[i]]
                        lbl = label_data[i, j, model_history:trial_lengths[i]]

                        if config.model_delays[j] != 0:
                            est = est[config.model_delays[j]:]
                            lbl = lbl[:-config.model_delays[j]]

                        valid_mask = ~torch.isnan(est) & ~torch.isnan(lbl)
                        est_valid = est[valid_mask]
                        lbl_valid = lbl[valid_mask]

                        if len(est_valid) > 0:
                            batch_losses.append(criterion(est_valid, lbl_valid))

                if len(batch_losses) > 0:
                    loss = torch.stack(batch_losses).mean()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{config.num_epochs}] - 训练损失: {avg_loss:.6f}, 学习率: {current_lr:.6e}")
        log_to_file(train_log_path, f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}, LR = {current_lr:.6e}")

        # 验证
        if (epoch + 1) % config.val_interval == 0:
            print(f"\n在测试集上进行验证 (Epoch {epoch + 1})...")
            val_metrics, val_loss = validate(model, test_loader, label_names, device, config.model_type, config)
            scheduler.step(val_loss)

            val_log_content = f"\n=== Epoch {epoch + 1} 验证结果 ===\n"
            val_log_content += f"验证损失: {val_loss:.6f}\n"
            print(val_log_content.strip())

            for label_name, metrics in val_metrics.items():
                result_str = (f"{label_name}:\n"
                              f"  RMSE: {metrics['rmse']:.4f} Nm/kg\n"
                              f"  R²: {metrics['r2']:.4f}\n"
                              f"  归一化MAE: {metrics['normalized_mae']:.4f}")
                print(result_str)
                val_log_content += result_str + "\n"

            log_to_file(val_log_path, val_log_content)
            print()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, "best_model.tar")
                save_model(model, save_dir, epoch + 1, config, optimizer, scheduler)
                shutil.copy(os.path.join(save_dir, f"model_epoch_{epoch + 1}.tar"), best_model_path)
                print(f"最佳模型已保存! 验证损失: {val_loss:.6f}")
                log_to_file(val_log_path, f"最佳模型更新 - Epoch {epoch + 1}, 验证损失: {val_loss:.6f}\n")

            if early_stopping(val_loss):
                print(f"\n早停触发! 在Epoch {epoch + 1}停止训练")
                log_to_file(train_log_path, f"\n早停触发 - Epoch {epoch + 1}")
                log_to_file(train_log_path, f"最佳验证损失: {best_val_loss:.6f}")
                break

        if (epoch + 1) % config.save_interval == 0:
            save_model(model, save_dir, epoch + 1, config, optimizer, scheduler)

    print("\n训练完成,保存最终模型...")
    save_model(model, save_dir, epoch + 1, config, optimizer, scheduler)

    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(train_log_path, f"\n训练结束时间: {end_time}")
    log_to_file(train_log_path, f"最佳验证损失: {best_val_loss:.6f}")

    print(f"\n所有结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()
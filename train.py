import argparse
import os
import shutil
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from utils.config_utils import *
from utils.utils import *
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
                    help="训练设备 (cpu, cuda, 0, 1, 2, 3, ...)")
parser.add_argument("--resume", type=str, default=None,
                    help="恢复训练的模型路径")
parser.add_argument("--num_workers", type=int, default=16,
                    help="DataLoader的工作进程数（0表示单进程）")
args = parser.parse_args()

# 加载配置
config = load_config(args.config_path)
config = apply_feature_selection(config)

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


def validate(model: nn.Module,
             dataloader: DataLoader,
             label_names: List[str],
             device: torch.device,
             model_type: str,
             config,
             reconstruction_method: str = "only_first") -> Tuple[dict, float]:
    """
    验证函数 - 在完整序列上计算指标

    参数:
        reconstruction_method: 'only_first' 或 'average'
    """
    model.eval()
    criterion = nn.MSELoss()

    total_loss = 0.0
    num_batches = 0
    num_outputs = len(label_names)

    with torch.no_grad():
        if model_type == 'TCN':
            # TCN: 逐试验处理长序列
            # 初始化指标累积器
            metrics_accumulator = {
                f"output_{i}": {"rmse": 0.0, "r2": 0.0, "mae_percent": 0.0, "count": 0}
                for i in range(num_outputs)
            }

            for batch_data in dataloader:
                input_data, label_data, trial_lengths = batch_data
                input_data = input_data.to(device)
                label_data = label_data.to(device)

                estimates = model(input_data)

                model_history = model.get_effective_history()
                batch_size = estimates.size(0)

                for i in range(batch_size):
                    est_trial = estimates[i, :, model_history:trial_lengths[i]]
                    lbl_trial = label_data[i, :, model_history:trial_lengths[i]]

                    # 找到最大delay，以此确定所有通道的统一有效长度
                    max_delay = max(config.model_delays)
                    valid_length = est_trial.size(1) - max_delay

                    if valid_length <= 0:
                        # 如果序列太短，跳过这个trial
                        continue

                    # 应用延迟并统一长度
                    valid_masks = []
                    est_shifted = []
                    lbl_shifted = []

                    for j in range(num_outputs):
                        delay = config.model_delays[j]

                        # 预测值：从max_delay位置开始取，确保所有通道对齐
                        est_j = est_trial[j, max_delay:]

                        # 标签：根据当前通道的delay进行偏移
                        # delay较小的通道，标签需要从更靠后的位置开始
                        lbl_start = max_delay - delay
                        lbl_end = lbl_start + valid_length
                        lbl_j = lbl_trial[j, lbl_start:lbl_end]

                        valid_mask_j = ~torch.isnan(est_j) & ~torch.isnan(lbl_j)

                        est_shifted.append(est_j)
                        lbl_shifted.append(lbl_j)
                        valid_masks.append(valid_mask_j)

                    est_stacked = torch.stack(est_shifted)
                    lbl_stacked = torch.stack(lbl_shifted)
                    mask_stacked = torch.stack(valid_masks)

                    trial_metrics = compute_metrics_tcn_trial(
                        est_stacked, lbl_stacked, mask_stacked, num_outputs
                    )

                    # 累积指标
                    for j in range(num_outputs):
                        if trial_metrics[f"output_{j}"]["count"] > 0:
                            metrics_accumulator[f"output_{j}"]["rmse"] += trial_metrics[f"output_{j}"]["rmse"]
                            metrics_accumulator[f"output_{j}"]["r2"] += trial_metrics[f"output_{j}"]["r2"]
                            metrics_accumulator[f"output_{j}"]["mae_percent"] += trial_metrics[f"output_{j}"][
                                "mae_percent"]
                            metrics_accumulator[f"output_{j}"]["count"] += 1

                    # 计算损失
                    valid_data = est_stacked[mask_stacked]
                    valid_labels = lbl_stacked[mask_stacked]
                    if len(valid_data) > 0:
                        loss = criterion(valid_data, valid_labels)
                        total_loss += loss.item()
                        num_batches += 1

            # 计算平均指标
            for j in range(num_outputs):
                count = metrics_accumulator[f"output_{j}"]["count"]
                if count > 0:
                    metrics_accumulator[f"output_{j}"]["rmse"] /= count
                    metrics_accumulator[f"output_{j}"]["r2"] /= count
                    metrics_accumulator[f"output_{j}"]["mae_percent"] /= count

        else:
            # Transformer: 收集所有预测和标签，然后重组序列
            all_estimates = []
            all_labels = []

            for batch_idx, batch_data in enumerate(dataloader):
                if model_type == 'GenerativeTransformer':
                    input_data, shifted_label_data, label_data = batch_data
                    input_data = input_data.to(device)
                    shifted_label_data = shifted_label_data.to(device)
                    label_data = label_data.to(device)

                    seq_len = shifted_label_data.size(2)
                    tgt_mask = GenerativeTransformer._generate_square_subsequent_mask(seq_len).to(device)

                    estimates = model(input_data, shifted_label_data, tgt_mask)

                else:
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

            # Concatenate所有批次的结果
            all_estimates = torch.cat(all_estimates, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # 使用优化的重组方法
            print(f"使用优化的 '{reconstruction_method}' 方法重组序列...")
            reconstructed_estimates, reconstructed_labels = reconstruct_sequences(
                all_estimates, all_labels,
                dataloader.dataset.trial_sequence_counts,
                method=reconstruction_method
            )

            # 在完整序列上计算指标
            metrics_accumulator = compute_metrics_on_sequences(
                reconstructed_estimates, reconstructed_labels, num_outputs
            )

    # 计算平均
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 构建结果字典
    result_dict = {}
    for j, label_name in enumerate(label_names):
        result_dict[label_name] = {
            "rmse": metrics_accumulator[f"output_{j}"]["rmse"],
            "r2": metrics_accumulator[f"output_{j}"]["r2"],
            "mae_percent": metrics_accumulator[f"output_{j}"]["mae_percent"]
        }

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
            "output_sequence_length":config.output_sequence_length,
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

    # 设置并验证设备
    device = setup_device(args.device)
    print(f"模型类型: {config.model_type}")

    # 获取reconstruction_method（仅对Transformer模型有效）
    reconstruction_method = getattr(config, 'reconstruction_method', 'only_first')
    if config.model_type in ['Transformer', 'GenerativeTransformer']:
        print(f"序列重组方法: {reconstruction_method}")

    # 创建保存目录
    save_dir = create_save_directory(args.config_path, config.model_type)
    copy_config_file(args.config_path, save_dir)

    # 创建日志文件
    train_log_path = os.path.join(save_dir, "train_log.txt")
    val_log_path = os.path.join(save_dir, "val_log.txt")

    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_to_file(train_log_path, f"训练开始时间: {start_time}")
    log_to_file(train_log_path, f"模型类型: {config.model_type}")
    log_to_file(train_log_path, f"设备: {device}")
    log_to_file(train_log_path, f"随机种子: {config.random_seed}")
    if config.model_type in ['Transformer', 'GenerativeTransformer']:
        log_to_file(train_log_path, f"序列重组方法: {reconstruction_method}")
    log_to_file(train_log_path, f"{'=' * 60}")

    # 创建模型
    model, start_epoch = create_model(config, device, args.resume)

    # 替换配置中的通配符
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    # 根据模型类型加载数据集
    if config.model_type in ["Transformer", "GenerativeTransformer"]:
        seq_len = config.gen_sequence_length if config.model_type == "GenerativeTransformer" else config.sequence_length
        output_seq_len = getattr(config, 'output_sequence_length', seq_len)

        print(f"\n加载{config.model_type}训练数据集...")
        train_dataset = SequenceDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            sequence_length=seq_len,
            output_sequence_length=output_seq_len,
            model_delays=config.model_delays,
            participant_masses=config.participant_masses,
            device=device,
            mode='train',
            model_type=config.model_type,
            start_token_value=config.start_token_value if config.model_type == "GenerativeTransformer" else 0.0,
            remove_nan=True,
            action_patterns=getattr(config, 'action_patterns', None),
            enable_action_filter=getattr(config, 'enable_action_filter', False)
        )

        print(f"\n加载{config.model_type}测试数据集...")
        test_dataset = SequenceDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            sequence_length=seq_len,
            output_sequence_length=output_seq_len,
            model_delays=config.model_delays,
            participant_masses=config.participant_masses,
            device=device,
            mode='test',
            model_type=config.model_type,
            start_token_value=config.start_token_value if config.model_type == "GenerativeTransformer" else 0.0,
            remove_nan=True,
            action_patterns=getattr(config, 'action_patterns', None),
            enable_action_filter=getattr(config, 'enable_action_filter', False)
        )

        collate_fn = collate_fn_generative if config.model_type == "GenerativeTransformer" else collate_fn_predictor

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

        test_batch_size = getattr(config, 'test_batch_size', config.batch_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )

    else:  # TCN
        print(f"\n加载TCN训练数据集...")
        train_dataset = TcnDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            participant_masses=config.participant_masses,
            device=device,
            mode='train',
            load_to_device=False,
            action_patterns=getattr(config, 'action_patterns', None),
            enable_action_filter=getattr(config, 'enable_action_filter', False)
        )

        print(f"\n加载TCN测试数据集...")
        test_dataset = TcnDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            participant_masses=config.participant_masses,
            device=device,
            mode='test',
            load_to_device=False,
            action_patterns=getattr(config, 'action_patterns', None),
            enable_action_filter=getattr(config, 'enable_action_filter', False)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn_tcn,
            num_workers=args.num_workers
        )

        test_batch_size = getattr(config, 'test_batch_size', config.batch_size)
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=collate_fn_tcn,
            num_workers=args.num_workers
        )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建优化器和调度器
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

    # ========== 训练前进行初始验证 ==========
    print("=" * 60)
    print("训练前初始验证 (Epoch 0)...")
    print("=" * 60)
    initial_metrics, initial_loss = validate(
        model, test_loader, label_names, device,
        config.model_type, config, reconstruction_method
    )

    initial_log_content = f"\n{'=' * 60}\n"
    initial_log_content += "=== 训练前初始验证结果 (Epoch 0) ===\n"
    initial_log_content += f"{'=' * 60}\n"
    initial_log_content += f"验证损失: {initial_loss:.6f}\n"
    print(f"\n验证损失: {initial_loss:.6f}")

    for label_name, metrics in initial_metrics.items():
        result_str = (f"\n{label_name}:\n"
                      f"  RMSE: {metrics['rmse']:.4f} Nm/kg\n"
                      f"  R²: {metrics['r2']:.4f}\n"
                      f"  MAE: {metrics['mae_percent']:.2f}%")
        print(result_str)
        initial_log_content += result_str + "\n"

    log_to_file(val_log_path, initial_log_content)
    print("=" * 60)
    print()
    # ========== 初始验证结束 ==========

    for epoch in range(start_epoch, config.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # ========== 训练批次限制 ==========
        train_batch_ratio = getattr(config, 'train_batch_ratio', 1.0)

        # 只对Transformer类型的模型应用batch限制
        if config.model_type in ['Transformer', 'GenerativeTransformer'] and train_batch_ratio < 1.0:
            max_batches = max(1, int(len(train_loader) * train_batch_ratio))
            if epoch == start_epoch:  # 只在第一个epoch打印一次
                print(f"\n训练批次限制已启用:")
                print(f"  总批次数: {len(train_loader)}")
                print(f"  每轮使用批次数: {max_batches} ({train_batch_ratio * 100:.1f}%)")
        else:
            max_batches = len(train_loader)
        # ========================================

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

                    # 检查是否达到最大batch数
                    if num_batches >= max_batches:
                        break

            elif config.model_type == 'Transformer':
                input_data, label_data = batch_data
                input_data = input_data.to(device)
                label_data = label_data.to(device)

                estimates = model(input_data)

                valid_mask = ~torch.isnan(estimates) & ~torch.isnan(label_data)
                if valid_mask.sum() > 0:
                    # breakpoint()
                    loss = criterion(estimates[valid_mask], label_data[valid_mask])

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    # 检查是否达到最大batch数
                    if num_batches >= max_batches:
                        break

            else:  # TCN
                input_data, label_data, trial_lengths = batch_data
                input_data = input_data.to(device)
                label_data = label_data.to(device)
                estimates = model(input_data)
                batch_losses = []
                model_history = model.get_effective_history()

                for i in range(len(trial_lengths)):
                    est_trial = estimates[i, :, model_history:trial_lengths[i]]
                    lbl_trial = label_data[i, :, model_history:trial_lengths[i]]
                    # breakpoint()
                    # 找到最大delay
                    max_delay = max(config.model_delays)
                    valid_length = est_trial.size(1) - max_delay

                    if valid_length <= 0:
                        # 如果序列太短，跳过这个trial
                        continue

                    for j in range(config.output_size):
                        delay = config.model_delays[j]

                        # 预测值：从max_delay位置开始取
                        est = est_trial[j, max_delay:]

                        # 标签：根据当前通道的delay进行偏移
                        lbl_start = max_delay - delay
                        lbl_end = lbl_start + valid_length
                        lbl = lbl_trial[j, lbl_start:lbl_end]

                        valid_mask = ~torch.isnan(est) & ~torch.isnan(lbl)
                        est_valid = est[valid_mask]
                        lbl_valid = lbl[valid_mask]

                        if len(est_valid) > 0:
                            if torch.isnan(criterion(est_valid, lbl_valid)):
                                print(f"NaN at trial={i}, channel={j}")
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
            val_metrics, val_loss = validate(
                model, test_loader, label_names, device,
                config.model_type, config, reconstruction_method
            )
            scheduler.step(val_loss)

            val_log_content = f"\n=== Epoch {epoch + 1} 验证结果 ===\n"
            val_log_content += f"验证损失: {val_loss:.6f}\n"
            print(val_log_content.strip())

            for label_name, metrics in val_metrics.items():
                result_str = (f"{label_name}:\n"
                              f"  RMSE: {metrics['rmse']:.4f} Nm/kg\n"
                              f"  R²: {metrics['r2']:.4f}\n"
                              f"  MAE: {metrics['mae_percent']:.2f}%")
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

    # ========== 评估最佳模型 ==========
    print("\n" + "=" * 60)
    print("加载并评估最佳模型...")
    print("=" * 60)

    best_model_path = os.path.join(save_dir, "best_model.tar")
    if os.path.exists(best_model_path):
        # 加载最佳模型
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # 在测试集上评估
        best_metrics, best_loss = validate(
            model, test_loader, label_names, device,
            config.model_type, config, reconstruction_method
        )

        # 输出并记录最佳模型性能
        best_log_content = f"\n{'=' * 60}\n"
        best_log_content += "=== 最佳模型性能评估 ===\n"
        best_log_content += f"{'=' * 60}\n"
        best_log_content += f"模型路径: {best_model_path}\n"
        best_log_content += f"验证损失: {best_loss:.6f}\n"

        print(f"\n最佳模型验证损失: {best_loss:.6f}")

        for label_name, metrics in best_metrics.items():
            result_str = (f"\n{label_name}:\n"
                          f"  RMSE: {metrics['rmse']:.4f} Nm/kg\n"
                          f"  R²: {metrics['r2']:.4f}\n"
                          f"  MAE: {metrics['mae_percent']:.2f}%")
            print(result_str)
            best_log_content += result_str + "\n"

        best_log_content += f"{'=' * 60}\n"

        # 记录到两个日志文件
        log_to_file(train_log_path, best_log_content)
        log_to_file(val_log_path, best_log_content)

        print("=" * 60)
    else:
        print(f"⚠ 警告: 未找到最佳模型文件 {best_model_path}")

    # ========== 评估结束 ==========

    print(f"\n所有结果已保存到: {save_dir}")


if __name__ == "__main__":
    main()
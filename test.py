import argparse
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.config_utils import load_config
from models.predictor_model import PredictorTransformer
from models.generative_model import GenerativeTransformer
from models.tcn import TCN
from dataset_loaders.sequence_dataloader import SequenceDataset
from dataset_loaders.dataloader import TcnDataset

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs.default_config",
                    help="配置文件路径")
parser.add_argument("--model_path", type=str, required=True,
                    help="训练好的模型路径")
parser.add_argument("--device", type=str, default="cpu",
                    help="设备 (cpu/cuda)")
parser.add_argument("--batch_size", type=int, default=32,
                    help="批次大小")
parser.add_argument("--use_generation", action="store_true",
                    help="生成式模型是否使用自回归生成模式（默认使用teacher forcing）")
args = parser.parse_args()

# 加载配置
config = load_config(args.config_path)


def load_model(model_path: str, device: torch.device):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_type = checkpoint.get("model_type", "TCN")
    print(f"模型类型: {model_type}")

    if model_type == "GenerativeTransformer":
        model = GenerativeTransformer(
            input_size=checkpoint["input_size"],
            output_size=checkpoint["output_size"],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_encoder_layers=checkpoint["num_encoder_layers"],
            num_decoder_layers=checkpoint["num_decoder_layers"],
            dim_feedforward=checkpoint["dim_feedforward"],
            dropout=checkpoint["dropout"],
            sequence_length=checkpoint["sequence_length"],
            encoder_type=checkpoint["encoder_type"],
            use_positional_encoding=checkpoint["use_positional_encoding"],
            center=checkpoint["center"],
            scale=checkpoint["scale"]
        ).to(device)
    elif model_type == "Transformer":
        model = PredictorTransformer(
            input_size=checkpoint["input_size"],
            output_size=checkpoint["output_size"],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_encoder_layers=checkpoint["num_encoder_layers"],
            dim_feedforward=checkpoint["dim_feedforward"],
            dropout=checkpoint["dropout"],
            sequence_length=checkpoint["sequence_length"],
            use_positional_encoding=checkpoint["use_positional_encoding"],
            center=checkpoint["center"],
            scale=checkpoint["scale"]
        ).to(device)
    else:  # TCN
        model = TCN(
            input_size=checkpoint["input_size"],
            output_size=checkpoint["output_size"],
            num_channels=checkpoint["num_channels"],
            ksize=checkpoint["ksize"],
            dropout=checkpoint["dropout"],
            eff_hist=checkpoint["eff_hist"],
            spatial_dropout=checkpoint["spatial_dropout"],
            activation=checkpoint["activation"],
            norm=checkpoint["norm"],
            center=checkpoint["center"],
            scale=checkpoint["scale"]
        ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    print(f"模型加载成功! Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, model_type


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


def compute_metrics_on_sequences(
        estimates_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        label_names: List[str]
) -> Dict:
    """
    在完整序列上计算指标

    参数:
        estimates_list: List[Tensor[num_outputs, seq_len]] 每个trial的预测序列
        labels_list: List[Tensor[num_outputs, seq_len]] 每个trial的标签序列
        label_names: 标签名称列表

    返回:
        metrics: 包含每个输出通道指标的字典
    """
    num_outputs = len(label_names)
    total_metrics = {
        label_name: {"rmse": 0.0, "r2": 0.0, "mae_percent": 0.0, "count": 0}
        for label_name in label_names
    }

    for j, label_name in enumerate(label_names):
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

            total_metrics[label_name]["rmse"] += rmse.item()
            total_metrics[label_name]["r2"] += r2.item()
            total_metrics[label_name]["mae_percent"] += mae_percent.item()
            total_metrics[label_name]["count"] += 1

    # 计算平均
    for label_name in label_names:
        count = total_metrics[label_name]["count"]
        if count > 0:
            total_metrics[label_name]["rmse"] /= count
            total_metrics[label_name]["r2"] /= count
            total_metrics[label_name]["mae_percent"] /= count

    return total_metrics


def print_results(metrics: Dict):
    """打印测试结果"""
    print("\n" + "=" * 60)
    print("测试结果 (在完整序列上计算):")
    print("=" * 60)

    for label_name, metric_values in metrics.items():
        count = metric_values["count"]
        if count > 0:
            print(f"\n{label_name}:")
            print(f"  RMSE: {metric_values['rmse']:.4f} Nm/kg")
            print(f"  R²: {metric_values['r2']:.4f}")
            print(f"  MAE: {metric_values['mae_percent']:.2f}%")
            print(f"  有效试验数: {count}")
        else:
            print(f"\n{label_name}: 无有效数据")

    print("=" * 60 + "\n")


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


def main():
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 加载模型
    model, model_type = load_model(args.model_path, device)
    model.eval()

    print(f"模型类型: {model_type}")

    # 获取reconstruction_method（仅对Transformer模型有效）
    reconstruction_method = getattr(config, 'reconstruction_method', 'only_first')
    if model_type in ['Transformer', 'GenerativeTransformer']:
        print(f"序列重组方法: {reconstruction_method}")

    # 替换配置中的通配符
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    # 根据模型类型加载数据集
    if model_type in ["Transformer", "GenerativeTransformer"]:
        seq_len = config.gen_sequence_length if model_type == "GenerativeTransformer" else config.sequence_length
        output_seq_len = getattr(config, 'output_sequence_length', seq_len)

        print(f"\n加载{model_type}测试数据集...")
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
            model_type=model_type,
            start_token_value=config.start_token_value if model_type == "GenerativeTransformer" else 0.0,
            remove_nan=True
        )

        collate_fn = collate_fn_generative if model_type == "GenerativeTransformer" else collate_fn_predictor

        # 使用配置中的测试批次大小
        test_batch_size = getattr(config, 'test_batch_size', args.batch_size)

        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        print(f"测试集大小: {len(test_dataset)} 个序列")
        print(f"测试批次数: {len(test_loader)}")

        # 进行测试 - 收集所有预测和标签
        all_estimates = []
        all_labels = []
        trial_info = []  # (trial_idx, input_start_idx, seq_len)

        print("\n开始预测...")
        with torch.no_grad():
            if model_type == "GenerativeTransformer":
                if args.use_generation:
                    print("使用自回归生成模式...")
                    for batch_idx, batch_data in enumerate(test_loader):
                        input_data, _, label_data = batch_data
                        input_data = input_data.to(device)
                        label_data = label_data.to(device)

                        # 自回归生成
                        estimates = model.generate(input_data, max_len=seq_len)

                        all_estimates.append(estimates)
                        all_labels.append(label_data)

                        # 记录trial信息
                        batch_size = estimates.size(0)
                        for i in range(batch_size):
                            idx = batch_idx * test_loader.batch_size + i
                            if idx < len(test_dataset):
                                trial_idx, input_start_idx = test_dataset.sequences[idx]
                                trial_info.append((trial_idx, input_start_idx, config.sequence_length))
                else:
                    print("使用Teacher Forcing模式...")
                    for batch_idx, batch_data in enumerate(test_loader):
                        input_data, shifted_label_data, label_data = batch_data
                        input_data = input_data.to(device)
                        shifted_label_data = shifted_label_data.to(device)
                        label_data = label_data.to(device)

                        # 创建因果掩码
                        tgt_mask = GenerativeTransformer._generate_square_subsequent_mask(seq_len).to(device)

                        # Teacher forcing
                        estimates = model(input_data, shifted_label_data, tgt_mask)

                        all_estimates.append(estimates)
                        all_labels.append(label_data)

                        # 记录trial信息
                        batch_size = estimates.size(0)
                        for i in range(batch_size):
                            idx = batch_idx * test_loader.batch_size + i
                            if idx < len(test_dataset):
                                trial_idx, input_start_idx = test_dataset.sequences[idx]
                                trial_info.append((trial_idx, input_start_idx, config.sequence_length))
            else:  # Transformer预测模型
                for batch_idx, (input_data, label_data) in enumerate(test_loader):
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)

                    estimates = model(input_data)

                    all_estimates.append(estimates)
                    all_labels.append(label_data)

                    # 记录trial信息
                    batch_size = estimates.size(0)
                    for i in range(batch_size):
                        idx = batch_idx * test_loader.batch_size + i
                        if idx < len(test_dataset):
                            trial_idx, input_start_idx = test_dataset.sequences[idx]
                            trial_info.append((trial_idx, input_start_idx, config.sequence_length))

        # 合并所有批次的结果
        all_estimates = torch.cat(all_estimates, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        print(f"预测完成! 共 {all_estimates.size(0)} 个短序列")

        # 重组序列
        print(f"\n使用 '{reconstruction_method}' 方法重组序列...")
        reconstructed_estimates, reconstructed_labels = reconstruct_sequences_from_predictions(
            all_estimates, all_labels, trial_info, method=reconstruction_method
        )

        print(f"重组完成! 共 {len(reconstructed_estimates)} 个完整序列")

        # 在完整序列上计算指标
        print("\n计算指标...")
        metrics = compute_metrics_on_sequences(
            reconstructed_estimates, reconstructed_labels, label_names
        )

        # 打印结果
        print_results(metrics)

    else:  # TCN
        print(f"\n加载TCN测试数据集...")
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

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # TCN使用batch_size=1避免长度不一致
            shuffle=False,
            collate_fn=collate_fn_tcn
        )

        print(f"测试集大小: {len(test_dataset)} 个试验")

        # 进行测试 - 逐个处理
        total_metrics = {
            label_name: {"rmse": 0.0, "r2": 0.0, "mae_percent": 0.0, "count": 0}
            for label_name in label_names
        }

        print("\n开始预测...")
        with torch.no_grad():
            for input_data, label_data, trial_lengths in test_loader:
                input_data = input_data.to(device)
                label_data = label_data.to(device)

                estimates = model(input_data)

                # 逐样本计算指标
                model_history = model.get_effective_history()
                batch_size = estimates.size(0)

                for i in range(batch_size):
                    for j, label_name in enumerate(label_names):
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

                            # 计算归一化MAE as percentage
                            label_range = torch.max(lbl_valid) - torch.min(lbl_valid)
                            mae = torch.mean(torch.abs(est_valid - lbl_valid))
                            mae_percent = (mae / (label_range + 1e-8)) * 100.0

                            # 累积指标
                            total_metrics[label_name]["rmse"] += rmse.item()
                            total_metrics[label_name]["r2"] += r2.item()
                            total_metrics[label_name]["mae_percent"] += mae_percent.item()
                            total_metrics[label_name]["count"] += 1

        # 计算平均并打印结果
        for label_name in label_names:
            count = total_metrics[label_name]["count"]
            if count > 0:
                total_metrics[label_name]["rmse"] /= count
                total_metrics[label_name]["r2"] /= count
                total_metrics[label_name]["mae_percent"] /= count

        print_results(total_metrics)


if __name__ == "__main__":
    main()
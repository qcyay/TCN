import argparse
from typing import List
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


def print_results(label_names: List[str],
                  estimates: torch.FloatTensor,
                  labels: torch.FloatTensor):
    """打印测试结果"""
    batch_size, num_outputs, seq_len = estimates.shape

    # 初始化累积指标
    total_metrics = {
        label_name: {"rmse": 0.0, "r2": 0.0, "normalized_mae": 0.0, "count": 0}
        for label_name in label_names
    }

    # 逐样本计算指标
    for i in range(batch_size):
        for j, label_name in enumerate(label_names):
            estimate = estimates[i, j, :]
            label = labels[i, j, :]

            # 忽略NaN值
            valid_mask = ~torch.isnan(estimate) & ~torch.isnan(label)
            estimate_valid = estimate[valid_mask]
            label_valid = label[valid_mask]

            if len(estimate_valid) == 0:
                continue

            # 计算RMSE
            rmse = torch.sqrt(torch.mean((estimate_valid - label_valid) ** 2))

            # 计算R²
            ss_res = torch.sum((label_valid - estimate_valid) ** 2)
            ss_tot = torch.sum((label_valid - torch.mean(label_valid)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))

            # 计算归一化MAE
            label_range = torch.max(label_valid) - torch.min(label_valid)
            mae = torch.mean(torch.abs(estimate_valid - label_valid))
            normalized_mae = mae / (label_range + 1e-8)

            total_metrics[label_name]["rmse"] += rmse.item()
            total_metrics[label_name]["r2"] += r2.item()
            total_metrics[label_name]["normalized_mae"] += normalized_mae.item()
            total_metrics[label_name]["count"] += 1

    # 打印平均结果
    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    for label_name in label_names:
        count = total_metrics[label_name]["count"]
        if count > 0:
            avg_rmse = total_metrics[label_name]["rmse"] / count
            avg_r2 = total_metrics[label_name]["r2"] / count
            avg_normalized_mae = total_metrics[label_name]["normalized_mae"] / count

            print(f"\n{label_name}:")
            print(f"  RMSE: {avg_rmse:.4f} Nm/kg")
            print(f"  R²: {avg_r2:.4f}")
            print(f"  归一化MAE: {avg_normalized_mae:.4f}")
            print(f"  有效样本数: {count}")
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

    # 替换配置中的通配符
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    # 根据模型类型加载数据集
    if model_type in ["Transformer", "GenerativeTransformer"]:
        seq_len = config.gen_sequence_length if model_type == "GenerativeTransformer" else config.sequence_length

        print(f"\n加载{model_type}测试数据集...")
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
            model_type=model_type,
            start_token_value=config.start_token_value if model_type == "GenerativeTransformer" else 0.0,
            remove_nan=True
        )

        collate_fn = collate_fn_generative if model_type == "GenerativeTransformer" else collate_fn_predictor

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        print(f"测试集大小: {len(test_dataset)} 个序列")
        print(f"测试批次数: {len(test_loader)}")

        # 进行测试
        all_estimates = []
        all_labels = []

        if model_type == "GenerativeTransformer":
            if args.use_generation:
                print("使用自回归生成模式...")
                with torch.no_grad():
                    for batch_data in test_loader:
                        input_data, _, label_data = batch_data
                        input_data = input_data.to(device)
                        label_data = label_data.to(device)

                        # 自回归生成
                        estimates = model.generate(input_data, max_len=seq_len)

                        all_estimates.append(estimates)
                        all_labels.append(label_data)
            else:
                print("使用Teacher Forcing模式...")
                with torch.no_grad():
                    for batch_data in test_loader:
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
        else:  # Transformer预测模型
            with torch.no_grad():
                for input_data, label_data in test_loader:
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)

                    estimates = model(input_data)

                    all_estimates.append(estimates)
                    all_labels.append(label_data)

        # 合并所有批次的结果
        all_estimates = torch.cat(all_estimates, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # 打印结果
        print_results(label_names, all_estimates, all_labels)

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

        # 进行测试 - 逐个处理避免concatenate问题
        # 初始化累积指标
        num_outputs = len(label_names)
        total_metrics = {
            label_name: {"rmse": 0.0, "r2": 0.0, "normalized_mae": 0.0, "count": 0}
            for label_name in label_names
        }

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

                            # 计算归一化MAE
                            label_range = torch.max(lbl_valid) - torch.min(lbl_valid)
                            mae = torch.mean(torch.abs(est_valid - lbl_valid))
                            normalized_mae = mae / (label_range + 1e-8)

                            # 累积指标
                            total_metrics[label_name]["rmse"] += rmse.item()
                            total_metrics[label_name]["r2"] += r2.item()
                            total_metrics[label_name]["normalized_mae"] += normalized_mae.item()
                            total_metrics[label_name]["count"] += 1

        # 打印结果
        print("\n" + "=" * 60)
        print("测试结果:")
        print("=" * 60)
        for label_name in label_names:
            count = total_metrics[label_name]["count"]
            if count > 0:
                avg_rmse = total_metrics[label_name]["rmse"] / count
                avg_r2 = total_metrics[label_name]["r2"] / count
                avg_normalized_mae = total_metrics[label_name]["normalized_mae"] / count

                print(f"\n{label_name}:")
                print(f"  RMSE: {avg_rmse:.4f} Nm/kg")
                print(f"  R²: {avg_r2:.4f}")
                print(f"  归一化MAE: {avg_normalized_mae:.4f}")
                print(f"  有效样本数: {count}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
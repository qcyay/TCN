import argparse
import os
import re
from typing import List, Tuple, Dict
from collections import defaultdict
import torch
import torch.nn as nn
# 设置Matplotlib使用Agg后端（无图形界面）
import matplotlib
matplotlib.use('Agg')  # 关键：使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader
from utils.config_utils import *
from utils.utils import *
from models.predictor_model import PredictorTransformer
from models.generative_model import GenerativeTransformer
from models.tcn import TCN
from dataset_loaders.sequence_dataloader import SequenceDataset
from dataset_loaders.dataloader import TcnDataset

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs.default_config",
                    help="配置文件路径")
parser.add_argument("--model_path", type=str, default=None,
                    help="训练好的模型路径")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="设备 (cpu, cuda, 0, 1, 2, 2, ...)")
parser.add_argument("--batch_size", type=str, default=32,
                    help="批次大小")
parser.add_argument("--use_generation", action="store_true",
                    help="生成式模型是否使用自回归生成模式（默认使用teacher forcing）")
args = parser.parse_args()

# 加载配置
config = load_config(args.config_path)
config = apply_feature_selection(config)

def load_model(model_path: str, device: torch.device):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_type = checkpoint.get("model_type", "TCN")
    print(f"模型类型: {model_type}")

    model = create_model_from_config(model_type, checkpoint, None, device)

    model.load_state_dict(checkpoint["state_dict"])
    print(f"模型加载成功! Epoch: {checkpoint.get('epoch', 'N/A')}")

    return model, model_type

def main():
    # 设置并验证设备
    device = setup_device(args.device)

    model_path = args.model_path if args.model_path else config.model_path

    # 加载模型
    model, model_type = load_model(model_path, device)
    model.eval()

    print(f"模型类型: {model_type}")

    # 获取reconstruction_method（仅对Transformer模型有效）
    reconstruction_method = getattr(config, 'reconstruction_method', 'only_first')
    if model_type in ['Transformer', 'GenerativeTransformer']:
        print(f"序列重组方法: {reconstruction_method}")

    # 替换配置中的通配符
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    # 获取类别映射和未见过任务列表
    action_to_category = ACTION_TO_CATEGORY
    unseen_patterns = UNSEEN_ACTION_PATTERNS

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
            output_sequence_length=config.output_sequence_length,
            model_delays=config.model_delays,
            participant_masses=config.participant_masses,
            device=device,
            mode='test',
            model_type=model_type,
            start_token_value=config.start_token_value if model_type == "GenerativeTransformer" else 0.0,
            remove_nan=True,
            action_patterns=getattr(config, 'action_patterns', None),
            enable_action_filter=getattr(config, 'enable_action_filter', False),
            activity_flag=config.activity_flag,
            min_sequence_length=getattr(config, 'min_sequence_length', -1)
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
        all_masks = [] if config.activity_flag else None

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
            else:  # Transformer预测模型
                for batch_idx, (input_data, label_data, mask_data) in enumerate(test_loader):
                    # 尺寸为[B,num_input_features,sequence_length]
                    input_data = input_data.to(device)
                    # 尺寸为[B,num_outputs,sequence_length]
                    label_data = label_data.to(device)
                    if mask_data is not None:
                        # 尺寸为[B,num_outputs,sequence_length]
                        mask_data = mask_data.to(device)

                    # 尺寸为[B,num_outputs,sequence_length]
                    estimates = model(input_data)

                    all_estimates.append(estimates)
                    all_labels.append(label_data)
                    if mask_data is not None:
                        all_masks.append(mask_data)

        # 合并所有批次的结果
        all_estimates = torch.cat(all_estimates, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        if mask_data is not None:
            all_masks = torch.cat(all_masks, dim=0)

        print(f"预测完成! 共 {all_estimates.size(0)} 个短序列")

        # 使用优化的重组方法
        print(f"\n使用 '{reconstruction_method}' 方法重组序列...")
        all_estimates, all_labels, all_masks = reconstruct_sequences(
            all_estimates, all_labels, all_masks,
            test_dataset.trial_sequence_counts,
            method=reconstruction_method
        )

        print(f"重组完成! 共 {len(all_estimates)} 个完整序列")

        # 获取trial名称列表
        trial_names = test_dataset.trial_names

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
            remove_nan=True,
            action_patterns=getattr(config, 'action_patterns', None),
            enable_action_filter=getattr(config, 'enable_action_filter', False),
            activity_flag=config.activity_flag,
            min_sequence_length=getattr(config, 'min_sequence_length', -1)
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # TCN使用batch_size=1避免长度不一致
            shuffle=False,
            collate_fn=collate_fn_tcn
        )

        print(f"测试集大小: {len(test_dataset)} 个试验")

        # 进行测试 - 逐个处理并收集完整序列
        all_estimates = []
        all_labels = []
        all_masks = [] if config.activity_flag else None

        print("\n开始预测...")
        with torch.no_grad():
            for input_data, label_data, trial_lengths, activity_masks in test_loader:
                input_data = input_data.to(device)
                label_data = label_data.to(device)

                mask_flag = True if activity_masks[0] is not None else False

                if mask_flag:
                    activity_masks = [m.to(device) for m in activity_masks]

                estimates = model(input_data)

                # TCN: 提取有效区域
                model_history = model.get_effective_history()
                batch_size = estimates.size(0)

                for i in range(batch_size):
                    # 尺寸为[N_label,n]
                    est_trial = estimates[i, :, model_history:trial_lengths[i]]
                    # 尺寸为[N_label,n]
                    lbl_trial = label_data[i, :, model_history:trial_lengths[i]]

                    # 获取activity_mask
                    if mask_flag:
                        # 尺寸为[n]
                        act_mask_trial = activity_masks[i][model_history:]

                    # 找到最大delay
                    max_delay = max(config.model_delays)
                    valid_length = est_trial.size(1) - max_delay

                    if valid_length <= 0:
                        continue

                    # 对每个输出通道应用delay
                    aligned_est = []
                    aligned_lbl = []

                    for j in range(len(label_names)):
                        delay = config.model_delays[j]

                        # 预测值：从max_delay位置开始取
                        # 尺寸为[n-max_delay]
                        est = est_trial[j, max_delay:]

                        # 标签：根据delay偏移
                        lbl_start = max_delay - delay
                        lbl_end = lbl_start + valid_length
                        # 尺寸为[n-max_delay]
                        lbl = lbl_trial[j, lbl_start:lbl_end]

                        aligned_est.append(est)
                        aligned_lbl.append(lbl)

                    # 列表,包含N_test个tensor,尺寸为[num_outputs, n-max_delay]
                    all_estimates.append(torch.stack(aligned_est))
                    # 列表,包含N_test个tensor,尺寸为[num_outputs, n-max_delay]
                    all_labels.append(torch.stack(aligned_lbl))

                    # 获取对应的activity_mask
                    # 尺寸为[n-max_delay]
                    if mask_flag:
                        act_mask_j = act_mask_trial[max_delay:]
                        all_masks.append(act_mask_j)

        # 获取trial名称列表
        trial_names = test_dataset.trial_names

    # 计算按类别分组的指标
    print("\n计算各类别指标...")
    category_metrics = compute_category_metrics(
        all_estimates,
        all_labels,
        all_masks,
        trial_names,
        label_names,
        action_to_category,
        unseen_patterns,
    )

    # 打印结果
    print_category_results(category_metrics, label_names)

    # 确定保存目录
    model_dir = os.path.dirname(model_path)
    save_dir = model_dir if model_dir else "."

    # 保存结果到文件
    save_metrics_to_file(category_metrics, label_names, save_dir)

    # 生成箱线图
    if getattr(config, 'generate_boxplots', True):
        print("\n生成箱线图...")

        # 三大类别的箱线图
        if getattr(config, 'plot_categories', True):
            create_boxplots(
                category_metrics,
                label_names,
                save_dir,
                categories_to_plot=['Cyclic', 'Impedance-like', 'Unstructured']
            )

        # 未见任务的箱线图（如果存在）
        if getattr(config, 'plot_unseen', True) and 'Unseen' in category_metrics:
            # 为未见任务创建单独的图（与某个基准类别对比）
            # 这里我们创建一个包含All和Unseen的对比图
            for label_name in label_names:
                fig, axes = plt.subplots(1, 3, figsize=(12, 5))
                fig.suptitle(f'{label_name} - All vs Unseen Tasks', fontsize=16, fontweight='bold')

                metric_info = {
                    'rmse': {'name': 'RMSE', 'ylabel': 'RMSE (Nm/kg)'},
                    'r2': {'name': 'R²', 'ylabel': 'R²'},
                    'mae_percent': {'name': 'MAE', 'ylabel': 'Normalized MAE (%)'}
                }

                for metric_idx, (metric_key, metric_data) in enumerate(metric_info.items()):
                    ax = axes[metric_idx]

                    data_to_plot = []
                    positions = []
                    labels_plot = []

                    for cat_idx, category in enumerate(['All', 'Unseen']):
                        if category in category_metrics and label_name in category_metrics[category]:
                            values = category_metrics[category][label_name].get(metric_key, [])
                            if values:
                                data_to_plot.append(values)
                                positions.append(cat_idx + 1)
                                labels_plot.append(category)

                    if data_to_plot:
                        bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                                        patch_artist=True, showfliers=True,
                                        boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5),
                                        medianprops=dict(color='black', linewidth=2),
                                        whiskerprops=dict(color='black', linewidth=1.5),
                                        capprops=dict(color='black', linewidth=1.5))

                        # 添加均值
                        for i, values in enumerate(data_to_plot):
                            mean_val = sum(values) / len(values)
                            ax.plot(positions[i], mean_val, marker='s', markersize=6,
                                    color='black', zorder=3)

                        ax.set_title(metric_data['name'], fontsize=14, fontweight='bold')
                        ax.set_ylabel(metric_data['ylabel'], fontsize=12)
                        ax.set_xticks(positions)
                        ax.set_xticklabels(labels_plot, fontsize=11)
                        ax.grid(axis='y', alpha=0.3, linestyle='--')

                        if metric_key == 'r2':
                            ax.set_facecolor('#f0f0f0')

                plt.tight_layout()
                save_path = os.path.join(save_dir, f'boxplot_{label_name}_All_vs_Unseen.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"箱线图已保存: {save_path}")
                plt.close()

        # 额外的自定义组别箱线图
        additional_groups = getattr(config, 'additional_plot_groups', [])
        for group in additional_groups:
            group_name = group.get('name', 'Custom')
            group_patterns = group.get('patterns', [])

            if not group_patterns:
                continue

            # 为这个组创建临时的类别映射
            temp_category_map = {pattern: group_name for pattern in group_patterns}

            # 计算这个组的指标
            temp_metrics = compute_category_metrics(
                all_estimates,
                all_labels,
                all_masks,
                trial_names,
                label_names,
                temp_category_map,
                []  # 不考虑unseen
            )

            # 如果有数据，创建箱线图
            if group_name in temp_metrics:
                create_boxplots(
                    temp_metrics,
                    label_names,
                    save_dir,
                    categories_to_plot=[group_name]
                )

    print("\n测试完成!")


if __name__ == "__main__":
    main()
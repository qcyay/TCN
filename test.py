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
    action_to_category = getattr(config, 'action_to_category', {})
    unseen_patterns = getattr(config, 'unseen_action_patterns', [])

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
                for batch_idx, (input_data, label_data) in enumerate(test_loader):
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)

                    estimates = model(input_data)

                    all_estimates.append(estimates)
                    all_labels.append(label_data)

        # 合并所有批次的结果
        all_estimates = torch.cat(all_estimates, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        print(f"预测完成! 共 {all_estimates.size(0)} 个短序列")

        # 使用优化的重组方法
        print(f"\n使用 '{reconstruction_method}' 方法重组序列...")
        reconstructed_estimates, reconstructed_labels = reconstruct_sequences(
            all_estimates, all_labels,
            test_dataset.trial_sequence_counts,
            method=reconstruction_method
        )

        print(f"重组完成! 共 {len(reconstructed_estimates)} 个完整序列")

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
            load_to_device=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # TCN使用batch_size=1避免长度不一致
            shuffle=False,
            collate_fn=collate_fn_tcn
        )

        print(f"测试集大小: {len(test_dataset)} 个试验")

        # 进行测试 - 逐个处理并收集完整序列
        reconstructed_estimates = []
        reconstructed_labels = []

        print("\n开始预测...")
        with torch.no_grad():
            for input_data, label_data, trial_lengths in test_loader:
                input_data = input_data.to(device)
                label_data = label_data.to(device)

                estimates = model(input_data)

                # TCN: 提取有效区域
                model_history = model.get_effective_history()
                batch_size = estimates.size(0)

                for i in range(batch_size):
                    est_trial = estimates[i, :, model_history:trial_lengths[i]]
                    lbl_trial = label_data[i, :, model_history:trial_lengths[i]]

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
                        est = est_trial[j, max_delay:]

                        # 标签：根据delay偏移
                        lbl_start = max_delay - delay
                        lbl_end = lbl_start + valid_length
                        lbl = lbl_trial[j, lbl_start:lbl_end]

                        aligned_est.append(est)
                        aligned_lbl.append(lbl)

                    # 堆叠为 [num_outputs, seq_len]
                    reconstructed_estimates.append(torch.stack(aligned_est))
                    reconstructed_labels.append(torch.stack(aligned_lbl))

        # 获取trial名称列表
        trial_names = test_dataset.trial_names

    # 计算按类别分组的指标
    print("\n计算各类别指标...")
    category_metrics = compute_category_metrics(
        reconstructed_estimates,
        reconstructed_labels,
        trial_names,
        label_names,
        action_to_category,
        unseen_patterns
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
                reconstructed_estimates,
                reconstructed_labels,
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
"""
可视化和分析工具
用于读取test.py生成的pt文件并进行可视化和分析
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 无需显示器的后端
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def load_test_results(pt_path: str):
    """
    加载测试结果文件

    参数:
        pt_path: .pt文件路径

    返回:
        all_estimates, all_labels, trial_names, category_metrics
    """
    data = torch.load(pt_path, map_location='cpu')
    return (
        data['all_estimates'],
        data['all_labels'],
        data['trial_names'],
        data['category_metrics']
    )


def visualize_trial(
        all_estimates: List[torch.Tensor],
        all_labels: List[torch.Tensor],
        trial_names: List[str],
        trial_idx: int,
        save_path: Optional[str] = None
):
    """
    可视化指定序号的试验预测结果

    参数:
        all_estimates: 预测值列表
        all_labels: 真值列表
        trial_names: 试验名称列表
        trial_idx: 要可视化的试验序号
        save_path: 保存路径（可选）
    """
    est = all_estimates[trial_idx].cpu().numpy()  # [num_outputs, N]
    lbl = all_labels[trial_idx].cpu().numpy()  # [num_outputs, N]
    trial_name = trial_names[trial_idx]

    num_outputs, seq_len = est.shape
    time = np.arange(seq_len) * 0.005  # 假设200Hz采样率

    fig, axes = plt.subplots(num_outputs, 1, figsize=(12, 4 * num_outputs))
    if num_outputs == 1:
        axes = [axes]

    joint_names = ['Hip', 'Knee', 'Ankle']  # 根据实际情况修改

    for i in range(num_outputs):
        ax = axes[i]
        ax.plot(time, lbl[i, :], 'b-', label='Ground Truth', linewidth=1.5)
        ax.plot(time, est[i, :], 'r--', label='Prediction', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Moment (Nm/kg)', fontsize=12)
        ax.set_title(f'{joint_names[i] if i < len(joint_names) else f"Joint {i}"} - {trial_name}',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存: {save_path}")
    else:
        plt.show()

    plt.close()

def save_metrics_summary(
        category_metrics: Dict,
        label_names: List[str],
        save_dir: str,
        sort_by: Optional[str] = None
):
    """
    保存指标摘要到txt文件

    参数:
        category_metrics: 类别指标字典
        label_names: 标签名称列表
        save_dir: 保存目录
        sort_by: 排序依据 ('rmse', 'r2', 'mae_percent', None)
    """
    os.makedirs(save_dir, exist_ok=True)

    # 遍历每个label_name
    for label_name in label_names:
        if label_name not in category_metrics['All']:
            continue

        # 提取数据
        metrics = category_metrics['All'][label_name]
        rmse_list = metrics['rmse']
        r2_list = metrics['r2']
        mae_list = metrics['mae_percent']
        trial_infos = metrics['trial_info']

        # 构建数据行列表
        data_rows = []
        for idx, (rmse, r2, mae, info) in enumerate(zip(rmse_list, r2_list, mae_list, trial_infos)):
            data_rows.append({
                'index': idx,
                'trial_name': info['trial_name'],
                'category': info['category'],
                'sequence_length': info['sequence_length'],
                'rmse': rmse,
                'r2': r2,
                'mae_percent': mae
            })

        # 排序（如果指定）
        if sort_by in ['rmse', 'r2', 'mae_percent']:
            reverse = (sort_by == 'r2')  # R²越大越好
            data_rows.sort(key=lambda x: x[sort_by], reverse=reverse)

        # 保存到文件
        save_path = os.path.join(save_dir, f'{label_name}_metrics_summary.txt')
        with open(save_path, 'w', encoding='utf-8') as f:
            # 写入表头
            f.write(f"{'=' * 120}\n")
            f.write(f"{label_name} - Metrics Summary\n")
            if sort_by:
                f.write(f"Sorted by: {sort_by} ({'descending' if sort_by == 'r2' else 'ascending'})\n")
            f.write(f"{'=' * 120}\n\n")

            # 写入列标题
            f.write(f"{'Idx':<6} {'Original_Idx':<13} {'Trial_Name':<40} {'Category':<15} "
                    f"{'Seq_Len':<8} {'RMSE':<10} {'R²':<10} {'MAE%':<10}\n")
            f.write(f"{'-' * 120}\n")

            # 写入数据行
            for i, row in enumerate(data_rows):
                f.write(f"{i:<6} {row['index']:<13} {row['trial_name']:<40} {row['category']:<15} "
                        f"{row['sequence_length']:<8} {row['rmse']:<10.4f} {row['r2']:<10.4f} "
                        f"{row['mae_percent']:<10.2f}\n")

            f.write(f"{'=' * 120}\n")

            # 写入统计信息
            f.write(f"\nStatistics:\n")
            f.write(f"  Total trials: {len(data_rows)}\n")
            f.write(f"  RMSE: {np.mean([r['rmse'] for r in data_rows]):.4f} ± "
                    f"{np.std([r['rmse'] for r in data_rows]):.4f}\n")
            f.write(f"  R²:   {np.mean([r['r2'] for r in data_rows]):.4f} ± "
                    f"{np.std([r['r2'] for r in data_rows]):.4f}\n")
            f.write(f"  MAE:  {np.mean([r['mae_percent'] for r in data_rows]):.2f} ± "
                    f"{np.std([r['mae_percent'] for r in data_rows]):.2f}%\n")

        print(f"摘要已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='可视化和分析测试结果')
    parser.add_argument('--pt_path', type=str, required=True,
                        help='测试结果.pt文件路径')
    parser.add_argument('--trial_idx', type=int, default=None,
                        help='要可视化的试验序号')
    parser.add_argument('--save_summary', action='store_true',
                        help='是否保存指标摘要')
    parser.add_argument('--sort_by', type=str, default=None,
                        choices=['rmse', 'r2', 'mae_percent'],
                        help='排序依据')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认与pt文件同目录）')

    args = parser.parse_args()

    # 加载数据
    print(f"加载测试结果: {args.pt_path}")
    all_estimates, all_labels, trial_names, category_metrics = load_test_results(args.pt_path)
    print(f"共加载 {len(trial_names)} 个试验")

    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.pt_path)
    # breakpoint()
    # 可视化指定试验
    if args.trial_idx is not None:
        if 0 <= args.trial_idx < len(trial_names):
            save_path = os.path.join(output_dir, f'visualization_trial_{args.trial_idx}.png')
            visualize_trial(all_estimates, all_labels, trial_names, args.trial_idx, save_path)
        else:
            print(f"错误: trial_idx {args.trial_idx} 超出范围 [0, {len(trial_names) - 1}]")

    # 保存指标摘要
    if args.save_summary:
        label_names = list(category_metrics['All'].keys())
        summary_dir = os.path.join(output_dir, 'metrics_summary')
        save_metrics_summary(category_metrics, label_names, summary_dir, args.sort_by)


if __name__ == '__main__':
    main()
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
from matplotlib import rcParams
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

rcParams['font.sans-serif'] = ['SimHei']   # 黑体（Windows 基本都有）
rcParams['axes.unicode_minus'] = False     # 解决负号显示问题

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

    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    vis_dir = 'visualization'

    os.makedirs(os.path.join(base_dir, vis_dir), exist_ok=True)

    # 全局风格（可放到函数外）
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.linewidth'] = 0.8

    est = all_estimates[trial_idx].cpu().numpy()  # [num_outputs, N]
    lbl = all_labels[trial_idx].cpu().numpy()  # [num_outputs, N]
    trial_name = trial_names[trial_idx]

    print(est.shape, lbl.shape)
    # # TCN
    # est = est[:, 600:]
    # lbl = lbl[:, 600:]
    # Transformer
    est = est[:, 648:]
    lbl = lbl[:, 648:]

    num_outputs, seq_len = est.shape
    time = np.arange(seq_len) * 0.005  # 假设200Hz采样率

    # fig, axes = plt.subplots(num_outputs, 1, figsize=(10, 3.0 * num_outputs), sharex=True)
    # if num_outputs == 1:
    #     axes = [axes]

    joint_names = ['Hip', 'Knee']  # 根据实际情况修改

    for i in range(num_outputs):
        # ax = axes[i]
        # 每个输出单独创建一张图
        fig, ax = plt.subplots(figsize=(6, 3.2))

        ax.plot(time, lbl[i, :], color='black', linewidth=1.8)
        ax.plot(time, est[i, :], color=(39/255,170/255,226/255), linewidth=1.6)
        # ax.plot(time, lbl[i, :], 'b-', label='真值', linewidth=1.5)
        # ax.plot(time, est[i, :], 'r--', label='预测值', linewidth=1.5, alpha=0.7)

        ax.set_xlabel('Time (s)', fontsize=20)
        ax.set_ylabel('Moment (Nm/kg)', fontsize=20)
        # ax.set_title(f'Upstairs', fontsize=14, fontweight='normal', pad=4)
        # ax.set_title(f'{joint_names[i] if i < len(joint_names) else f"Joint {i}"} - {trial_name}',
        #              fontsize=14, fontweight='bold')
        # ax.set_xlabel('时间 (秒)', fontsize=12)
        # ax.set_ylabel('力矩 (牛米/千克)', fontsize=12)
        # ax.set_title(f'上楼', fontsize=14, fontweight='bold')

        # ax.legend(loc='best')
        # ax.grid(True, alpha=0.3)
        # 取消网格线
        ax.grid(False)

        # 只保留左边和下边坐标轴线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        # 刻度样式
        ax.tick_params(axis='both', which='major', labelsize=16, width=0.8, length=4)

        plt.tight_layout()

        if save_path:
            png_save_path = os.path.join(base_dir, vis_dir, f'{base_name}_{joint_names[i]}.png')
            pdf_save_path = os.path.join(base_dir, vis_dir, f'{base_name}_{joint_names[i]}.pdf')
            plt.savefig(png_save_path, dpi=300, bbox_inches='tight')
            plt.savefig(pdf_save_path, bbox_inches='tight')
            print(f"图片已保存: {png_save_path}")
            print(f"图片已保存: {pdf_save_path}")
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


def plot_metrics_boxplots(
        pt_paths: List[str],
        save_dir: str,
        categories: List[str] = ['All', 'Cyclic', 'Impedance-like', 'Unstructured', 'Unseen']
):
    """
    绘制多个pt文件的指标箱线图

    参数:
        pt_paths: pt文件路径列表
        save_dir: 保存目录
        categories: 要绘制的类别列表
    """
    os.makedirs(save_dir, exist_ok=True)

    # 加载所有pt文件的数据
    all_category_metrics = []
    pt_names = []
    for pt_path in pt_paths:
        _, _, _, category_metrics = load_test_results(pt_path)
        all_category_metrics.append(category_metrics)
        # 根据路径中的关键词确定模型名称
        pt_path_lower = pt_path.lower()
        if 'tcn' in pt_path_lower:
            pt_name = 'TCN'
        elif 'transformer' in pt_path_lower:
            pt_name = 'MomentFormer'
        else:
            # 如果没有匹配关键词，使用文件名
            pt_name = os.path.splitext(os.path.basename(pt_path))[0]
        pt_names.append(pt_name)

    # 获取所有label_names（从第一个文件）
    if 'All' not in all_category_metrics[0]:
        print("错误: category_metrics中没有'All'类别")
        return

    label_names = list(all_category_metrics[0]['All'].keys())
    metric_names = ['rmse', 'r2', 'mae_percent']
    # metric_display_names = {'rmse': 'RMSE', 'r2': 'R²', 'mae_percent': 'MAE (%)'}
    metric_display_names = {'rmse': '均方根误差', 'r2': '决定系数', 'mae_percent': '归一化平均绝对误差 (%)'}

    # 定义颜色方案（最多支持10个pt文件）
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # 对每个label_name和metric_name组合绘制图表
    for label_name in label_names:
        for metric_name in metric_names:
            fig, ax = plt.subplots(figsize=(10, 6))

            # 收集所有类别的数据
            all_data = []
            positions = []
            box_colors = []
            labels = []

            pos_counter = 0
            for cat_idx, category in enumerate(categories):
                # 检查该类别是否存在于所有pt文件中
                category_exists = all(category in cm for cm in all_category_metrics)
                if not category_exists:
                    continue

                # 为每个pt文件收集该类别的数据
                for pt_idx, category_metrics in enumerate(all_category_metrics):
                    if category not in category_metrics:
                        continue
                    if label_name not in category_metrics[category]:
                        continue

                    metrics = category_metrics[category][label_name]
                    if metric_name not in metrics:
                        continue

                    data = metrics[metric_name]
                    if len(data) == 0:
                        continue

                    all_data.append(data)
                    positions.append(pos_counter)
                    box_colors.append(colors[pt_idx])

                    # 只为第一个类别添加pt文件标签
                    if cat_idx == 0:
                        labels.append(pt_names[pt_idx] if len(pt_paths) > 1 else category)

                    pos_counter += 1

                # 在不同类别之间添加间隔
                pos_counter += 0.5

            if not all_data:
                print(f"警告: {label_name} - {metric_name} 没有数据")
                plt.close()
                continue

            # 绘制箱线图
            bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                           showmeans=True, meanprops=dict(marker='s', markerfacecolor='black',
                                                          markeredgecolor='black', markersize=6),
                           medianprops=dict(color='black', linewidth=1.5),
                           boxprops=dict(linewidth=1.5),
                           whiskerprops=dict(linewidth=1.5),
                           capprops=dict(linewidth=1.5))

            # 设置箱体颜色
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # 设置x轴标签
            if len(pt_paths) > 1:
                # 多个pt文件：显示类别标签
                category_positions = []
                category_labels = []
                current_pos = 0
                for category in categories:
                    # 计算该类别对应的所有box的中心位置
                    n_boxes = sum(1 for cm in all_category_metrics
                                 if category in cm and label_name in cm.get(category, {}))
                    if n_boxes > 0:
                        category_positions.append(current_pos + (n_boxes - 1) / 2)
                        category_labels.append(category)
                        current_pos += n_boxes + 0.5

                category_labels = ['所有活动','周期性活动','阻抗式活动','非结构化活动','未见活动']

                ax.set_xticks(category_positions)
                ax.set_xticklabels(category_labels, fontsize=11)

                # 添加图例
                legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.7)
                                for i in range(len(pt_names))]
                ax.legend(legend_handles, pt_names, loc='upper right', fontsize=9)
            else:
                # 单个pt文件：显示类别标签
                ax.set_xticks(positions)
                ax.set_xticklabels([categories[i // len(pt_paths)] for i in range(len(positions))],
                                  fontsize=11)

            # 设置标题和标签
            ax.set_ylabel(metric_display_names[metric_name], fontsize=12, fontweight='bold')
            # ax.set_title(f'{label_name} - {metric_display_names[metric_name]}',
            #             fontsize=14, fontweight='bold')
            ax.set_title(f'关节力矩 - {metric_display_names[metric_name]}',
                         fontsize=14, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

            # 设置y轴范围（R²特殊处理）
            if metric_name == 'r2':
                ax.set_ylim([0, 1.05])

            plt.tight_layout()

            # 保存图片
            save_path = os.path.join(save_dir, f'{label_name}_{metric_name}_boxplot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"箱线图已保存: {save_path}")

            plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化和分析测试结果')
    parser.add_argument('--pt_path', type=str, nargs='+', required=True,
                        help='测试结果.pt文件路径（可以指定多个）')
    parser.add_argument('--trial_idx', type=int, default=None,
                        help='要可视化的试验序号')
    parser.add_argument('--model_name', type=str, default=None, choices=['tcn', 'transformer'],
                        help='模型类型，可选: tcn 或 transformer')
    parser.add_argument('--save_summary', action='store_true',
                        help='是否保存指标摘要')
    parser.add_argument('--save_metrics_plots', action='store_true',
                        help='是否保存评估指标箱线图')
    parser.add_argument('--sort_by', type=str, default=None,
                        choices=['rmse', 'r2', 'mae_percent'],
                        help='排序依据')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录（默认与第一个pt文件同目录）')

    args = parser.parse_args()

    # 确定输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.pt_path[0])

    # 如果需要可视化试验或保存摘要，加载第一个pt文件
    if args.trial_idx is not None or args.save_summary:
        print(f"加载测试结果: {args.pt_path[0]}")
        all_estimates, all_labels, trial_names, category_metrics = load_test_results(args.pt_path[0])
        print(f"共加载 {len(trial_names)} 个试验")

        # 可视化指定试验
        if args.trial_idx is not None:
            if 0 <= args.trial_idx < len(trial_names):
                if args.model_name:
                    save_path = os.path.join(output_dir, f'visualization_trial_{args.trial_idx}_{args.model_name}.png')
                else:
                    save_path = os.path.join(output_dir, f'visualization_trial_{args.trial_idx}.png')
                visualize_trial(all_estimates, all_labels, trial_names, args.trial_idx, save_path)
            else:
                print(f"错误: trial_idx {args.trial_idx} 超出范围 [0, {len(trial_names) - 1}]")

        # 保存指标摘要
        if args.save_summary:
            label_names = list(category_metrics['All'].keys())
            summary_dir = os.path.join(output_dir, 'metrics_summary')
            save_metrics_summary(category_metrics, label_names, summary_dir, args.sort_by)

    # 保存评估指标箱线图
    if args.save_metrics_plots:
        print(f"\n开始生成评估指标箱线图...")
        plots_dir = os.path.join(output_dir, 'metrics_boxplots')
        plot_metrics_boxplots(args.pt_path, plots_dir)
        print(f"所有箱线图已保存至: {plots_dir}")


if __name__ == '__main__':
    main()
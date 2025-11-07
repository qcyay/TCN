"""
数据集内容分析工具
用于统计和分析TCN数据集中train和test文件夹的内容

运行方式：
    python utils/analyze_dataset.py
    python utils/analyze_dataset.py --data_dir data/example
    python utils/analyze_dataset.py --output analysis_result.txt
"""

import os
import re
import argparse
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from datetime import datetime


class OutputWriter:
    """同时输出到屏幕和文件的写入器"""

    def __init__(self, output_file: Optional[str] = None):
        self.output_file = output_file
        self.file_handle = None

        if output_file:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # 打开文件用于写入
            self.file_handle = open(output_file, 'w', encoding='utf-8')
            print(f"分析结果将保存到: {output_file}\n")

    def write(self, text: str):
        """同时写入屏幕和文件"""
        print(text, end='')
        if self.file_handle:
            self.file_handle.write(text)

    def writeln(self, text: str = ""):
        """写入一行（包含换行符）"""
        self.write(text + "\n")

    def close(self):
        """关闭文件"""
        if self.file_handle:
            self.file_handle.close()
            print(f"\n分析结果已保存到: {self.output_file}")


def parse_activity_folder(folder_name: str) -> Tuple[str, str]:
    """
    解析运动模式文件夹名称，提取运动形式和类型

    例如:
        ball_toss_1_2_center_off -> (ball_toss, center_off)
        normal_walk_3_2-0_off -> (normal_walk, off)
        stair_ascent_1_on -> (stair_ascent, on)

    参数:
        folder_name: 运动模式文件夹名称

    返回:
        (运动形式, 类型) 元组
    """
    # 分割文件夹名称
    parts = folder_name.split('_')

    # 提取运动形式（通常是开头的非数字部分）
    activity_parts = []
    for part in parts:
        # 如果部分完全是数字或包含数字和连字符，停止
        if re.match(r'^[\d\-]+$', part):
            break
        # 如果是单个数字，也停止
        if part.isdigit():
            break
        activity_parts.append(part)

    activity_type = '_'.join(activity_parts) if activity_parts else parts[0]

    # 提取类型（通常是最后的状态描述，如 on/off 或 center_off）
    # 从后往前找，直到找到包含 'on' 或 'off' 的部分
    condition_parts = []
    for part in reversed(parts):
        condition_parts.insert(0, part)
        # 如果找到了 on 或 off，停止
        if 'on' in part.lower() or 'off' in part.lower():
            break
        # 如果这部分是数字，继续
        if re.match(r'^[\d\-]+$', part):
            condition_parts.pop(0)
            continue

    condition = '_'.join(condition_parts) if condition_parts else 'unknown'

    # 如果提取失败，使用简单方法
    if not activity_type or not condition or condition == 'unknown':
        # 备用方案：假设最后一个非数字部分是类型
        if parts[-1] in ['on', 'off']:
            condition = parts[-1]
        elif len(parts) >= 2 and parts[-1] in ['on', 'off']:
            condition = '_'.join(parts[-2:])
        else:
            condition = parts[-1]

        # 运动形式是第一个或前两个单词
        if len(parts) >= 2:
            activity_type = '_'.join(parts[:2])
        else:
            activity_type = parts[0]

    return activity_type, condition


def scan_dataset_folder(data_dir: str, mode: str, writer: OutputWriter) -> Tuple[Dict[str, Dict[str, List[str]]], Set[str]]:
    """
    扫描train或test文件夹，收集运动模式信息

    参数:
        data_dir: 数据根目录
        mode: 'train' 或 'test'
        writer: 输出写入器

    返回:
        (嵌套字典: {运动形式: {类型: [参与者列表]}}, 所有运动模式文件夹集合)
    """
    mode_dir = os.path.join(data_dir, mode)

    if not os.path.exists(mode_dir):
        writer.writeln(f"警告: 目录不存在 - {mode_dir}")
        return {}, set()

    # 数据结构: {activity_type: {condition: [participants]}}
    activity_data = defaultdict(lambda: defaultdict(list))

    # 记录所有运动模式文件夹
    all_activity_folders = set()

    # 遍历参与者目录
    for participant in os.listdir(mode_dir):
        participant_dir = os.path.join(mode_dir, participant)

        # 跳过非目录和隐藏文件
        if not os.path.isdir(participant_dir) or participant.startswith('.'):
            continue

        # 遍历运动模式目录
        for activity_folder in os.listdir(participant_dir):
            activity_path = os.path.join(participant_dir, activity_folder)

            # 跳过非目录和隐藏文件
            if not os.path.isdir(activity_path) or activity_folder.startswith('.'):
                continue

            # 记录完整的运动模式文件夹名称
            all_activity_folders.add(activity_folder)

            # 解析运动形式和类型
            activity_type, condition = parse_activity_folder(activity_folder)

            # 记录参与者
            if participant not in activity_data[activity_type][condition]:
                activity_data[activity_type][condition].append(participant)

    return dict(activity_data), all_activity_folders


def print_section_header(writer: OutputWriter, title: str, char: str = '='):
    """打印分节标题"""
    writer.writeln(f"\n{char * 80}")
    writer.writeln(f"{title:^80}")
    writer.writeln(f"{char * 80}\n")


def print_activity_summary(writer: OutputWriter, activity_data: Dict[str, Dict[str, List[str]]], mode: str):
    """打印运动模式统计摘要"""
    print_section_header(writer, f"{mode.upper()} 数据集统计", '=')

    # 统计总数
    total_activities = len(activity_data)
    total_conditions = sum(len(conditions) for conditions in activity_data.values())
    total_trials = sum(
        len(participants)
        for conditions in activity_data.values()
        for participants in conditions.values()
    )

    writer.writeln(f"运动形式总数: {total_activities}")
    writer.writeln(f"运动类型总数: {total_conditions}")
    writer.writeln(f"试验总数: {total_trials}")

    # 按运动形式排序
    sorted_activities = sorted(activity_data.items())

    for activity_type, conditions in sorted_activities:
        writer.writeln(f"\n{'─' * 80}")
        writer.writeln(f"运动形式: {activity_type}")
        writer.writeln(f"{'─' * 80}")

        # 按类型排序
        sorted_conditions = sorted(conditions.items())

        for condition, participants in sorted_conditions:
            writer.writeln(f"  ├─ 类型: {condition}")
            writer.writeln(f"  │  └─ 参与者数量: {len(participants)}")
            writer.writeln(f"  │     参与者: {', '.join(sorted(participants))}")


def compare_datasets(writer: OutputWriter,
                    train_data: Dict[str, Dict[str, List[str]]],
                    test_data: Dict[str, Dict[str, List[str]]],
                    train_folders: Set[str],
                    test_folders: Set[str]):
    """比较train和test数据集"""
    print_section_header(writer, "数据集对比分析", '=')

    # 1. 比较运动形式
    train_activities = set(train_data.keys())
    test_activities = set(test_data.keys())

    common_activities = train_activities & test_activities
    train_only = train_activities - test_activities
    test_only = test_activities - train_activities

    writer.writeln("【运动形式对比】")
    writer.writeln(f"  共同的运动形式: {len(common_activities)}")
    if common_activities:
        writer.writeln(f"    {', '.join(sorted(common_activities))}")

    writer.writeln(f"\n  仅在TRAIN中的运动形式: {len(train_only)}")
    if train_only:
        writer.writeln(f"    {', '.join(sorted(train_only))}")

    writer.writeln(f"\n  仅在TEST中的运动形式: {len(test_only)}")
    if test_only:
        writer.writeln(f"    {', '.join(sorted(test_only))}")

    # 2. 比较每个运动形式的类型
    writer.writeln(f"\n\n{'─' * 80}")
    writer.writeln("【各运动形式的类型对比】")
    writer.writeln(f"{'─' * 80}")

    for activity in sorted(common_activities):
        train_conditions = set(train_data[activity].keys())
        test_conditions = set(test_data[activity].keys())

        common_cond = train_conditions & test_conditions
        train_only_cond = train_conditions - test_conditions
        test_only_cond = test_conditions - train_conditions

        writer.writeln(f"\n运动形式: {activity}")
        writer.writeln(f"  共同类型: {', '.join(sorted(common_cond)) if common_cond else '无'}")
        if train_only_cond:
            writer.writeln(f"  仅TRAIN有: {', '.join(sorted(train_only_cond))}")
        if test_only_cond:
            writer.writeln(f"  仅TEST有: {', '.join(sorted(test_only_cond))}")

    # 2. 比较完整的运动模式文件夹名称
    writer.writeln(f"\n\n{'─' * 80}")
    writer.writeln("【运动模式文件夹名称对比】")
    writer.writeln(f"{'─' * 80}")

    common_folders = train_folders & test_folders
    train_only_folders = train_folders - test_folders
    test_only_folders = test_folders - train_folders

    writer.writeln(f"\n共同的运动模式文件夹: {len(common_folders)}")
    if common_folders and len(common_folders) <= 20:
        for folder in sorted(common_folders):
            writer.writeln(f"  • {folder}")
    elif common_folders:
        writer.writeln(f"  (共 {len(common_folders)} 个，仅显示前20个)")
        for folder in sorted(list(common_folders))[:20]:
            writer.writeln(f"  • {folder}")

    writer.writeln(f"\n仅在TRAIN中的运动模式文件夹: {len(train_only_folders)}")
    if train_only_folders:
        for folder in sorted(train_only_folders):
            writer.writeln(f"  • {folder}")

    writer.writeln(f"\n⚠️  仅在TEST中的运动模式文件夹: {len(test_only_folders)}")
    if test_only_folders:
        writer.writeln("\n这些运动模式在训练集中没有对应数据：")
        for folder in sorted(test_only_folders):
            activity_type, condition = parse_activity_folder(folder)
            writer.writeln(f"  • {folder}")
            writer.writeln(f"    └─ 解析为: 运动形式='{activity_type}', 类型='{condition}'")


def print_statistics_summary(writer: OutputWriter,
                            train_data: Dict[str, Dict[str, List[str]]],
                            test_data: Dict[str, Dict[str, List[str]]]):
    """打印统计摘要"""
    print_section_header(writer, "整体统计摘要", '=')

    # Train统计
    train_activities = len(train_data)
    train_conditions = sum(len(conditions) for conditions in train_data.values())
    train_trials = sum(
        len(participants)
        for conditions in train_data.values()
        for participants in conditions.values()
    )

    # Test统计
    test_activities = len(test_data)
    test_conditions = sum(len(conditions) for conditions in test_data.values())
    test_trials = sum(
        len(participants)
        for conditions in test_data.values()
        for participants in conditions.values()
    )

    writer.writeln(f"{'指标':<30} {'TRAIN':>15} {'TEST':>15} {'总计':>15}")
    writer.writeln(f"{'-' * 75}")
    writer.writeln(f"{'运动形式数量':<30} {train_activities:>15} {test_activities:>15} {train_activities + test_activities:>15}")
    writer.writeln(f"{'运动类型数量':<30} {train_conditions:>15} {test_conditions:>15} {train_conditions + test_conditions:>15}")
    writer.writeln(f"{'试验数量':<30} {train_trials:>15} {test_trials:>15} {train_trials + test_trials:>15}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='分析TCN数据集的内容和结构',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
    python utils/analyze_dataset.py
    python utils/analyze_dataset.py --data_dir data/example
    python utils/analyze_dataset.py --output logs/dataset_analysis.txt
    python utils/analyze_dataset.py --data_dir data/full_dataset --verbose --output analysis.txt
        """
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='数据根目录路径 (默认: data)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径，如果不指定则只在屏幕显示 (例如: logs/dataset_analysis.txt)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='显示详细信息'
    )

    args = parser.parse_args()

    # 创建输出写入器
    writer = OutputWriter(args.output)

    try:
        # 写入标题和时间戳
        print_section_header(writer, "TCN 数据集分析工具", '█')
        writer.writeln(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        writer.writeln(f"数据目录: {args.data_dir}")

        # 检查数据目录是否存在
        if not os.path.exists(args.data_dir):
            writer.writeln(f"\n错误: 数据目录不存在 - {args.data_dir}")
            writer.writeln("请检查路径或使用 --data_dir 参数指定正确的数据目录")
            return

        # 扫描train和test文件夹
        writer.writeln("\n正在扫描数据集...")
        train_data, train_folders = scan_dataset_folder(args.data_dir, 'train', writer)
        test_data, test_folders = scan_dataset_folder(args.data_dir, 'test', writer)

        if not train_data and not test_data:
            writer.writeln("\n错误: 未找到任何数据")
            writer.writeln("请确保数据目录包含 'train' 和/或 'test' 子目录")
            return

        # 打印详细统计（如果启用verbose）
        if args.verbose:
            if train_data:
                print_activity_summary(writer, train_data, 'train')
            if test_data:
                print_activity_summary(writer, test_data, 'test')

        # 打印整体统计
        print_statistics_summary(writer, train_data, test_data)

        # 比较数据集
        if train_data and test_data:
            compare_datasets(writer, train_data, test_data, train_folders, test_folders)
        elif train_data:
            print_section_header(writer, "注意", '!')
            writer.writeln("只找到TRAIN数据集，无法进行对比分析")
        elif test_data:
            print_section_header(writer, "注意", '!')
            writer.writeln("只找到TEST数据集，无法进行对比分析")

        writer.writeln("\n" + "=" * 80)
        writer.writeln("分析完成!")
        writer.writeln("=" * 80 + "\n")

    finally:
        # 确保文件被正确关闭
        writer.close()


if __name__ == "__main__":
    main()
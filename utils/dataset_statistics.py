"""
数据集统计工具

用于统计指定文件夹下运动数据文件的数量,并根据config中的action_patterns进行分类统计。

功能:
1. 统计所有运动数据文件的总数量
2. 统计每个运动类型(基于action_patterns)的文件数量
2. 列出每个运动类型下的所有文件名

使用方法:
    python dataset_statistics.py --config configs/partial_motion_knee_config.py --data_dir data/train
    python dataset_statistics.py --config configs/partial_motion_knee_config.py --data_dir data/test
    python dataset_statistics.py --config configs/Transformer/partial_motion_knee_config.py --data_dir data/train
    python dataset_statistics.py --config ../configs/Transformer/partial_motion_knee_config.py --data_dir ./data/train
    python dataset_statistics.py --config configs/partial_motion_knee_config.py --data_dir data/train --save_output stats_train.txt
"""

import os
import re
import sys
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
import importlib.util


def load_config_from_file(config_path: str):
    """
    从文件路径直接加载配置文件

    使用importlib.util.spec_from_file_location来加载配置,
    可以处理相对路径、绝对路径和各种路径格式

    参数:
        config_path: 配置文件路径

    返回:
        配置模块
    """
    # 转换为绝对路径
    abs_config_path = os.path.abspath(config_path)

    if not os.path.exists(abs_config_path):
        raise FileNotFoundError(f"配置文件不存在: {abs_config_path}")

    print(f"加载配置文件: {abs_config_path}")

    # 生成模块名称
    module_name = "config_module"

    # 使用spec_from_file_location加载配置
    spec = importlib.util.spec_from_file_location(module_name, abs_config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载配置文件: {abs_config_path}")

    config = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = config
    spec.loader.exec_module(config)

    return config


def parse_action_patterns_from_config(config) -> List[Tuple[str, List[str]]]:
    """
    从配置文件中解析action_patterns

    返回:
        列表,每个元素是元组 (注释名称, [正则表达式列表])
    """
    # 检查是否启用了action_filter
    if not hasattr(config, 'enable_action_filter') or not config.enable_action_filter:
        print("警告: enable_action_filter未启用,将统计所有文件")
        return []

    # 获取action_patterns
    if not hasattr(config, 'action_patterns'):
        print("警告: 配置中没有找到action_patterns")
        return []

    action_patterns = config.action_patterns

    # 解析action_patterns(可能包含注释行)
    parsed_patterns = []

    for pattern_line in action_patterns:
        if isinstance(pattern_line, str) and pattern_line.strip():
            # 分割逗号分隔的多个正则表达式
            patterns = [p.strip() for p in pattern_line.split(',') if p.strip()]

            if patterns:
                # 尝试从pattern中提取描述(通常没有描述,我们用模式本身)
                parsed_patterns.append(("模式", patterns))

    return parsed_patterns


def parse_action_patterns_from_file(config_file_path: str) -> List[Tuple[str, List[str]]]:
    """
    直接从配置文件中解析action_patterns(包括注释信息)

    这个方法读取原始文件,可以获取注释行信息

    返回:
        列表,每个元素是元组 (注释名称, [正则表达式列表])
    """
    abs_path = os.path.abspath(config_file_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"配置文件不存在: {abs_path}")

    parsed_patterns = []
    in_action_patterns = False

    with open(abs_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()

            # 检测action_patterns开始
            if 'action_patterns' in line and '=' in line and '[' in line:
                in_action_patterns = True
                continue

            # 检测action_patterns结束
            if in_action_patterns and ']' in line:
                break

            # 处理action_patterns内的行
            if in_action_patterns:
                # 跳过空行
                if not line.strip():
                    continue

                # 跳过完全注释的行(以#开头,没有r"开头的正则)
                stripped = line.strip()
                if stripped.startswith('#') and 'r"' not in line:
                    continue

                # 提取注释(如果有)
                comment = ""
                if '#' in line:
                    # 找到行末注释
                    parts = line.split('#', 1)
                    if len(parts) > 1:
                        comment = parts[1].strip()

                # 提取正则表达式
                # 查找所有 r"..." 或 r'...' 格式的字符串
                pattern_matches = re.findall(r'r["\']([^"\']+)["\']', line)

                if pattern_matches:
                    # 如果这行没有被注释掉(不是以#开头)
                    if not stripped.startswith('#'):
                        parsed_patterns.append((comment if comment else "未命名模式", pattern_matches))

    return parsed_patterns


def scan_data_directory(data_dir: str, mode: str = None) -> List[str]:
    """
    扫描数据目录,获取所有试验文件名

    目录结构: data_dir/(train/test)/参与者/试验名称/

    参数:
        data_dir: 数据目录路径
        mode: 'train' 或 'test',如果为None则自动检测

    返回:
        试验名称列表(格式: "参与者/试验名称")
    """
    # 转换为绝对路径
    data_dir = os.path.abspath(data_dir)

    # 如果data_dir已经包含train或test子目录,直接使用
    if mode is None:
        if os.path.exists(os.path.join(data_dir, 'train')):
            mode = 'train'
        elif os.path.exists(os.path.join(data_dir, 'test')):
            mode = 'test'
        elif os.path.basename(data_dir) in ['train', 'test']:
            mode = os.path.basename(data_dir)
            data_dir = os.path.dirname(data_dir)
        else:
            # 直接使用data_dir作为模式目录
            mode = ''

    # 构建模式目录路径
    if mode:
        mode_dir = os.path.join(data_dir, mode)
    else:
        mode_dir = data_dir

    if not os.path.exists(mode_dir):
        raise FileNotFoundError(f"目录不存在: {mode_dir}")

    print(f"扫描目录: {mode_dir}")

    # 获取所有参与者目录
    participants = []
    for item in os.listdir(mode_dir):
        item_path = os.path.join(mode_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            participants.append(item)

    if not participants:
        raise ValueError(f"在目录 {mode_dir} 中未找到参与者数据")

    # 收集所有试验名称
    trial_names = []

    for participant in sorted(participants):
        participant_dir = os.path.join(mode_dir, participant)

        # 获取参与者的所有试验
        for trial_item in sorted(os.listdir(participant_dir)):
            trial_path = os.path.join(participant_dir, trial_item)
            if os.path.isdir(trial_path) and not trial_item.startswith('.'):
                # 试验名称格式: 参与者/试验项目
                trial_name = os.path.join(participant, trial_item)
                trial_names.append(trial_name)

    return trial_names


def match_trial_to_patterns(trial_name: str,
                            action_patterns: List[Tuple[str, List[str]]]) -> List[str]:
    """
    将试验名称与action_patterns匹配

    参数:
        trial_name: 试验名称(格式: "参与者/试验名称")
        action_patterns: 解析后的action_patterns列表

    返回:
        匹配到的模式名称列表(可能匹配多个)
    """
    # 提取试验基础名称(不包含参与者名称)
    trial_basename = os.path.basename(trial_name)

    matched_patterns = []

    for pattern_name, patterns in action_patterns:
        for pattern in patterns:
            if re.match(pattern, trial_basename):
                if pattern_name not in matched_patterns:
                    matched_patterns.append(pattern_name)
                break  # 该模式组已匹配,跳到下一个模式组

    return matched_patterns


def generate_statistics(trial_names: List[str],
                        action_patterns: List[Tuple[str, List[str]]]) -> Dict:
    """
    生成统计信息

    返回:
        统计字典,包含:
        - 'total': 总文件数
        - 'matched': 匹配的文件数
        - 'unmatched': 未匹配的文件数
        - 'patterns': {模式名称: [匹配的文件列表]}
    """
    stats = {
        'total': len(trial_names),
        'matched': 0,
        'unmatched': 0,
        'patterns': defaultdict(list),
        'unmatched_files': []
    }

    if not action_patterns:
        # 如果没有patterns,所有文件都算未匹配
        stats['unmatched'] = len(trial_names)
        stats['unmatched_files'] = trial_names
        return stats

    # 对每个试验名称进行匹配
    for trial_name in trial_names:
        matched = match_trial_to_patterns(trial_name, action_patterns)

        if matched:
            stats['matched'] += 1
            for pattern_name in matched:
                stats['patterns'][pattern_name].append(trial_name)
        else:
            stats['unmatched'] += 1
            stats['unmatched_files'].append(trial_name)

    return stats


def print_statistics(stats: Dict, action_patterns: List[Tuple[str, List[str]]],
                     show_filenames: bool = True):
    """
    打印统计信息

    参数:
        stats: 统计字典
        action_patterns: action_patterns列表
        show_filenames: 是否显示具体文件名
    """
    print("\n" + "=" * 80)
    print("数据集统计结果")
    print("=" * 80)

    # 总体统计
    print(f"\n总文件数: {stats['total']}")

    if action_patterns:
        print(f"匹配的文件数: {stats['matched']}")
        print(f"未匹配的文件数: {stats['unmatched']}")
        print(f"匹配率: {100 * stats['matched'] / stats['total']:.2f}%")

        # 各运动类型统计
        print(f"\n共有 {len(action_patterns)} 个运动类型模式")
        print("-" * 80)

        for i, (pattern_name, patterns) in enumerate(action_patterns, 1):
            matched_files = stats['patterns'].get(pattern_name, [])
            count = len(matched_files)

            print(f"\n{i}. {pattern_name}")
            print(f"   正则表达式: {patterns}")
            print(f"   匹配文件数: {count}")

            if show_filenames and matched_files:
                print(f"   匹配的文件:")
                for file in sorted(matched_files):
                    # 只显示试验名称部分(不含参与者)
                    trial_basename = os.path.basename(file)
                    print(f"      - {trial_basename}")

        # 未匹配的文件
        if stats['unmatched'] > 0:
            print(f"\n未匹配任何模式的文件 ({stats['unmatched']}个):")
            if show_filenames:
                for file in sorted(stats['unmatched_files']):
                    trial_basename = os.path.basename(file)
                    print(f"   - {trial_basename}")
    else:
        print("\n注意: 未启用action_filter或未找到action_patterns")
        print("显示所有文件:")
        if show_filenames:
            for file in sorted(stats['unmatched_files']):
                trial_basename = os.path.basename(file)
                print(f"   - {trial_basename}")

    print("\n" + "=" * 80)


def save_statistics_to_file(stats: Dict, action_patterns: List[Tuple[str, List[str]]],
                            output_file: str):
    """
    将统计信息保存到文件

    参数:
        stats: 统计字典
        action_patterns: action_patterns列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("数据集统计结果\n")
        f.write("=" * 80 + "\n\n")

        # 总体统计
        f.write(f"总文件数: {stats['total']}\n")

        if action_patterns:
            f.write(f"匹配的文件数: {stats['matched']}\n")
            f.write(f"未匹配的文件数: {stats['unmatched']}\n")
            f.write(f"匹配率: {100 * stats['matched'] / stats['total']:.2f}%\n")

            # 各运动类型统计
            f.write(f"\n共有 {len(action_patterns)} 个运动类型模式\n")
            f.write("-" * 80 + "\n")

            for i, (pattern_name, patterns) in enumerate(action_patterns, 1):
                matched_files = stats['patterns'].get(pattern_name, [])
                count = len(matched_files)

                f.write(f"\n{i}. {pattern_name}\n")
                f.write(f"   正则表达式: {patterns}\n")
                f.write(f"   匹配文件数: {count}\n")

                if matched_files:
                    f.write(f"   匹配的文件:\n")
                    for file in sorted(matched_files):
                        trial_basename = os.path.basename(file)
                        f.write(f"      - {trial_basename}\n")

            # 未匹配的文件
            if stats['unmatched'] > 0:
                f.write(f"\n未匹配任何模式的文件 ({stats['unmatched']}个):\n")
                for file in sorted(stats['unmatched_files']):
                    trial_basename = os.path.basename(file)
                    f.write(f"   - {trial_basename}\n")
        else:
            f.write("\n注意: 未启用action_filter或未找到action_patterns\n")
            f.write("所有文件:\n")
            for file in sorted(stats['unmatched_files']):
                trial_basename = os.path.basename(file)
                f.write(f"   - {trial_basename}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\n统计结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='统计数据集中的运动数据文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dataset_statistics.py --config configs/partial_motion_knee_config.py --data_dir data/train
  python dataset_statistics.py --config configs/partial_motion_knee_config.py --data_dir data/test
  python dataset_statistics.py --config ../configs/Transformer/partial_motion_knee_config.py --data_dir ./data/train
  python dataset_statistics.py --config configs/partial_motion_knee_config.py --data_dir data/train --save_output stats.txt
  python dataset_statistics.py --config configs/partial_motion_knee_config.py --data_dir data/train --no_filenames
        """
    )

    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径 (例如: configs/partial_motion_knee_config.py 或 ../configs/partial_motion_knee_config.py)')
    parser.add_argument('--data_dir', type=str, default='data/train',
                        help='数据目录路径 (默认: data/train)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], default=None,
                        help='数据模式 (train/test),如果不指定则自动检测')
    parser.add_argument('--save_output', type=str, default=None,
                        help='保存统计结果到文件 (可选)')
    parser.add_argument('--no_filenames', action='store_true',
                        help='不显示具体的文件名列表')

    args = parser.parse_args()

    try:
        # 加载配置
        config = load_config_from_file(args.config)

        # 解析action_patterns(从文件中读取,包含注释)
        config_file_path = args.config
        if not config_file_path.endswith('.py'):
            config_file_path += '.py'

        action_patterns = parse_action_patterns_from_file(config_file_path)

        if action_patterns:
            print(f"从配置文件中读取到 {len(action_patterns)} 个有效的运动类型模式")
        else:
            print("警告: 未找到有效的action_patterns,将统计所有文件")

        # 扫描数据目录
        trial_names = scan_data_directory(args.data_dir, args.mode)
        print(f"找到 {len(trial_names)} 个试验文件")

        # 生成统计
        stats = generate_statistics(trial_names, action_patterns)

        # 打印统计结果
        show_filenames = not args.no_filenames
        print_statistics(stats, action_patterns, show_filenames)

        # 保存到文件(如果指定)
        if args.save_output:
            save_statistics_to_file(stats, action_patterns, args.save_output)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
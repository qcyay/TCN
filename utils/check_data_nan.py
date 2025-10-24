"""
数据集NaN检测工具

功能：
1. 扫描训练集和测试集中的所有CSV文件
2. 检查config中input_names指定的每一列是否包含NaN
3. 统计每列的NaN数量和比例
1. 生成详细的检测报告

使用方法:
    # 从项目根目录执行
    python utils/check_data_nan.py --config_path configs.TCN.default_config

    # 或使用模块方式
    python -m utils.check_data_nan --config_path configs.TCN.default_config

    # 只检查训练集
    python utils/check_data_nan.py --mode train

    # 只检查测试集
    python utils/check_data_nan.py --mode test
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = parent_dir if os.path.basename(current_dir) == 'utils' else current_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 修改工作目录到项目根目录
if os.getcwd() != project_root:
    os.chdir(project_root)
    print(f"工作目录已切换到: {os.getcwd()}\n")

import argparse
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from utils.config_utils import load_config

# 解析命令行参数
parser = argparse.ArgumentParser(description="检测数据集中的NaN值")
parser.add_argument("--config_path", type=str, default="configs.TCN.default_config",
                    help="配置文件路径")
parser.add_argument("--mode", type=str, default="all", choices=["train", "test", "all"],
                    help="检测模式: train(仅训练集), test(仅测试集), all(全部)")
parser.add_argument("--detailed", action="store_true",
                    help="显示每个文件的详细信息")
parser.add_argument("--export", type=str, default=None,
                    help="导出报告到指定文件 (例如: nan_report.txt)")
args = parser.parse_args()

# 加载配置
config = load_config(args.config_path)


class NaNChecker:
    """NaN检测器类"""

    def __init__(self, data_dir: str, input_names: List[str], side: str):
        self.data_dir = data_dir
        self.input_names = input_names
        self.side = side

        # 替换通配符为实际侧别
        self.actual_input_names = [name.replace("*", side) for name in input_names]

        # 统计信息
        self.stats = {
            'train': defaultdict(lambda: {'total': 0, 'nan_count': 0, 'files_with_nan': 0, 'files_checked': 0}),
            'test': defaultdict(lambda: {'total': 0, 'nan_count': 0, 'files_with_nan': 0, 'files_checked': 0})
        }

        # 详细信息（如果需要）
        self.detailed_info = {
            'train': [],
            'test': []
        }

    def check_mode(self, mode: str) -> Tuple[int, int]:
        """
        检查指定模式的数据集
        返回: (总文件数, 包含NaN的文件数)
        """
        mode_dir = os.path.join(self.data_dir, mode)

        if not os.path.exists(mode_dir):
            print(f"⚠️  警告: 目录不存在 {mode_dir}")
            return 0, 0

        print(f"\n{'=' * 70}")
        print(f"检查 {mode.upper()} 数据集")
        print(f"{'=' * 70}")
        print(f"数据目录: {mode_dir}\n")

        # 扫描所有参与者和试验
        files_checked = 0
        files_with_nan = 0

        for participant in os.listdir(mode_dir):
            participant_dir = os.path.join(mode_dir, participant)

            if not os.path.isdir(participant_dir) or participant.startswith('.'):
                continue

            for trial in os.listdir(participant_dir):
                trial_dir = os.path.join(participant_dir, trial)

                if not os.path.isdir(trial_dir) or trial.startswith('.'):
                    continue

                # 构建输入文件路径
                input_file = f"{participant}_{trial}_exo.csv"
                input_file_path = os.path.join(trial_dir, input_file)

                if not os.path.exists(input_file_path):
                    print(f"⚠️  文件不存在: {input_file_path}")
                    continue

                # 检查文件
                has_nan = self._check_file(input_file_path, mode, participant, trial)
                files_checked += 1
                if has_nan:
                    files_with_nan += 1

                # 显示进度
                if files_checked % 10 == 0:
                    print(f"  已检查 {files_checked} 个文件...")

        print(f"\n✓ {mode.upper()} 数据集检查完成")
        print(f"  总文件数: {files_checked}")
        print(f"  包含NaN的文件数: {files_with_nan}")

        return files_checked, files_with_nan

    def _check_file(self, file_path: str, mode: str, participant: str, trial: str) -> bool:
        """
        检查单个CSV文件
        返回: 是否包含NaN
        """
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)

            has_nan = False
            file_detail = {
                'participant': participant,
                'trial': trial,
                'file': file_path,
                'columns': {}
            }

            # 检查每一列
            for col_name in self.actual_input_names:
                if col_name not in df.columns:
                    print(f"⚠️  警告: 列 '{col_name}' 不存在于文件 {file_path}")
                    continue

                # 统计NaN
                col_data = df[col_name]
                total_count = len(col_data)
                nan_count = col_data.isna().sum()

                # 更新统计信息
                self.stats[mode][col_name]['total'] += total_count
                self.stats[mode][col_name]['nan_count'] += nan_count
                self.stats[mode][col_name]['files_checked'] += 1

                if nan_count > 0:
                    self.stats[mode][col_name]['files_with_nan'] += 1
                    has_nan = True

                    # 记录详细信息
                    file_detail['columns'][col_name] = {
                        'total': total_count,
                        'nan_count': nan_count,
                        'nan_percentage': 100 * nan_count / total_count
                    }

            # 保存详细信息
            if has_nan and args.detailed:
                self.detailed_info[mode].append(file_detail)

            return has_nan

        except Exception as e:
            print(f"❌  读取文件失败 {file_path}: {e}")
            return False

    def print_summary(self):
        """打印汇总统计"""
        print(f"\n{'=' * 70}")
        print("NaN 检测汇总报告")
        print(f"{'=' * 70}")

        modes_to_check = []
        if args.mode in ['train', 'all']:
            modes_to_check.append('train')
        if args.mode in ['test', 'all']:
            modes_to_check.append('test')

        for mode in modes_to_check:
            if not self.stats[mode]:
                continue

            print(f"\n{'-' * 70}")
            print(f"{mode.upper()} 数据集统计")
            print(f"{'-' * 70}")

            # 按列名排序
            sorted_columns = sorted(self.stats[mode].keys())

            # 表头
            print(f"\n{'列名':<45} {'总数据点':<12} {'NaN数量':<12} {'NaN比例':<10} {'文件数':<8} {'含NaN文件'}")
            print(f"{'-' * 70}")

            total_data_points = 0
            total_nan_count = 0

            for col_name in sorted_columns:
                stats = self.stats[mode][col_name]
                total = stats['total']
                nan_count = stats['nan_count']
                files_checked = stats['files_checked']
                files_with_nan = stats['files_with_nan']

                total_data_points += total
                total_nan_count += nan_count

                nan_percentage = 100 * nan_count / total if total > 0 else 0

                # 根据NaN比例着色显示
                if nan_count == 0:
                    status = "✓"
                elif nan_percentage < 1:
                    status = "⚠️"
                else:
                    status = "❌"

                print(
                    f"{col_name:<45} {total:<12} {nan_count:<12} {nan_percentage:>8.2f}% {files_checked:<8} {files_with_nan} {status}")

            # 总计
            print(f"{'-' * 70}")
            overall_percentage = 100 * total_nan_count / total_data_points if total_data_points > 0 else 0
            print(f"{'总计':<45} {total_data_points:<12} {total_nan_count:<12} {overall_percentage:>8.2f}%")
            print(f"{'-' * 70}")

            # 汇总信息
            clean_columns = sum(1 for stats in self.stats[mode].values() if stats['nan_count'] == 0)
            total_columns = len(self.stats[mode])

            print(f"\n汇总:")
            print(f"  总列数: {total_columns}")
            print(f"  无NaN列数: {clean_columns}")
            print(f"  含NaN列数: {total_columns - clean_columns}")

            if total_nan_count == 0:
                print(f"\n✓✓✓ {mode.upper()} 数据集完全干净，没有发现NaN！")
            elif overall_percentage < 0.1:
                print(f"\n✓ {mode.upper()} 数据集基本干净，NaN比例很低 ({overall_percentage:.2f}%)")
            elif overall_percentage < 1:
                print(f"\n⚠️  {mode.upper()} 数据集有少量NaN ({overall_percentage:.2f}%)")
            else:
                print(f"\n❌ {mode.upper()} 数据集包含较多NaN ({overall_percentage:.2f}%)，建议清理数据")

    def print_detailed_info(self):
        """打印详细的文件信息"""
        if not args.detailed:
            return

        modes_to_check = []
        if args.mode in ['train', 'all']:
            modes_to_check.append('train')
        if args.mode in ['test', 'all']:
            modes_to_check.append('test')

        for mode in modes_to_check:
            if not self.detailed_info[mode]:
                continue

            print(f"\n{'=' * 70}")
            print(f"{mode.upper()} 数据集 - 包含NaN的文件详情")
            print(f"{'=' * 70}")

            for file_info in self.detailed_info[mode]:
                print(f"\n文件: {file_info['participant']}/{file_info['trial']}")
                print(f"路径: {file_info['file']}")
                print(f"包含NaN的列:")

                for col_name, col_stats in file_info['columns'].items():
                    print(f"  - {col_name}: {col_stats['nan_count']}/{col_stats['total']} "
                          f"({col_stats['nan_percentage']:.2f}%)")

    def export_report(self, output_file: str):
        """导出报告到文件"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 重定向打印输出
                import sys
                original_stdout = sys.stdout
                sys.stdout = f

                self.print_summary()
                if args.detailed:
                    self.print_detailed_info()

                sys.stdout = original_stdout

            print(f"\n✓ 报告已导出到: {output_file}")
        except Exception as e:
            print(f"\n❌ 导出报告失败: {e}")


def print_header():
    """打印标题"""
    print("\n" + "=" * 70)
    print(" " * 20 + "数据集 NaN 检测工具")
    print("=" * 70)
    print(f"配置文件: {args.config_path}")
    print(f"数据目录: {config.data_dir}")
    print(f"检测模式: {args.mode}")
    print(f"输入特征数: {len(config.input_names)}")
    print(f"身体侧别: {config.side}")


def main():
    # 打印标题
    print_header()

    # 创建检测器
    checker = NaNChecker(
        data_dir=config.data_dir,
        input_names=config.input_names,
        side=config.side
    )

    # 执行检测
    total_files = 0
    total_files_with_nan = 0

    if args.mode in ['train', 'all']:
        files, files_nan = checker.check_mode('train')
        total_files += files
        total_files_with_nan += files_nan

    if args.mode in ['test', 'all']:
        files, files_nan = checker.check_mode('test')
        total_files += files
        total_files_with_nan += files_nan

    # 打印汇总
    checker.print_summary()

    # 打印详细信息
    if args.detailed:
        checker.print_detailed_info()

    # 导出报告
    if args.export:
        checker.export_report(args.export)

    # 最终总结
    print(f"\n{'=' * 70}")
    print("检测完成")
    print(f"{'=' * 70}")
    print(f"总文件数: {total_files}")
    print(f"包含NaN的文件数: {total_files_with_nan}")

    if total_files_with_nan == 0:
        print("\n✓✓✓ 所有数据集都是干净的！")
    else:
        print(f"\n⚠️  发现 {total_files_with_nan} 个文件包含NaN，建议进一步检查")

    print("\n提示:")
    print("  - 使用 --detailed 参数查看每个文件的详细信息")
    print("  - 使用 --export report.txt 将报告导出到文件")
    print("  - 使用 --mode train/test 只检查特定数据集")


if __name__ == "__main__":
    main()
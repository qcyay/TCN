import sys
import os
import re
import argparse
from typing import List, Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class TcnDataset(Dataset):
    '''
    TCN数据集 - 预加载版本
    在初始化时预加载所有数据到内存，提高训练稳定性和速度
    '''

    def __init__(self,
                 data_dir: str,
                 input_names: List[str],
                 label_names: List[str],
                 side: str,
                 participant_masses: Dict[str, float] = {},
                 device: torch.device = torch.device("cpu"),
                 mode: str = "train",
                 file_suffix: Dict[str, str] = None,
                 remove_nan: bool = True,
                 action_patterns: Optional[List[str]] = None,
                 enable_action_filter: bool = False,
                 activity_flag: bool = False):
        """
        初始化数据集，支持训练/测试模式和新文件结构
        在初始化时预加载所有数据到内存

        参数:
            data_dir: 数据根目录路径
            input_names: 输入特征列名列表
            label_names: 标签列名列表
            side: 身体侧别 ('l' 或 'r')
            participant_masses: 参与者体重字典
            device: 计算设备（用于记录）
            mode: 数据集模式，'train' 或 'test'
            file_suffix: 文件后缀映射字典，默认为 None 时使用预设值
            remove_nan: 是否自动检测并移除包含NaN的行（默认True）
            action_patterns: 运动类型筛选的正则表达式列表
            enable_action_filter: 是否启用action_patterns筛选
            activity_flag: 是否启用activity_flag掩码功能（默认False）
        """
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.participant_masses = participant_masses
        self.device = device
        self.mode = mode.lower()
        self.remove_nan = remove_nan
        self.action_patterns = action_patterns
        self.enable_action_filter = enable_action_filter
        self.activity_flag = activity_flag

        # 设置文件后缀映射
        if file_suffix is None:
            self.file_suffix = {
                "input": "_exo.csv",
                "label": "_moment_filt.csv",
                "flag":"_activity_flag.csv"}
        else:
            self.file_suffix = file_suffix

        # 获取试验名称列表
        self.trial_names = self._get_trial_names()

        # ## 测试,正式训练时该行需要注释
        # self.trial_names = self.trial_names[:10]

        # 统计信息
        self.nan_removal_stats = {
            'trials_with_nan': 0,
            'total_rows_removed': 0,
            'trials_processed': 0,
            'trials_with_all_nan_labels': 0  # 新增：标签全为NaN的试验数
        }

        filter_status = "启用" if self.enable_action_filter else "禁用"
        print(f"开始加载 {self.mode} 数据集 (TCN)...")
        print(f"找到 {len(self.trial_names)} 个试验 (动作筛选: {filter_status})")

        # === 预加载所有数据到内存（关键修改）===
        self.all_input_data = []  # 存储所有试验的输入数据
        self.all_label_data = []  # 存储所有试验的标签数据
        self.trial_lengths = []  # 存储每个试验的原始长度
        self.all_activity_mask = []  # 存储所有试验的activity flag掩码
        self._preload_all_data()
        breakpoint()

        # === 检测并移除标签全为NaN的序列 ===
        self._remove_invalid_label_sequences()

        # # 检测并移除包含NaN的标签序列
        # self._remove_label_sequences_with_any_nan()

        print(f"数据集初始化完成 - 模式: {self.mode}, 试验数量: {len(self.trial_names)}")
        if self.remove_nan:
            self.print_nan_removal_summary()

    def __len__(self):
        '''返回数据集中试验的总数'''
        return len(self.trial_names)

    def __getitem__(self, idx: int):
        '''
        根据提供的索引获取预加载的数据
        返回格式与原版相同，保持与collate_fn_tcn的兼容性

        参数:
            idx: 整数索引

        返回:
            input_data: 输入数据张量 [batch_size, num_input_features, sequence_length]
            label_data: 标签数据张量 [batch_size, num_label_features, sequence_length]
            trial_sequence_lengths: 每个试验的原始长度列表
            activity_masks: activity flag掩码列表 (如果启用) 或 None
        '''

        # 从numpy转换为tensor
        # 尺寸为[C,N]
        input_data = torch.from_numpy(self.all_input_data[idx]).float()
        #尺寸为[N_label,N]
        label_data = torch.from_numpy(self.all_label_data[idx]).float()
        if self.activity_flag:
            #尺寸为[N]
            activity_mask = torch.from_numpy(self.all_activity_mask[idx]).float()
        else:
            activity_mask = None

        trial_length = [self.trial_lengths[idx]]

        # print(f"加载试验名称: {self.trial_names[i]}， 尺寸为 {input_data.size()}")

        return input_data, label_data, trial_length, activity_mask

    def get_trial_names(self):
        '''返回所有试验名称'''
        return self.trial_names

    def get_mode(self):
        '''返回当前数据集模式'''
        return self.mode

    def get_nan_removal_stats(self):
        '''返回NaN移除统计信息'''
        return self.nan_removal_stats

    def _match_action_patterns(self, trial_name: str) -> bool:
        """
        检查试验名称是否匹配任何一个action_pattern

        参数:
            trial_name: 试验名称，格式为 "参与者/运动类型"

        返回:
            如果匹配返回True，否则返回False
        """
        if not self.enable_action_filter or not self.action_patterns:
            return True

        # 提取试验项目名称（不包含参与者名称）
        trial_basename = os.path.basename(trial_name)

        # 遍历所有pattern，如果任何一个匹配就返回True
        for pattern_line in self.action_patterns:
            # 每一行可能包含多个正则表达式，用逗号分隔
            patterns = [p.strip() for p in pattern_line.split(',')]

            for pattern in patterns:
                if pattern and re.match(pattern, trial_basename):
                    return True

        return False

    def _get_trial_names(self):
        '''
        扫描数据目录，获取所有试验名称。
        新目录结构: data_dir/(train/test)/人名/运动状态/
        支持基于action_patterns的筛选
        '''
        # 构建模式子目录路径
        mode_dir = os.path.join(self.data_dir, self.mode)

        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"模式目录不存在: {mode_dir}")

        # 获取参与者目录（排除隐藏文件和无关文件）
        participants = []
        for item in os.listdir(mode_dir):
            item_path = os.path.join(mode_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # 包含参与者目录的列表
                participants.append(item)

        if not participants:
            raise ValueError(f"在目录 {mode_dir} 中未找到参与者数据")

        # 遍历参与者目录，收集试验名称
        trial_names = []
        filtered_out_count = 0

        for participant in participants:
            participant_dir = os.path.join(mode_dir, participant)

            # 获取参与者的所有试验
            for trial_item in os.listdir(participant_dir):
                trial_path = os.path.join(participant_dir, trial_item)
                if os.path.isdir(trial_path) and not trial_item.startswith('.'):
                    # 试验名称格式: 参与者/试验项目
                    trial_name = os.path.join(participant, trial_item)

                    # 检查是否匹配action_patterns
                    if self._match_action_patterns(trial_name):
                        # 包含试验名称的列表
                        trial_names.append(trial_name)
                    else:
                        filtered_out_count += 1

        if not trial_names:
            raise ValueError(f"在参与者目录中未找到试验数据（可能被action_patterns过滤）")

        if self.enable_action_filter and filtered_out_count > 0:
            print(f"过滤掉 {filtered_out_count} 个不匹配的试验")

        return trial_names

    def _find_valid_range(self, df: pd.DataFrame, columns: List[str]) -> Tuple[int, int]:
        """
        在指定列中查找有效数据范围(不含NaN的行范围)

        重要: 此方法会检测并移除文件**开头**和**结尾**的NaN行

        参数:
            df: DataFrame数据
            columns: 要检查的列名列表

        返回:
            (start_index, end_index): 有效数据的起始和结束索引(前闭后开区间)
            如果所有行都包含NaN,返回 (0, 0)
        """
        # 检查指定列是否存在
        valid_columns = [col for col in columns if col in df.columns]

        if not valid_columns:
            # 如果没有有效列，返回全部数据范围
            return 0, len(df)

        # 检查指定列中是否有NaN
        subset_df = df[valid_columns]
        nan_mask = subset_df.isna().any(axis=1)
        valid_mask = ~nan_mask

        # 获取所有有效行的索引
        valid_indices = valid_mask[valid_mask].index.tolist()

        if not valid_indices:
            # 所有行都包含NaN
            return 0, 0

        # 找到第一个和最后一个有效行
        start_index = valid_indices[0]
        end_index = valid_indices[-1] + 1  # +1 because we use [start:end) slicing

        return start_index, end_index

    def _preload_all_data(self):
        """
        预加载所有试验数据到内存（关键优化）
        这样只需要在初始化时读取一次CSV文件
        """
        print("预加载所有试验数据到内存...")

        for trial_idx, trial_name in enumerate(self.trial_names):
            if (trial_idx + 1) % 10 == 0 or (trial_idx + 1) == len(self.trial_names):
                print(f"  加载进度: {trial_idx + 1}/{len(self.trial_names)}")

            # 加载单个试验数据
            input_data, label_data, activity_mask = self._load_trial_data(trial_name)

            # 检查数据有效性
            if torch.isnan(input_data).any():
                print(f"  警告: {trial_name} 包含NaN值")

            # 存储为numpy数组（更节省内存）
            self.all_input_data.append(input_data.numpy())
            self.all_label_data.append(label_data.numpy())
            if activity_mask is not None:
                self.all_activity_mask.append(activity_mask.numpy())

            # 记录原始长度
            self.trial_lengths.append(input_data.shape[1])
            if input_data.shape[1] == 0:
                break

        print("所有数据预加载完成!")

    def _remove_invalid_label_sequences(self):
        """
        检测并移除标签全为NaN的序列
        同时更新trial_names、all_input_data、all_label_data和trial_lengths
        """
        if not self.remove_nan:
            return

        print("检测标签全为NaN的序列...")

        valid_indices = []
        removed_trials = []

        for idx in range(len(self.trial_names)):
            label_data = self.all_label_data[idx]

            # 检查标签数据是否全为NaN
            if isinstance(label_data, np.ndarray):
                is_all_nan = np.all(np.isnan(label_data))
            else:  # torch.Tensor
                is_all_nan = torch.all(torch.isnan(label_data)).item()

            if is_all_nan:
                removed_trials.append(self.trial_names[idx])
                self.nan_removal_stats['trials_with_all_nan_labels'] += 1
            else:
                valid_indices.append(idx)

        # 如果有需要移除的试验
        if removed_trials:
            print(f"  发现 {len(removed_trials)} 个标签全为NaN的试验，正在移除...")
            for trial_name in removed_trials:
                print(f"    - {trial_name}")

            # 更新所有相关列表
            self.trial_names = [self.trial_names[i] for i in valid_indices]
            self.all_input_data = [self.all_input_data[i] for i in valid_indices]
            self.all_label_data = [self.all_label_data[i] for i in valid_indices]
            self.trial_lengths = [self.trial_lengths[i] for i in valid_indices]
            self.all_activity_mask = [self.all_activity_mask[i] for i in valid_indices]

            print(f"  移除完成，剩余 {len(self.trial_names)} 个有效试验")
        else:
            print("  未发现标签全为NaN的序列")

    def _remove_label_sequences_with_any_nan(self):
        """
        检测并移除【标签中含有 NaN】的序列
        只要标签中出现 NaN，就移除对应 trial
        同时更新 trial_names、all_input_data、all_label_data 和 trial_lengths
        """
        if not self.remove_nan:
            return

        print("检测标签中含有 NaN 的序列...")

        valid_indices = []
        removed_trials = []

        # 如果你有统计信息，可以先确保这个 key 存在
        if hasattr(self, "nan_removal_stats"):
            self.nan_removal_stats.setdefault('trials_with_any_nan_labels', 0)

        for idx in range(len(self.trial_names)):
            label_data = self.all_label_data[idx]

            # 检查标签数据是否“包含 NaN”
            if isinstance(label_data, np.ndarray):
                has_nan = np.any(np.isnan(label_data))
            else:  # torch.Tensor
                has_nan = torch.isnan(label_data).any().item()

            if has_nan:
                removed_trials.append(self.trial_names[idx])
                if hasattr(self, "nan_removal_stats"):
                    self.nan_removal_stats['trials_with_any_nan_labels'] += 1
            else:
                valid_indices.append(idx)

        # 如果有需要移除的试验
        if removed_trials:
            print(f"  发现 {len(removed_trials)} 个标签中含有 NaN 的试验，正在移除...")
            for trial_name in removed_trials:
                print(f"    - {trial_name}")

            # 更新所有相关列表
            self.trial_names = [self.trial_names[i] for i in valid_indices]
            self.all_input_data = [self.all_input_data[i] for i in valid_indices]
            self.all_label_data = [self.all_label_data[i] for i in valid_indices]
            self.trial_lengths = [self.trial_lengths[i] for i in valid_indices]

            print(f"  移除完成，剩余 {len(self.trial_names)} 个有效试验")
        else:
            print("  未发现标签中含有 NaN 的序列")

    def _load_trial_data(self, trial_name: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        '''
        从单个试验目录加载数据，适配新的文件命名格式
        支持自动检测和移除包含NaN的行（包括文件开头和结尾）

        参数:
            trial_name: 试验名称，格式为 "参与者/试验项目"

        返回:
            input_data: [num_input_features, sequence_length]
            label_data: [num_label_features, sequence_length]
            activity_mask: [sequence_length] or None
        '''
        # 构建完整的试验目录路径
        trial_dir = os.path.join(self.data_dir, self.mode, trial_name)

        if not os.path.exists(trial_dir):
            raise FileNotFoundError(f"试验目录不存在: {trial_dir}")

        # 从试验名称中提取参与者姓名
        participant = trial_name.split("/")[0].split("\\")[0]

        if participant not in self.participant_masses:
            print(f"  警告: 参与者 {participant} 的体重信息未提供")

        # 构建新的文件名（基于参与者姓名和试验项目）
        file_prefix = f"{participant}_{os.path.basename(trial_name)}"

        # 构建输入和标签文件路径（新命名格式）
        input_filename = f"{file_prefix}{self.file_suffix['input']}"
        label_filename = f"{file_prefix}{self.file_suffix['label']}"

        input_file_path = os.path.join(trial_dir, input_filename)
        label_file_path = os.path.join(trial_dir, label_filename)

        # 检查文件是否存在
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"输入文件不存在: {input_file_path}")

        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"标签文件不存在: {label_file_path}")

        # 加载输入数据
        # input_data,输入传感器数据，尺寸为[num_input_features, sequence_length],valid_range，有效数据范围，[start,end]
        input_data, valid_range = self._load_input_data(input_file_path,
                                                        body_mass=self.participant_masses.get(participant, 1.0))

        # 加载标签数据（使用相同的有效范围）
        # label_data,标签数据，尺寸为[num_label_features, sequence_length]
        label_data = self._load_label_data(label_file_path, valid_range=valid_range)

        # 加载activity_flag数据（如果启用）
        activity_mask = None
        if self.activity_flag:
            activity_filename = f"{file_prefix}{self.file_suffix['flag']}"
            activity_file_path = os.path.join(trial_dir, activity_filename)

            if os.path.exists(activity_file_path):
                # activity_mask，数据是否有效的掩码，[sequence_length]
                activity_mask = self._load_activity_flag(activity_file_path, valid_range=valid_range)
            else:
                raise FileNotFoundError(f"  警告: activity_flag已启用但文件不存在: {activity_file_path}")

        # 更新统计信息
        self.nan_removal_stats['trials_processed'] += 1
        if valid_range is not None:
            self.nan_removal_stats['trials_with_nan'] += 1

        return input_data, label_data, activity_mask

    def _load_input_data(self, file_path: str, body_mass: float) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        '''
        加载并预处理输入数据CSV文件
        支持自动检测和移除包含NaN的行（包括文件开头和结尾）

        返回:
            input_data: 处理后的输入数据张量 [num_input_features, sequence_length]
            valid_range: 有效数据范围（如果有截断），或None
        '''
        # 读取CSV文件
        df = pd.read_csv(file_path)
        # breakpoint()
        original_length = len(df)

        # 如果启用NaN移除，检测并截断
        valid_range = None
        if self.remove_nan:
            start_idx, end_idx = self._find_valid_range(df, self.input_names)

            if end_idx == 0:
                # 所有行都是NaN
                print(f"  警告: {file_path} 所有行都包含NaN，返回空张量")
                return torch.zeros((len(self.input_names), 0), dtype=torch.float32), (start_idx, end_idx)

            if start_idx > 0 or end_idx < original_length:
                # 需要截断
                df = df.iloc[start_idx:end_idx].copy()
                valid_range = (start_idx, end_idx)
                removed_rows = (start_idx) + (original_length - end_idx)
                self.nan_removal_stats['total_rows_removed'] += removed_rows

        # 体重标准化压力鞋垫数据
        if "insole_l_force_y" in df.columns:
            df.loc[:, "insole_l_force_y"] /= body_mass
        if "insole_r_force_y" in df.columns:
            df.loc[:, "insole_r_force_y"] /= body_mass

        # 左侧身体数据镜像处理
        if self.side == "l":
            mirror_columns = ["foot_imu_l_gyro_x", "foot_imu_l_gyro_y", "foot_imu_l_accel_z",
                              "shank_imu_l_gyro_x", "shank_imu_l_gyro_y", "shank_imu_l_accel_z",
                              "thigh_imu_l_gyro_x", "thigh_imu_l_gyro_y", "thigh_imu_l_accel_z",
                              "insole_l_cop_z"]
            for col in mirror_columns:
                if col in df.columns:
                    df.loc[:, col] *= -1.0

        # 检查是否有缺失的必需列
        missing_cols = [col for col in self.input_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺失必需的列: {missing_cols}")

        # 提取数据并转换为tensor
        extracted_data = df[self.input_names].values
        input_data = torch.tensor(extracted_data, dtype=torch.float32).transpose(0, 1)

        return input_data, valid_range

    def _load_label_data(self, file_path: str, valid_range: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        '''
        加载标签数据CSV文件

        参数:
            file_path: 标签文件路径
            valid_range: 如果不为None，使用指定的有效范围以匹配输入数据

        返回:
            label_data: [num_label_features, sequence_length]
        '''
        df = pd.read_csv(file_path)

        # 如果指定了有效范围，使用该范围截断标签数据
        if valid_range is not None:
            start_idx, end_idx = valid_range
            if end_idx == 0:
                # 返回空张量
                return torch.zeros((len(self.label_names), 0), dtype=torch.float32)
            df = df.iloc[start_idx:end_idx].copy()

        # 检查是否有缺失的必需列
        missing_cols = [col for col in self.label_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"标签数据缺失必需的列: {missing_cols}")

        # 提取数据并转换为tensor
        extracted_data = df[self.label_names].values
        label_data = torch.tensor(extracted_data, dtype=torch.float32).transpose(0, 1)

        return label_data

    def _load_activity_flag(self, file_path: str, valid_range: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        '''
        加载activity_flag数据CSV文件

        参数:
            file_path: activity_flag文件路径
            valid_range: 如果不为None，使用指定的有效范围以匹配输入数据

        返回:
            activity_mask: [sequence_length] 值为0或1的掩码
        '''
        df = pd.read_csv(file_path)

        # 根据side选择列名
        side_column = "left" if self.side == "l" else "right"

        if side_column not in df.columns:
            raise ValueError(f"activity_flag文件缺失必需的列: {side_column}")

        # 如果指定了有效范围，截断activity_flag数据以匹配输入数据
        if valid_range is not None:
            start_idx, end_idx = valid_range
            if end_idx == 0:
                # 返回空张量
                return torch.zeros(0, dtype=torch.float32)
            df = df.iloc[start_idx:end_idx].copy()

        # 提取数据并转换为tensor
        activity_mask = torch.tensor(df[side_column].values, dtype=torch.float32)

        return activity_mask

    def print_nan_removal_summary(self):
        """打印NaN移除统计摘要"""
        stats = self.nan_removal_stats
        print(f"\n{'=' * 60}")
        print(f"NaN移除统计摘要 - {self.mode.upper()} 数据集")
        print(f"{'=' * 60}")
        print(f"处理的试验总数: {stats['trials_processed']}")
        print(f"包含NaN的试验数: {stats['trials_with_nan']}")
        print(f"移除的数据行总数: {stats['total_rows_removed']}")
        print(f"标签全为NaN的试验数: {stats['trials_with_all_nan_labels']}")

        if stats['trials_processed'] > 0:
            nan_trial_percentage = 100 * stats['trials_with_nan'] / stats['trials_processed']
            print(f"包含NaN的试验比例: {nan_trial_percentage:.2f}%")

        if stats['trials_with_nan'] > 0:
            avg_rows_removed = stats['total_rows_removed'] / stats['trials_with_nan']
            print(f"平均每个含NaN试验移除的行数: {avg_rows_removed:.1f}")

        print(f"{'=' * 60}\n")


def main():
    sys.path.insert(0, '.')
    from torch.utils.data import DataLoader
    from utils.config_utils import load_config, apply_feature_selection
    from utils.utils import collate_fn_tcn

    parser = argparse.ArgumentParser(
        description="快速测试 TcnDataset 数据加载流程，打印样本形状和统计信息。"
    )
    parser.add_argument("--config", type=str, default="configs.default_config",
                        help="配置文件模块路径，默认 configs.default_config")
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="选择加载训练集或测试集")
    parser.add_argument("--device", type=str, default="cpu",
                        help="将数据加载到的设备，如 cpu 或 cuda:0")
    parser.add_argument("--max_trials", type=int, default=3,
                        help="最多打印多少个样本的尺寸信息")

    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_feature_selection(config)

    # 替换配置中的通配符
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    device = torch.device(args.device)

    tcn_kwargs = dict(data_dir=config.data_dir, input_names=input_names, label_names=label_names, side=config.side,
                      participant_masses=config.participant_masses, device=device, action_patterns=getattr(config, 'action_patterns', None),
                      enable_action_filter=getattr(config, 'enable_action_filter', False), activity_flag=config.activity_flag)
    # print(tcn_kwargs)

    dataset = TcnDataset(mode='train', **tcn_kwargs)

    dataset_size = len(dataset)
    lengths = getattr(dataset, "trial_lengths", [])

    print("\n" + "=" * 70)
    print("TCN 数据集测试概览")
    print("=" * 70)
    print(f"模式: {args.mode}")
    print(f"试验数量: {dataset_size}")
    print(f"输入特征数: {len(input_names)}, 标签特征数: {len(label_names)}")
    if lengths:
        print(f"序列长度统计 (帧) -> 平均 {np.mean(lengths):.1f}, "
              f"中位 {np.median(lengths):.1f}, 范围 [{np.min(lengths)}, {np.max(lengths)}]")
    else:
        print("序列长度信息不可用。")
    print("=" * 70 + "\n")

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn_tcn)

    print("示例样本：")
    for batch_idx, batch in enumerate(data_loader):
        print(f"批次索引: {batch_idx}")
        inputs = batch[0]
        labels = batch[1]
        seq_lengths = batch[2]
        masks = batch[3]

        print(f'输入传感器数据形状：{inputs.size()}')
        print(f'输入标签数据形状：{labels.size()}')
        print(f'输入数据长度：{seq_lengths}')
        if masks[0] is not None:
            for i in range(len(masks)):
                print(f'输入掩码数据形状：{masks[i].size()}')

        if batch_idx >= args.max_trials:
            break

if __name__ == "__main__":
    main()

import os
import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset


class TcnDataset(Dataset):
    '''Dataset for dynamically loading input and label data based on indices with train/test mode support.'''

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
                 load_to_device: bool = False,
                 action_patterns: Optional[List[str]] = None,
                 enable_action_filter: bool = False):
        """
        初始化数据集，支持训练/测试模式和新文件结构

        参数:
            data_dir: 数据根目录路径
            input_names: 输入特征列名列表
            label_names: 标签列名列表
            side: 身体侧别 ('l' 或 'r')
            participant_masses: 参与者体重字典
            device: 计算设备（用于记录，实际加载位置由load_to_device控制）
            mode: 数据集模式，'train' 或 'test'
            file_suffix: 文件后缀映射字典，默认为 None 时使用预设值
            remove_nan: 是否自动检测并移除包含NaN的行（默认True）
            load_to_device: 是否在__getitem__中直接加载到device（False时返回CPU tensor，支持多进程）
            action_patterns: 运动类型筛选的正则表达式列表
            enable_action_filter: 是否启用action_patterns筛选
        """
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.participant_masses = participant_masses
        self.device = device
        self.mode = mode.lower()
        self.remove_nan = remove_nan
        self.load_to_device = load_to_device
        self.action_patterns = action_patterns
        self.enable_action_filter = enable_action_filter

        # 设置文件后缀映射
        if file_suffix is None:
            self.file_suffix = {
                "input": "_exo.csv",
                "label": "_moment_filt.csv"
            }
        else:
            self.file_suffix = file_suffix

        # 验证模式参数
        if self.mode not in ["train", "test"]:
            raise ValueError(f"模式参数必须是 'train' 或 'test'，当前为: {mode}")

        # 获取试验名称列表
        self.trial_names = self._get_trial_names()

        # 统计信息
        self.nan_removal_stats = {
            'trials_with_nan': 0,
            'total_rows_removed': 0,
            'trials_processed': 0
        }

        load_mode = "GPU" if self.load_to_device else "CPU"
        filter_status = "启用" if self.enable_action_filter else "禁用"
        print(f"数据集初始化完成 - 模式: {self.mode}, 试验数量: {len(self.trial_names)}, "
              f"加载模式: {load_mode}, 动作筛选: {filter_status}")
        if self.remove_nan:
            print(f"NaN自动移除: 启用 (将移除文件开头和结尾的NaN行)")

    def __len__(self):
        '''返回数据集中试验的总数'''
        return len(self.trial_names)

    def __getitem__(self, idx: int or List[int] or slice):
        '''
        根据提供的索引加载一个或多个试验的数据。
        使用零填充将不同长度的试验数据统一长度以便批量处理。

        参数:
            idx: 整数索引、索引列表或切片对象

        返回:
            input_data: 填充后的输入数据张量 [batch_size, num_input_features, max_sequence_length]
            label_data: 填充后的标签数据张量 [batch_size, num_label_features, max_sequence_length]
            trial_sequence_lengths: 每个试验的原始长度列表
        '''
        # 根据索引类型获取试验名称列表
        if isinstance(idx, list):
            trial_names = [self.trial_names[i] for i in idx]
        else:
            trial_names = self.trial_names[idx]
            trial_names = [trial_names] if not isinstance(trial_names, list) else trial_names

        # 加载试验数据
        data = [list(self._load_trial_data(trial_name)) for trial_name in trial_names]

        # 零填充处理
        data, trial_sequence_lengths = self._add_zero_padding(data)

        # 拼接张量
        input_data, label_data = zip(*data)
        input_data = torch.cat(input_data, dim=0)
        label_data = torch.cat(label_data, dim=0)

        return input_data, label_data, trial_sequence_lengths

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
            trial_name: 试验名称，格式为 "参与者/试验项目"

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

        print(f"扫描目录: {mode_dir}")

        # 获取参与者目录（排除隐藏文件和无关文件）
        participants = []
        for item in os.listdir(mode_dir):
            item_path = os.path.join(mode_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
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
                        trial_names.append(trial_name)
                    else:
                        filtered_out_count += 1

        if not trial_names:
            raise ValueError(f"在参与者目录中未找到试验数据（可能被action_patterns过滤）")

        print(f"找到 {len(trial_names)} 个试验", end="")
        if self.enable_action_filter and filtered_out_count > 0:
            print(f" (过滤掉 {filtered_out_count} 个不匹配的试验)")
        else:
            print()

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
            print(f"    警告: 指定的列都不存在于数据中")
            return 0, len(df)

        # 只检查有效列
        subset_df = df[valid_columns]

        # 检查每一行是否包含NaN
        nan_mask = subset_df.isna().any(axis=1)

        # 找到所有不含NaN的行
        valid_mask = ~nan_mask
        valid_indices = valid_mask[valid_mask].index.tolist()

        if not valid_indices:
            # 所有行都包含NaN
            print(f"    警告: 所有行都包含NaN!")
            return 0, 0

        # 找到第一个和最后一个有效行
        start_index = valid_indices[0]
        end_index = valid_indices[-1] + 1  # +1 because we use [start:end) slicing

        if start_index > 0 or end_index < len(df):
            num_removed_front = start_index
            num_removed_back = len(df) - end_index
            # print(f"    检测到NaN: 移除前{num_removed_front}行, 后{num_removed_back}行")

        return start_index, end_index

    def _load_trial_data(self, trial_name: str):
        '''
        从单个试验目录加载数据，适配新的文件命名格式
        支持自动检测和移除包含NaN的行（包括文件开头和结尾）

        参数:
            trial_name: 试验名称，格式为 "参与者/试验项目"
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
        trial_basename = os.path.basename(trial_name)
        file_prefix = f"{participant}_{trial_basename}"

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
        input_data, valid_range = self._load_input_data(
            input_file_path,
            body_mass=self.participant_masses.get(participant, 1.0)
        )
        # if valid_range is not None:
        #     print(valid_range)
        #     print(f"  {trial_dir}  输入数据长度: {input_data.size()}")
        if torch.isnan(input_data).any():
            print(f"警告：{trial_dir}中张量包含NaN值")

        # 加载标签数据（使用相同的有效范围）
        label_data = self._load_label_data(label_file_path, valid_range=valid_range)

        # 更新统计信息
        self.nan_removal_stats['trials_processed'] += 1
        if valid_range is not None:
            self.nan_removal_stats['trials_with_nan'] += 1

        return input_data, label_data

    def _load_input_data(self, file_path: str, body_mass: float):
        '''
        加载并预处理输入数据CSV文件
        支持自动检测和移除包含NaN的行（包括文件开头和结尾）

        返回:
            input_data: 处理后的输入数据张量
            valid_range: 有效数据范围（如果有截断），或None
        '''
        # 读取CSV文件
        df = pd.read_csv(file_path)
        original_length = len(df)

        # 如果启用NaN移除，检测并截断
        valid_range = None
        if self.remove_nan:
            start_idx, end_idx = self._find_valid_range(df, self.input_names)

            if end_idx == 0:
                # 所有行都是NaN
                print(f"  警告: {file_path} 所有行都包含NaN，跳过该文件")
                # 返回空张量
                target_device = self.device if self.load_to_device else torch.device('cpu')
                empty_tensor = torch.zeros((1, len(self.input_names), 0), device=target_device)
                return empty_tensor, (start_idx, end_idx)

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
            mirror_columns = [
                "foot_imu_l_gyro_x", "foot_imu_l_gyro_y", "foot_imu_l_accel_z",
                "shank_imu_l_gyro_x", "shank_imu_l_gyro_y", "shank_imu_l_accel_z",
                "thigh_imu_l_gyro_x", "thigh_imu_l_gyro_y", "thigh_imu_l_accel_z",
                "insole_l_cop_z"
            ]
            for col in mirror_columns:
                if col in df.columns:
                    df.loc[:, col] *= -1.0

        # 检查是否有缺失的必需列
        missing_cols = [col for col in self.input_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺失必需的列: {missing_cols}")

        # 提取数据
        extracted_data = df[self.input_names].values

        # 根据load_to_device决定加载位置
        target_device = self.device if self.load_to_device else torch.device('cpu')

        # 转换为PyTorch张量并调整维度
        input_data = torch.tensor(
            extracted_data,
            device=target_device
        ).transpose(0, 1).unsqueeze(0).float()

        return input_data, valid_range

    def _load_label_data(self, file_path: str, valid_range: Optional[Tuple[int, int]] = None):
        '''
        加载标签数据CSV文件

        参数:
            file_path: 标签文件路径
            valid_range: 如果不为None，使用指定的有效范围以匹配输入数据
        '''
        df = pd.read_csv(file_path)

        # 如果指定了有效范围，使用该范围截断标签数据
        if valid_range is not None:
            start_idx, end_idx = valid_range
            if end_idx == 0:
                # 返回空张量
                target_device = self.device if self.load_to_device else torch.device('cpu')
                empty_tensor = torch.zeros((1, len(self.label_names), 0), device=target_device)
                return empty_tensor
            df = df.iloc[start_idx:end_idx].copy()

        # 检查是否有缺失的必需列
        missing_cols = [col for col in self.label_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"标签数据缺失必需的列: {missing_cols}")

        # 提取数据
        extracted_data = df[self.label_names].values

        # 根据load_to_device决定加载位置
        target_device = self.device if self.load_to_device else torch.device('cpu')

        label_data = torch.tensor(
            extracted_data,
            device=target_device
        ).transpose(0, 1).unsqueeze(0).float()

        return label_data

    def _add_zero_padding(self, data: List[List[torch.FloatTensor]]):
        '''
        对试验数据进行零填充，使所有试验序列长度一致
        '''
        trial_sequence_lengths = [trial_data[0].shape[-1] for trial_data in data]
        max_sequence_length = max(trial_sequence_lengths) if trial_sequence_lengths else 0

        if max_sequence_length == 0:
            # 所有序列都是空的
            return data, trial_sequence_lengths

        # 对长度不足的试验进行填充
        for i in range(len(data)):
            trial_sequence_length = trial_sequence_lengths[i]
            if trial_sequence_length < max_sequence_length:
                padding_length = max_sequence_length - trial_sequence_length
                for j in range(len(data[i])):
                    padding = torch.zeros(
                        (1, data[i][j].shape[1], padding_length),
                        device=data[i][j].device
                    )
                    data[i][j] = torch.cat((data[i][j], padding), dim=2)

        return data, trial_sequence_lengths

    def print_nan_removal_summary(self):
        """打印NaN移除统计摘要"""
        stats = self.nan_removal_stats
        print(f"\n{'=' * 60}")
        print(f"NaN移除统计摘要 - {self.mode.upper()} 数据集")
        print(f"{'=' * 60}")
        print(f"处理的试验总数: {stats['trials_processed']}")
        print(f"包含NaN的试验数: {stats['trials_with_nan']}")
        print(f"移除的数据行总数: {stats['total_rows_removed']}")

        if stats['trials_processed'] > 0:
            nan_trial_percentage = 100 * stats['trials_with_nan'] / stats['trials_processed']
            print(f"包含NaN的试验比例: {nan_trial_percentage:.2f}%")

        if stats['trials_with_nan'] > 0:
            avg_rows_removed = stats['total_rows_removed'] / stats['trials_with_nan']
            print(f"平均每个含NaN试验移除的行数: {avg_rows_removed:.1f}")

        print(f"{'=' * 60}\n")
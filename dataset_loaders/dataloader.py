import os
import re
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
                 load_to_device: bool = False,
                 action_patterns: Optional[List[str]] = None,
                 enable_action_filter: bool = False):
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
            load_to_device: 是否在初始化时直接加载到device（False时存储为numpy数组）
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

        filter_status = "启用" if self.enable_action_filter else "禁用"
        print(f"开始加载 {self.mode} 数据集 (TCN)...")
        print(f"找到 {len(self.trial_names)} 个试验 (动作筛选: {filter_status})")

        # === 预加载所有数据到内存（关键修改）===
        self.all_input_data = []  # 存储所有试验的输入数据
        self.all_label_data = []  # 存储所有试验的标签数据
        self.trial_lengths = []  # 存储每个试验的原始长度
        self._preload_all_data()

        print(f"数据集初始化完成 - 模式: {self.mode}, 试验数量: {len(self.trial_names)}")
        if self.remove_nan:
            self.print_nan_removal_summary()

    def __len__(self):
        '''返回数据集中试验的总数'''
        return len(self.trial_names)

    def __getitem__(self, idx: int or List[int] or slice):
        '''
        根据提供的索引获取预加载的数据
        返回格式与原版相同，保持与collate_fn_tcn的兼容性

        参数:
            idx: 整数索引、索引列表或切片对象

        返回:
            input_data: 输入数据张量 [batch_size, num_input_features, sequence_length]
            label_data: 标签数据张量 [batch_size, num_label_features, sequence_length]
            trial_sequence_lengths: 每个试验的原始长度列表
        '''
        # 根据索引类型处理
        if isinstance(idx, int):
            indices = [idx]
        elif isinstance(idx, slice):
            indices = list(range(*idx.indices(len(self))))
        elif isinstance(idx, list):
            indices = idx
        else:
            raise TypeError(f"不支持的索引类型: {type(idx)}")

        # 从预加载的数据中提取
        batch_input_data = []
        batch_label_data = []
        batch_lengths = []

        for i in indices:
            # 直接从预加载的数据中获取
            if self.load_to_device:
                # 如果已经加载到device，直接使用
                input_data = self.all_input_data[i]
                label_data = self.all_label_data[i]
            else:
                # 从numpy转换为tensor
                input_data = torch.from_numpy(self.all_input_data[i]).float()
                label_data = torch.from_numpy(self.all_label_data[i]).float()

            # 添加batch维度 [C, N] -> [1, C, N]
            input_data = input_data.unsqueeze(0)
            label_data = label_data.unsqueeze(0)

            batch_input_data.append(input_data)
            batch_label_data.append(label_data)
            batch_lengths.append(self.trial_lengths[i])

        # 如果只请求一个样本，进行零填充以保持格式一致
        if len(batch_input_data) == 1:
            return batch_input_data[0], batch_label_data[0], batch_lengths

        # 多个样本：进行零填充
        padded_data, padded_lengths = self._add_zero_padding(
            list(zip(batch_input_data, batch_label_data))
        )

        # 拼接张量
        input_data = torch.cat([d[0] for d in padded_data], dim=0)
        label_data = torch.cat([d[1] for d in padded_data], dim=0)

        return input_data, label_data, padded_lengths

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
            input_data, label_data = self._load_trial_data(trial_name)

            # 检查数据有效性
            if torch.isnan(input_data).any():
                print(f"  警告: {trial_name} 包含NaN值")

            # 存储数据
            if self.load_to_device:
                # 直接存储为GPU tensor
                self.all_input_data.append(input_data.to(self.device))
                self.all_label_data.append(label_data.to(self.device))
            else:
                # 存储为numpy数组（更节省内存）
                self.all_input_data.append(input_data.numpy())
                self.all_label_data.append(label_data.numpy())

            # 记录原始长度
            self.trial_lengths.append(input_data.shape[1])

        print("所有数据预加载完成!")

    def _load_trial_data(self, trial_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        从单个试验目录加载数据，适配新的文件命名格式
        支持自动检测和移除包含NaN的行（包括文件开头和结尾）

        参数:
            trial_name: 试验名称，格式为 "参与者/试验项目"

        返回:
            input_data: [num_input_features, sequence_length]
            label_data: [num_label_features, sequence_length]
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

        # 加载标签数据（使用相同的有效范围）
        label_data = self._load_label_data(label_file_path, valid_range=valid_range)

        # 更新统计信息
        self.nan_removal_stats['trials_processed'] += 1
        if valid_range is not None:
            self.nan_removal_stats['trials_with_nan'] += 1

        return input_data, label_data

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

    def _add_zero_padding(self, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[
        List[Tuple[torch.Tensor, torch.Tensor]], List[int]]:
        '''
        对试验数据进行零填充，使所有试验序列长度一致

        参数:
            data: [(input_tensor, label_tensor), ...] 列表

        返回:
            padded_data: 填充后的数据列表
            trial_sequence_lengths: 原始长度列表
        '''
        trial_sequence_lengths = [trial_data[0].shape[-1] for trial_data in data]
        max_sequence_length = max(trial_sequence_lengths) if trial_sequence_lengths else 0

        if max_sequence_length == 0:
            # 所有序列都是空的
            return data, trial_sequence_lengths

        padded_data = []
        # 对长度不足的试验进行填充
        for i, (inp, lbl) in enumerate(data):
            trial_sequence_length = trial_sequence_lengths[i]
            if trial_sequence_length < max_sequence_length:
                padding_length = max_sequence_length - trial_sequence_length

                # 填充输入数据
                inp_pad = torch.zeros((inp.shape[0], inp.shape[1], padding_length), device=inp.device)
                inp_padded = torch.cat([inp, inp_pad], dim=2)

                # 填充标签数据
                lbl_pad = torch.zeros((lbl.shape[0], lbl.shape[1], padding_length), device=lbl.device)
                lbl_padded = torch.cat([lbl, lbl_pad], dim=2)

                padded_data.append((inp_padded, lbl_padded))
            else:
                padded_data.append((inp, lbl))

        return padded_data, trial_sequence_lengths

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
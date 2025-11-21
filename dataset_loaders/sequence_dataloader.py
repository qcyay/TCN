import os
import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class SequenceDataset(Dataset):
    '''
    优化的通用序列数据集，支持预测模型和生成式模型
    - 在初始化时预加载所有数据到内存
    - 使用numpy数组存储数据以提高访问速度
    - 序列索引直接指向预加载的数据
    '''

    def __init__(self,
                 data_dir: str,
                 input_names: List[str],
                 label_names: List[str],
                 side: str,
                 sequence_length: int,
                 output_sequence_length: int,
                 model_delays: List[int],
                 participant_masses: Dict[str, float] = {},
                 device: torch.device = torch.device("cpu"),
                 mode: str = "train",
                 model_type: str = "Transformer",
                 start_token_value: float = 0.0,
                 file_suffix: Dict[str, str] = None,
                 remove_nan: bool = True,
                 action_patterns: Optional[List[str]] = None,
                 enable_action_filter: bool = False):
        """
        初始化序列数据集

        参数:
            data_dir: 数据根目录路径
            input_names: 输入特征列名列表
            label_names: 标签列名列表
            side: 身体侧别 ('l' 或 'r')
            sequence_length: 输入序列长度（窗口大小）
            output_sequence_length: 输出序列长度（预测/生成的时间步数）
            model_delays: 每个输出的预测延迟
            participant_masses: 参与者体重字典
            device: 计算设备
            mode: 数据集模式，'train' 或 'test'
            model_type: 模型类型 'Transformer' 或 'GenerativeTransformer'
            start_token_value: 生成式模型的起始token值
            file_suffix: 文件后缀映射字典
            remove_nan: 是否自动检测并移除包含NaN的行
            action_patterns: 运动类型筛选的正则表达式列表
            enable_action_filter: 是否启用action_patterns筛选
        """
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.sequence_length = sequence_length
        self.output_sequence_length = output_sequence_length
        self.model_delays = model_delays
        self.participant_masses = participant_masses
        self.device = device
        self.mode = mode.lower()
        self.model_type = model_type
        self.start_token_value = start_token_value
        self.remove_nan = remove_nan
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
            'trials_with_all_nan_labels': 0  # 标签全为NaN的试验数
        }

        filter_status = "启用" if self.enable_action_filter else "禁用"
        print(f"开始加载 {self.mode} 数据集 ({model_type})...")
        print(f"找到 {len(self.trial_names)} 个试验 (动作筛选: {filter_status})")

        # 预加载所有数据到内存
        self.all_input_data = []  # 存储所有试验的输入数据
        self.all_label_data = []  # 存储所有试验的标签数据
        self._preload_all_data()

        # === 检测并移除标签全为NaN的序列 ===
        self._remove_invalid_label_sequences()

        # 生成序列索引
        print(f"生成序列索引...")
        self.sequences = self._generate_sequences()

        # 记录每个trial生成的子序列数量（用于高效重组）
        self.trial_sequence_counts = self._compute_trial_sequence_counts()

        print(f"数据集初始化完成 - 模式: {self.mode}, "
              f"试验数量: {len(self.trial_names)}, "
              f"序列数量: {len(self.sequences)}")

        if self.remove_nan and self.nan_removal_stats['trials_with_all_nan_labels'] > 0:
            self.print_nan_removal_summary()

    def __len__(self):
        '''返回数据集中序列的总数'''
        return len(self.sequences)

    def __getitem__(self, idx: int):
        '''
        获取单个序列样本（优化版本：直接从预加载的数据中索引）

        对于Transformer预测模型:
            返回: (input_data, label_data)
            - input_data: [num_input_features, sequence_length]
            - label_data: [num_label_features, output_sequence_length]
                         注意：每个输出通道根据model_delays[i]从不同位置开始

        对于GenerativeTransformer:
            返回: (input_data, shifted_label_data, label_data)
            - input_data: [num_input_features, sequence_length]
            - shifted_label_data: [num_label_features, output_sequence_length]
            - label_data: [num_label_features, output_sequence_length]
                         注意：每个输出通道根据model_delays[i]从不同位置开始
        '''
        # 获取序列索引信息
        trial_idx, input_start_idx = self.sequences[idx]

        # 提取输入序列
        input_seq = self.all_input_data[trial_idx][:, input_start_idx:input_start_idx + self.sequence_length]

        # 提取标签序列 - 每个输出通道根据其delay分别提取
        num_outputs = len(self.model_delays)
        label_data = self.all_label_data[trial_idx]

        # 为每个输出通道创建标签序列
        label_seqs = []
        input_end_idx = input_start_idx + self.sequence_length

        for i in range(num_outputs):
            # 每个输出通道从 input_end + model_delays[i] 开始
            label_start = input_end_idx + self.model_delays[i]
            label_end = label_start + self.output_sequence_length
            channel_label = label_data[i:i + 1, label_start:label_end]  # [1, output_sequence_length]
            label_seqs.append(channel_label)

        # 拼接所有输出通道
        label_seq = np.concatenate(label_seqs, axis=0)  # [num_outputs, output_sequence_length]

        # 转换为tensor
        input_seq = torch.from_numpy(input_seq).float()
        label_seq = torch.from_numpy(label_seq).float()

        if self.model_type == 'GenerativeTransformer':
            # 生成式模型需要shifted的标签作为解码器输入
            shifted_label_seq = self._create_shifted_target(label_seq)
            return input_seq, shifted_label_seq, label_seq
        else:
            # 预测模型直接返回
            return input_seq, label_seq

    def _create_shifted_target(self, label_seq: torch.Tensor) -> torch.Tensor:
        """
        创建右移的目标序列（用于解码器输入）

        参数:
            label_seq: [num_outputs, output_sequence_length]

        返回:
            shifted_seq: [num_outputs, output_sequence_length]
        """
        num_outputs, seq_len = label_seq.shape

        # 创建起始token
        start_token = torch.full(
            (num_outputs, 1),
            self.start_token_value,
            dtype=label_seq.dtype,
            device=label_seq.device
        )

        # 右移：[start_token, label[:-1]]
        shifted_seq = torch.cat([start_token, label_seq[:, :-1]], dim=1)

        return shifted_seq

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

    def _get_trial_names(self) -> List[str]:
        '''
        扫描数据目录，获取所有试验名称
        支持基于action_patterns的筛选
        '''
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
        valid_columns = [col for col in columns if col in df.columns]

        if not valid_columns:
            return 0, len(df)

        subset_df = df[valid_columns]
        nan_mask = subset_df.isna().any(axis=1)
        valid_mask = ~nan_mask
        valid_indices = valid_mask[valid_mask].index.tolist()

        if not valid_indices:
            return 0, 0

        start_index = valid_indices[0]
        end_index = valid_indices[-1] + 1

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

            # 转换为numpy数组并存储
            self.all_input_data.append(input_data.numpy())
            self.all_label_data.append(label_data.numpy())

        print("所有数据预加载完成!")

    def _remove_invalid_label_sequences(self):
        """
        检测并移除标签全为NaN的序列
        同时更新trial_names、all_input_data和all_label_data
        """
        if not self.remove_nan:
            return

        print("检测标签全为NaN的序列...")

        valid_indices = []
        removed_trials = []

        for idx in range(len(self.trial_names)):
            label_data = self.all_label_data[idx]

            # 检查标签数据是否全为NaN
            is_all_nan = np.all(np.isnan(label_data))

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

            print(f"  移除完成，剩余 {len(self.trial_names)} 个有效试验")
        else:
            print("  未发现标签全为NaN的序列")

    def _load_trial_data(self, trial_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        加载单个试验的完整数据

        返回:
            input_data: [num_input_features, sequence_length]
            label_data: [num_label_features, sequence_length]
        '''
        trial_dir = os.path.join(self.data_dir, self.mode, trial_name)

        if not os.path.exists(trial_dir):
            raise FileNotFoundError(f"试验目录不存在: {trial_dir}")

        # 从试验名称中提取参与者姓名
        participant = trial_name.split("/")[0].split("\\")[0]
        trial_basename = os.path.basename(trial_name)
        file_prefix = f"{participant}_{trial_basename}"

        # 构建文件路径
        input_filename = f"{file_prefix}{self.file_suffix['input']}"
        label_filename = f"{file_prefix}{self.file_suffix['label']}"
        input_file_path = os.path.join(trial_dir, input_filename)
        label_file_path = os.path.join(trial_dir, label_filename)

        # 检查文件是否存在
        if not os.path.exists(input_file_path):
            raise FileNotFoundError(f"输入文件不存在: {input_file_path}")
        if not os.path.exists(label_file_path):
            raise FileNotFoundError(f"标签文件不存在: {label_file_path}")

        # 加载数据
        body_mass = self.participant_masses.get(participant, 1.0)
        input_data, valid_range = self._load_input_data(input_file_path, body_mass)
        label_data = self._load_label_data(label_file_path, valid_range)

        # 注意：不在这里应用延迟，延迟会在生成序列索引时处理
        return input_data, label_data

    def _load_input_data(self, file_path: str, body_mass: float) -> Tuple[torch.Tensor, Optional[Tuple[int, int]]]:
        '''加载并预处理输入数据'''
        df = pd.read_csv(file_path)
        original_length = len(df)

        # 如果启用NaN移除，检测并截断
        valid_range = None
        if self.remove_nan:
            start_idx, end_idx = self._find_valid_range(df, self.input_names)
            if end_idx == 0:
                # 所有行都是NaN
                print(f"  警告: {file_path} 所有行都包含NaN")
                return torch.zeros((len(self.input_names), 0), dtype=torch.float32), (0, 0)

            if start_idx > 0 or end_idx < original_length:
                df = df.iloc[start_idx:end_idx].copy()
                valid_range = (start_idx, end_idx)

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
        '''加载标签数据'''
        df = pd.read_csv(file_path)

        # 如果指定了有效范围，截断标签数据以匹配输入数据
        if valid_range is not None:
            start_idx, end_idx = valid_range
            if end_idx == 0:
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

    def _generate_sequences(self) -> List[Tuple[int, int]]:
        '''
        生成所有可用的序列索引（关键优化）
        返回: [(trial_idx, input_start_idx), ...]

        注意：label的起始位置不在这里记录，而是在__getitem__中根据每个通道的delay计算

        对于所有模型类型：
        - input_seq: 从 input_start_idx 开始，长度为 sequence_length
        - label_seq的每个通道: 从 input_start_idx + sequence_length + model_delays[i] 开始，
                               长度为 output_sequence_length
        '''
        sequences = []

        # 计算最大延迟（用于确定所需的数据长度）
        max_delay = max(self.model_delays)

        for trial_idx in range(len(self.trial_names)):
            # 获取该试验的数据长度
            input_len = self.all_input_data[trial_idx].shape[1]
            label_len = self.all_label_data[trial_idx].shape[1]

            # 确保输入和标签长度一致
            data_len = min(input_len, label_len)

            # 所有模型都需要：sequence_length + max_delay + output_sequence_length
            required_length = self.sequence_length + max_delay + self.output_sequence_length

            if data_len < required_length:
                print(f"警告: 试验 {self.trial_names[trial_idx]} 数据长度不足 "
                      f"(需要{required_length}, 实际{data_len})，跳过")
                continue

            # 生成所有有效的起始索引
            max_start_idx = data_len - required_length

            for input_start_idx in range(max_start_idx + 1):
                sequences.append((trial_idx, input_start_idx))

        return sequences

    def _compute_trial_sequence_counts(self):
        """
        计算每个trial生成的子序列数量
        返回列表，索引对应trial_idx，值为该trial的子序列数量
        """
        counts = [0] * len(self.trial_names)
        for trial_idx, _ in self.sequences:
            counts[trial_idx] += 1
        return counts

    def print_nan_removal_summary(self):
        """打印NaN移除统计摘要"""
        stats = self.nan_removal_stats
        print(f"\n{'=' * 60}")
        print(f"NaN移除统计摘要 - {self.mode.upper()} 数据集")
        print(f"{'=' * 60}")
        print(f"标签全为NaN的试验数: {stats['trials_with_all_nan_labels']}")
        print(f"{'=' * 60}\n")
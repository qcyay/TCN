import os
from typing import List, Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np


class SequenceDataset(Dataset):
    '''
    通用序列数据集，支持预测模型和生成式模型
    '''

    def __init__(self,
                 data_dir: str,
                 input_names: List[str],
                 label_names: List[str],
                 side: str,
                 sequence_length: int,
                 model_delays: List[int],
                 participant_masses: Dict[str, float] = {},
                 device: torch.device = torch.device("cpu"),
                 mode: str = "train",
                 model_type: str = "Transformer",
                 start_token_value: float = 0.0,
                 file_suffix: Dict[str, str] = None,
                 remove_nan: bool = True):
        """
        初始化序列数据集

        参数:
            data_dir: 数据根目录路径
            input_names: 输入特征列名列表
            label_names: 标签列名列表
            side: 身体侧别 ('l' 或 'r')
            sequence_length: 序列长度（窗口大小）
            model_delays: 每个输出的预测延迟
            participant_masses: 参与者体重字典
            device: 计算设备
            mode: 数据集模式，'train' 或 'test'
            model_type: 模型类型 'Transformer' 或 'GenerativeTransformer'
            start_token_value: 生成式模型的起始token值
            file_suffix: 文件后缀映射字典
            remove_nan: 是否自动检测并移除包含NaN的行
        """
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.sequence_length = sequence_length
        self.model_delays = model_delays
        self.participant_masses = participant_masses
        self.device = device
        self.mode = mode.lower()
        self.model_type = model_type
        self.start_token_value = start_token_value
        self.remove_nan = remove_nan

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

        # 加载所有数据并生成序列索引
        print(f"加载 {self.mode} 数据集 ({model_type})...")
        self.sequences = self._generate_sequences()

        print(f"数据集初始化完成 - 模式: {self.mode}, "
              f"试验数量: {len(self.trial_names)}, "
              f"序列数量: {len(self.sequences)}")

    def __len__(self):
        '''返回数据集中序列的总数'''
        return len(self.sequences)

    def __getitem__(self, idx: int):
        '''
        获取单个序列样本

        对于Transformer预测模型:
            返回: (input_data, label_data)
            - input_data: [num_input_features, sequence_length]
            - label_data: [num_label_features, sequence_length]

        对于GenerativeTransformer:
            返回: (input_data, shifted_label_data, label_data)
            - input_data: [num_input_features, sequence_length]
            - shifted_label_data: [num_label_features, sequence_length] (解码器输入)
            - label_data: [num_label_features, sequence_length] (目标输出)
        '''
        trial_idx, start_idx = self.sequences[idx]
        trial_name = self.trial_names[trial_idx]

        # 加载试验数据
        input_data, label_data = self._load_trial_data(trial_name)

        # 提取序列片段
        end_idx = start_idx + self.sequence_length
        input_seq = input_data[:, start_idx:end_idx]
        label_seq = label_data[:, start_idx:end_idx]
        # breakpoint()

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
            label_seq: [num_outputs, sequence_length]

        返回:
            shifted_seq: [num_outputs, sequence_length]
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

    def _get_trial_names(self) -> List[str]:
        '''扫描数据目录，获取所有试验名称'''
        mode_dir = os.path.join(self.data_dir, self.mode)

        if not os.path.exists(mode_dir):
            raise FileNotFoundError(f"模式目录不存在: {mode_dir}")

        # 获取参与者目录
        participants = []
        for item in os.listdir(mode_dir):
            item_path = os.path.join(mode_dir, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                participants.append(item)

        if not participants:
            raise ValueError(f"在目录 {mode_dir} 中未找到参与者数据")

        # 遍历参与者目录，收集试验名称
        trial_names = []
        for participant in participants:
            participant_dir = os.path.join(mode_dir, participant)

            for trial_item in os.listdir(participant_dir):
                trial_path = os.path.join(participant_dir, trial_item)
                if os.path.isdir(trial_path) and not trial_item.startswith('.'):
                    trial_names.append(os.path.join(participant, trial_item))

        if not trial_names:
            raise ValueError(f"在参与者目录中未找到试验数据")

        return trial_names

    def _find_nan_cutoff(self, df: pd.DataFrame, columns: List[str]) -> int:
        """在指定列中查找第一个包含NaN的行索引"""
        valid_columns = [col for col in columns if col in df.columns]

        if not valid_columns:
            return len(df)

        subset_df = df[valid_columns]
        nan_mask = subset_df.isna().any(axis=1)
        nan_indices = nan_mask[nan_mask].index.tolist()

        if nan_indices:
            return nan_indices[0]
        else:
            return len(df)

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
        input_data, cutoff_length = self._load_input_data(input_file_path, body_mass)
        label_data = self._load_label_data(label_file_path, cutoff_length)

        # 对于预测模型，应用延迟；对于生成式模型，不在这里应用延迟
        if self.model_type != 'GenerativeTransformer':
            label_data = self._apply_delays(label_data)

        return input_data, label_data

    def _load_input_data(self, file_path: str, body_mass: float) -> Tuple[torch.Tensor, Optional[int]]:
        '''加载并预处理输入数据'''
        df = pd.read_csv(file_path)
        original_length = len(df)

        # 如果启用NaN移除，检测并截断
        cutoff_length = None
        if self.remove_nan:
            cutoff_index = self._find_nan_cutoff(df, self.input_names)
            if cutoff_index < original_length:
                df = df.iloc[:cutoff_index].copy()
                cutoff_length = cutoff_index

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

        return input_data, cutoff_length

    def _load_label_data(self, file_path: str, cutoff_length: Optional[int] = None) -> torch.Tensor:
        '''加载标签数据'''
        df = pd.read_csv(file_path)
        original_length = len(df)

        # 如果指定了截断长度，截断标签数据以匹配输入数据
        if cutoff_length is not None and cutoff_length < original_length:
            df = df.iloc[:cutoff_length].copy()

        # 检查是否有缺失的必需列
        missing_cols = [col for col in self.label_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"标签数据缺失必需的列: {missing_cols}")

        # 提取数据并转换为tensor
        extracted_data = df[self.label_names].values
        label_data = torch.tensor(extracted_data, dtype=torch.float32).transpose(0, 1)

        return label_data

    def _apply_delays(self, label_data: torch.Tensor) -> torch.Tensor:
        '''
        应用预测延迟到标签数据（仅用于预测模型）

        参数:
            label_data: [num_outputs, seq_len]

        返回:
            delayed_label_data: [num_outputs, seq_len]
        '''
        num_outputs, seq_len = label_data.shape
        delayed_label_data = label_data.clone()

        for i, delay in enumerate(self.model_delays):
            if delay > 0:
                # 将标签向前移动delay个位置
                delayed_label_data[i, :-delay] = label_data[i, delay:]
                # 末尾用NaN填充
                delayed_label_data[i, -delay:] = float('nan')

        return delayed_label_data

    def _generate_sequences(self) -> List[Tuple[int, int]]:
        '''
        生成所有可用的序列索引

        返回:
            sequences: [(trial_idx, start_idx), ...]
        '''
        sequences = []

        for trial_idx, trial_name in enumerate(self.trial_names):
            # 加载试验数据
            input_data, label_data = self._load_trial_data(trial_name)
            seq_len = input_data.shape[1]

            # 计算最大延迟
            max_delay = max(self.model_delays) if self.model_type != 'GenerativeTransformer' else 0

            # 生成所有有效的起始索引
            max_start_idx = seq_len - self.sequence_length - max_delay

            if max_start_idx < 0:
                print(f"警告: 试验 {trial_name} 数据长度不足，跳过")
                continue

            for start_idx in range(max_start_idx + 1):
                sequences.append((trial_idx, start_idx))
            # breakpoint()

        return sequences


# 使用示例
if __name__ == "__main__":
    # 示例配置
    config = {
        "data_dir": "data/example",
        "input_names": ["foot_imu_r_gyro_x", "foot_imu_r_gyro_y"],
        "label_names": ["hip_flexion_r_moment", "knee_angle_r_moment"],
        "side": "r",
        "sequence_length": 50,
        "model_delays": [10, 0],
        "participant_masses": {"BT23": 67.23, "BT24": 77.79},
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "mode": "train",
        "remove_nan": True
    }

    print("=" * 60)
    print("测试预测模型数据集")
    print("=" * 60)

    # 创建预测模型数据集
    pred_dataset = SequenceDataset(**config, model_type="Transformer")
    print(f"训练集大小: {len(pred_dataset)}")

    # 加载一个样本
    input_seq, label_seq = pred_dataset[0]
    print(f"输入序列形状: {input_seq.shape}")
    print(f"标签序列形状: {label_seq.shape}")

    print("\n" + "=" * 60)
    print("测试生成式模型数据集")
    print("=" * 60)

    # 创建生成式模型数据集
    gen_dataset = SequenceDataset(**config, model_type="GenerativeTransformer", start_token_value=0.0)
    print(f"训练集大小: {len(gen_dataset)}")

    # 加载一个样本
    input_seq, shifted_label, label_seq = gen_dataset[0]
    print(f"输入序列形状: {input_seq.shape}")
    print(f"Shifted标签形状: {shifted_label.shape}")
    print(f"目标标签形状: {label_seq.shape}")
    print(f"Shifted标签第一个值: {shifted_label[:, 0]}")
    print(f"原始标签第一个值: {label_seq[:, 0]}")
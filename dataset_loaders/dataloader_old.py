import os
from typing import List, Dict, Optional
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
                 load_to_device: bool = False):  # 新增：是否在__getitem__中加载到device
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
        """
        self.data_dir = data_dir
        self.input_names = input_names
        self.label_names = label_names
        self.side = side
        self.participant_masses = participant_masses
        self.device = device
        self.mode = mode.lower()
        self.remove_nan = remove_nan
        self.load_to_device = load_to_device  # 新增

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
        print(f"数据集初始化完成 - 模式: {self.mode}, 试验数量: {len(self.trial_names)}, 加载模式: {load_mode}")
        if self.remove_nan:
            print(f"NaN自动移除: 启用")

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

    def _get_trial_names(self):
        '''
        扫描数据目录，获取所有试验名称。
        新目录结构: data_dir/(train/test)/人名/运动状态/
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
        for participant in participants:
            participant_dir = os.path.join(mode_dir, participant)

            # 获取参与者的所有试验
            for trial_item in os.listdir(participant_dir):
                trial_path = os.path.join(participant_dir, trial_item)
                if os.path.isdir(trial_path) and not trial_item.startswith('.'):
                    # 试验名称格式: 参与者/试验项目
                    trial_names.append(os.path.join(participant, trial_item))

        if not trial_names:
            raise ValueError(f"在参与者目录中未找到试验数据")

        print(f"找到 {len(trial_names)} 个试验")
        return trial_names

    def _find_nan_cutoff(self, df: pd.DataFrame, columns: List[str]) -> int:
        """
        在指定列中查找第一个包含NaN的行索引

        参数:
            df: DataFrame数据
            columns: 要检查的列名列表

        返回:
            cutoff_index: 截断索引，如果没有NaN则返回数据长度
        """
        # 检查指定列是否存在
        valid_columns = [col for col in columns if col in df.columns]

        if not valid_columns:
            print(f"    警告: 指定的列都不存在于数据中")
            return len(df)

        # 只检查有效列
        subset_df = df[valid_columns]

        # 检查每一行是否包含NaN
        nan_mask = subset_df.isna().any(axis=1)

        # 找到第一个包含NaN的行
        nan_indices = nan_mask[nan_mask].index.tolist()

        if nan_indices:
            cutoff_index = nan_indices[0]
            num_nan_rows = len(df) - cutoff_index
            # print(f"    检测到NaN: 从第 {cutoff_index} 行开始, 将移除 {num_nan_rows} 行")
            return cutoff_index
        else:
            return len(df)

    def _load_trial_data(self, trial_name: str):
        '''
        从单个试验目录加载数据，适配新的文件命名格式
        支持自动检测和移除包含NaN的行

        参数:
            trial_name: 试验名称，格式为 "参与者/试验项目"
        '''
        # print(f"加载试验数据: {trial_name}")

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
        input_data, input_cutoff = self._load_input_data(
            input_file_path,
            body_mass=self.participant_masses.get(participant, 1.0)
        )

        # 加载标签数据（使用相同的截断长度）
        label_data = self._load_label_data(label_file_path, cutoff_length=input_cutoff)

        # 更新统计信息
        self.nan_removal_stats['trials_processed'] += 1
        if input_cutoff is not None:
            self.nan_removal_stats['trials_with_nan'] += 1

        return input_data, label_data

    def _load_input_data(self, file_path: str, body_mass: float):
        '''
        加载并预处理输入数据CSV文件
        支持自动检测和移除包含NaN的行

        返回:
            input_data: 处理后的输入数据张量
            cutoff_length: 截断后的数据长度（如果有截断），或None
        '''
        # 读取CSV文件
        df = pd.read_csv(file_path)
        original_length = len(df)
        # print(f"  输入数据: {file_path}, 原始数据形状: {df.shape}")

        # 如果启用NaN移除，检测并截断
        cutoff_length = None
        if self.remove_nan:
            cutoff_index = self._find_nan_cutoff(df, self.input_names)

            if cutoff_index < original_length:
                # 需要截断
                df = df.iloc[:cutoff_index].copy()
                cutoff_length = cutoff_index
                removed_rows = original_length - cutoff_index
                self.nan_removal_stats['total_rows_removed'] += removed_rows
                # print(f"  截断后数据形状: {df.shape}")

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

        return input_data, cutoff_length

    def _load_label_data(self, file_path: str, cutoff_length: Optional[int] = None):
        '''
        加载标签数据CSV文件

        参数:
            file_path: 标签文件路径
            cutoff_length: 如果不为None，截断到指定长度以匹配输入数据
        '''
        df = pd.read_csv(file_path)
        original_length = len(df)

        # 如果指定了截断长度，截断标签数据以匹配输入数据
        if cutoff_length is not None and cutoff_length < original_length:
            df = df.iloc[:cutoff_length].copy()

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
        max_sequence_length = max(trial_sequence_lengths)

        # 对长度不足的试验进行填充
        for i in range(len(data)):
            trial_sequence_length = trial_sequence_lengths[i]
            if trial_sequence_length < max_sequence_length:
                padding_length = max_sequence_length - trial_sequence_length
                for j in range(len(data[i])):
                    padding = torch.zeros(
                        (1, data[i][j].shape[1], padding_length),
                        device=data[i][j].device  # 使用数据本身的device
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


# 使用示例
if __name__ == "__main__":
    # 示例配置
    config = {
        "data_dir": "data",
        "input_names": ["foot_imu_r_gyro_x", "foot_imu_r_gyro_y"],
        "label_names": ["hip_flexion_r_moment", "knee_angle_r_moment"],
        "side": "r",
        "participant_masses": {"BT23": 67.23, "BT24": 77.79},
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "mode": "train",
        "remove_nan": True  # 启用NaN自动移除
    }

    # 创建训练数据集
    train_dataset = TcnDataset(**config)
    print(f"训练集大小: {len(train_dataset)}")

    # 加载数据（这会触发NaN检测和移除）
    sample_input, sample_label, lengths = train_dataset[1]
    breakpoint()
    print(f"输入数据形状: {sample_input.shape}")
    print(f"标签数据形状: {sample_label.shape}")
    print(f"序列长度: {lengths}")

    # 打印NaN移除统计
    train_dataset.print_nan_removal_summary()

    # 创建测试数据集
    test_config = config.copy()
    test_config["mode"] = "test"
    test_dataset = TcnDataset(**test_config)
    print(f"测试集大小: {len(test_dataset)}")

    # 打印测试集的NaN移除统计
    test_dataset.print_nan_removal_summary()
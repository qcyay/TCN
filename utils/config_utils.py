import importlib
import torch
from typing import List

def load_config(config_path: str):
	'''Load config file as module.'''
	config_path = config_path.replace("/", ".").replace("\\", ".")
	if config_path.endswith(".py"):
		config_path = config_path[:-3]
	print(f"Loading config file from {config_path}.")
	return importlib.import_module(config_path)

"""
配置处理工具函数
提供特征选择等配置后处理功能
"""

def apply_feature_selection(config):
	"""
    根据selected_feature_indices更新配置参数

    如果启用了特征选择，此函数会：
    1. 选择对应索引的input_names
    2. 选择对应索引的center和scale参数
    3. 更新input_size

    参数:
        config: 配置对象

    返回:
        config: 更新后的配置对象
    """
	# 检查是否启用特征选择
	if not hasattr(config, 'enable_feature_selection'):
		config.enable_feature_selection = False

	if not config.enable_feature_selection:
		print("特征选择未启用，使用所有特征")
		return config

	# 检查是否有选择的特征索引
	if not hasattr(config, 'selected_feature_indices') or \
			config.selected_feature_indices is None or \
			len(config.selected_feature_indices) == 0:
		print("未指定特征索引，使用所有特征")
		config.enable_feature_selection = False
		return config

	# 获取选择的索引
	selected_indices = config.selected_feature_indices

	# 验证索引的有效性
	max_features = len(config.input_names)
	for idx in selected_indices:
		if idx < 0 or idx >= max_features:
			raise ValueError(
				f"特征索引 {idx} 超出范围 [0, {max_features - 1}]"
			)

	# 保存原始配置（用于记录）
	original_input_names = config.input_names.copy()
	original_input_size = len(original_input_names)

	# 应用特征选择
	print(f"\n{'=' * 60}")
	print("应用特征选择...")
	print(f"{'=' * 60}")
	print(f"原始特征数量: {original_input_size}")
	print(f"选择的特征索引: {selected_indices}")

	# 1. 选择对应的input_names
	config.input_names = [config.input_names[i] for i in selected_indices]

	# 2. 选择对应的center和scale
	# center和scale的形状是 [num_features, 1]
	if isinstance(config.center, torch.Tensor):
		config.center = config.center[selected_indices, :]
	else:
		# 如果是numpy或list，先转换为tensor
		config.center = torch.tensor(config.center)[selected_indices, :]

	if isinstance(config.scale, torch.Tensor):
		config.scale = config.scale[selected_indices, :]
	else:
		config.scale = torch.tensor(config.scale)[selected_indices, :]

	# 3. 更新input_size
	config.input_size = len(config.input_names)

	# 打印选择结果
	print(f"选择后特征数量: {config.input_size}")
	print(f"\n选择的特征列表:")
	for i, (idx, name) in enumerate(zip(selected_indices, config.input_names)):
		print(f"  {i:2d}. [原索引 {idx:2d}] {name}")

	print(f"\n归一化参数已相应更新:")
	print(f"  center shape: {config.center.shape}")
	print(f"  scale shape: {config.scale.shape}")
	print(f"{'=' * 60}\n")

	return config


def get_feature_group_indices(group_name: str) -> List[int]:
	"""
    获取预定义特征组的索引

    特征索引对照表（共25个特征）:
      0-2:   foot_imu  gyro (x, y, z)
      3-5:   foot_imu  accel (x, y, z)
      6-8:   shank_imu gyro (x, y, z)
      9-11:  shank_imu accel (x, y, z)
      12-14: thigh_imu gyro (x, y, z)
      15-17: thigh_imu accel (x, y, z)
      18-20: insole (cop_x, cop_z, force_y)
      21:    hip_angle
      22:    hip_angle_velocity_filt
      23:    knee_angle
      24:    knee_angle_velocity_filt

    参数:
        group_name: 特征组名称

    支持的组名:
        - 'all': 所有特征 (0-24)
        - 'imu': 所有IMU传感器 (0-17)
        - 'foot_imu': 脚部IMU (0-5)
        - 'shank_imu': 小腿IMU (6-11)
        - 'thigh_imu': 大腿IMU (12-17)
        - 'insole': 压力鞋垫 (18-20)
        - 'joint_angles': 关节角度和角速度 (21-24)
        - 'imu_joint': IMU + 关节角度（不含鞋垫）(0-17, 21-24)

    返回:
        特征索引列表
    """
	feature_groups = {
		'all': list(range(25)),
		'imu': list(range(18)),
		'foot_imu': list(range(0, 6)),
		'shank_imu': list(range(6, 12)),
		'thigh_imu': list(range(12, 18)),
		'insole': list(range(18, 21)),
		'joint_angles': list(range(21, 25)),
		'imu_joint': list(range(18)) + list(range(21, 25)),
	}

	if group_name not in feature_groups:
		available_groups = ', '.join(feature_groups.keys())
		raise ValueError(
			f"未知的特征组名称: '{group_name}'\n"
			f"可用的组名: {available_groups}"
		)

	return feature_groups[group_name]


def print_feature_indices():
	"""
    打印所有特征的索引对照表
    """
	feature_names = [
		"foot_imu_*_gyro_x", "foot_imu_*_gyro_y", "foot_imu_*_gyro_z",
		"foot_imu_*_accel_x", "foot_imu_*_accel_y", "foot_imu_*_accel_z",
		"shank_imu_*_gyro_x", "shank_imu_*_gyro_y", "shank_imu_*_gyro_z",
		"shank_imu_*_accel_x", "shank_imu_*_accel_y", "shank_imu_*_accel_z",
		"thigh_imu_*_gyro_x", "thigh_imu_*_gyro_y", "thigh_imu_*_gyro_z",
		"thigh_imu_*_accel_x", "thigh_imu_*_accel_y", "thigh_imu_*_accel_z",
		"insole_*_cop_x", "insole_*_cop_z", "insole_*_force_y",
		"hip_angle_*", "hip_angle_*_velocity_filt",
		"knee_angle_*", "knee_angle_*_velocity_filt"
	]

	print("\n" + "=" * 70)
	print("特征索引对照表 (共25个特征)")
	print("=" * 70)

	categories = [
		("脚部IMU", 0, 6),
		("小腿IMU", 6, 12),
		("大腿IMU", 12, 18),
		("压力鞋垫", 18, 21),
		("关节角度", 21, 25)
	]

	for category_name, start_idx, end_idx in categories:
		print(f"\n{category_name}:")
		for i in range(start_idx, end_idx):
			print(f"  [{i:2d}] {feature_names[i]}")

	print("\n" + "=" * 70)

	# 打印预定义组
	print("\n预定义特征组:")
	groups = {
		'all': '所有特征',
		'imu': '所有IMU传感器',
		'foot_imu': '脚部IMU',
		'shank_imu': '小腿IMU',
		'thigh_imu': '大腿IMU',
		'insole': '压力鞋垫',
		'joint_angles': '关节角度和角速度',
		'imu_joint': 'IMU + 关节角度（不含鞋垫）'
	}

	for group_key, group_desc in groups.items():
		indices = get_feature_group_indices(group_key)
		print(f"  '{group_key}': {group_desc}")
		print(f"    索引: {indices}")

	print("=" * 70 + "\n")


# 示例用法
if __name__ == "__main__":
	# 打印特征索引对照表
	print_feature_indices()

	# 测试特征组获取
	print("\n测试特征组获取:")
	for group_name in ['imu', 'joint_angles', 'imu_joint']:
		indices = get_feature_group_indices(group_name)
		print(f"{group_name}: {indices}")
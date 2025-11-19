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
	print(f"  center: {config.center}")
	print(f"  scale: {config.scale}")
	print(f"{'=' * 60}\n")

	return config

# 示例用法
if __name__ == "__main__":
	pass
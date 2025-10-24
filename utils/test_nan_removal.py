"""
测试数据加载器的NaN自动移除功能

使用方法:
    python test_nan_removal.py --config_path configs.TCN.default_config
"""

import sys
import os

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import torch
from utils.config_utils import load_config
from dataset_loaders.dataloader import TcnDataset

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs.TCN.default_config")
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

config = load_config(args.config_path)


def test_dataset_loading(mode: str):
    """测试指定模式的数据集加载"""
    print(f"\n{'=' * 70}")
    print(f"测试 {mode.upper()} 数据集加载")
    print(f"{'=' * 70}\n")

    device = torch.device(args.device)

    # 准备输入和标签名称
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    # 创建数据集（启用NaN移除）
    dataset = TcnDataset(
        data_dir=config.data_dir,
        input_names=input_names,
        label_names=label_names,
        side=config.side,
        participant_masses=config.participant_masses,
        device=device,
        mode=mode,
        remove_nan=True  # 启用NaN自动移除
    )

    print(f"\n数据集大小: {len(dataset)} 个trials")

    # 测试加载所有数据
    print(f"\n{'=' * 70}")
    print(f"测试加载所有数据...")
    print(f"{'=' * 70}\n")

    try:
        input_data, label_data, trial_lengths = dataset[:]

        print(f"✓ 成功加载所有数据")
        print(f"  输入数据形状: {input_data.shape}")
        print(f"  标签数据形状: {label_data.shape}")
        print(f"  Trial数量: {len(trial_lengths)}")

        # 检查是否有NaN
        input_has_nan = torch.isnan(input_data).any().item()
        label_has_nan = torch.isnan(label_data).any().item()

        print(f"\n数据质量检查:")
        if input_has_nan:
            num_nan = torch.isnan(input_data).sum().item()
            print(f"  ❌ 输入数据包含 {num_nan} 个NaN")
        else:
            print(f"  ✓ 输入数据完全干净，无NaN")

        if label_has_nan:
            num_nan = torch.isnan(label_data).sum().item()
            print(f"  ℹ️  标签数据包含 {num_nan} 个NaN (这是正常的)")
        else:
            print(f"  ✓ 标签数据完全干净，无NaN")

        # 打印统计信息
        dataset.print_nan_removal_summary()

        return not input_has_nan  # 返回输入数据是否干净

    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_trials(mode: str, num_trials: int = 5):
    """测试单独加载几个trials"""
    print(f"\n{'=' * 70}")
    print(f"测试单独加载 {mode.upper()} 数据集的前 {num_trials} 个trials")
    print(f"{'=' * 70}\n")

    device = torch.device(args.device)

    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    dataset = TcnDataset(
        data_dir=config.data_dir,
        input_names=input_names,
        label_names=label_names,
        side=config.side,
        participant_masses=config.participant_masses,
        device=device,
        mode=mode,
        remove_nan=True
    )

    max_trials = min(num_trials, len(dataset))
    all_clean = True

    for i in range(max_trials):
        trial_name = dataset.get_trial_names()[i]
        print(f"\nTrial {i + 1}/{max_trials}: {trial_name}")
        print(f"{'-' * 70}")

        try:
            input_data, label_data, trial_length = dataset[i]

            # 检查NaN
            input_nan = torch.isnan(input_data).any().item()

            if input_nan:
                num_nan = torch.isnan(input_data).sum().item()
                print(f"  ❌ 包含 {num_nan} 个NaN")
                all_clean = False
            else:
                print(f"  ✓ 数据干净")
                print(f"  序列长度: {trial_length[0]}")
                print(f"  输入形状: {input_data.shape}")
                print(f"  标签形状: {label_data.shape}")

        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            all_clean = False

    return all_clean


def test_with_and_without_removal():
    """对比启用和禁用NaN移除的效果"""
    print(f"\n{'=' * 70}")
    print(f"对比测试：启用 vs 禁用 NaN移除")
    print(f"{'=' * 70}\n")

    device = torch.device(args.device)
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]

    # 测试1: 禁用NaN移除
    print("1️⃣  禁用NaN移除:")
    print("-" * 70)
    try:
        dataset_no_removal = TcnDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            participant_masses=config.participant_masses,
            device=device,
            mode='train',
            remove_nan=False  # 禁用
        )

        input_data, _, _ = dataset_no_removal[0]
        nan_count_disabled = torch.isnan(input_data).sum().item()
        print(f"第一个trial的NaN数量: {nan_count_disabled}")

    except Exception as e:
        print(f"加载失败: {e}")
        nan_count_disabled = "错误"

    # 测试2: 启用NaN移除
    print(f"\n2️⃣  启用NaN移除:")
    print("-" * 70)
    try:
        dataset_with_removal = TcnDataset(
            data_dir=config.data_dir,
            input_names=input_names,
            label_names=label_names,
            side=config.side,
            participant_masses=config.participant_masses,
            device=device,
            mode='train',
            remove_nan=True  # 启用
        )

        input_data, _, _ = dataset_with_removal[0]
        nan_count_enabled = torch.isnan(input_data).sum().item()
        print(f"第一个trial的NaN数量: {nan_count_enabled}")

        dataset_with_removal.print_nan_removal_summary()

    except Exception as e:
        print(f"加载失败: {e}")
        nan_count_enabled = "错误"

    # 对比结果
    print(f"\n{'=' * 70}")
    print(f"对比结果:")
    print(f"{'=' * 70}")
    print(f"禁用NaN移除的NaN数量: {nan_count_disabled}")
    print(f"启用NaN移除的NaN数量: {nan_count_enabled}")

    if nan_count_enabled == 0 and nan_count_disabled != 0:
        print(f"\n✓✓✓ NaN移除功能工作正常！")
    elif nan_count_enabled == 0 and nan_count_disabled == 0:
        print(f"\nℹ️  数据本身就是干净的，无NaN")
    else:
        print(f"\n⚠️  NaN移除可能存在问题，请检查")


def main():
    print("\n" + "=" * 70)
    print(" " * 20 + "NaN移除功能测试")
    print("=" * 70)
    print(f"配置文件: {args.config_path}")
    print(f"数据目录: {config.data_dir}")
    print(f"设备: {args.device}")

    # 测试1: 对比启用/禁用NaN移除
    test_with_and_without_removal()

    # 测试2: 测试训练集
    train_clean = test_dataset_loading('train')

    # 测试3: 测试测试集
    test_clean = test_dataset_loading('test')

    # 测试4: 单独测试几个trials
    test_individual_trials('train', num_trials=3)

    # 最终总结
    print(f"\n{'=' * 70}")
    print("测试总结")
    print(f"{'=' * 70}")

    if train_clean and test_clean:
        print("✓✓✓ 所有测试通过！数据集已成功清理NaN")
        print("\n下一步:")
        print("  1. 运行: python compute_normalization_params.py")
        print("  2. 更新配置文件中的center和scale")
        print("  3. 运行: python diagnose_nan.py")
        print("  1. 开始训练: python train.py")
    else:
        print("⚠️  部分测试未通过，请检查数据或配置")
        print("\n建议:")
        print("  1. 运行: python utils/check_data_nan.py --detailed")
        print("  2. 检查数据文件是否正确")
        print("  3. 确认配置文件中的input_names是否正确")


if __name__ == "__main__":
    main()
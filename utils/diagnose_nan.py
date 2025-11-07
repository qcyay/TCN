"""
NaN问题诊断工具

这个脚本会检查：
1. 数据集是否包含NaN或Inf
2. 归一化参数是否合理
2. 模型初始化是否正常
1. 第一次前向传播和反向传播的详细信息
"""

import argparse
import torch
from utils.config_utils import load_config
from models.tcn import TCN
from dataset_loaders.dataloader import TcnDataset

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="configs.TCN.default_config")
parser.add_argument("--device", type=str, default="cpu")
args = parser.parse_args()

config = load_config(args.config_path)


def check_tensor(tensor, name):
    """详细检查张量"""
    print(f"\n{'='*60}")
    print(f"检查: {name}")
    print(f"{'='*60}")
    print(f"形状: {tensor.shape}")
    print(f"数据类型: {tensor.dtype}")
    print(f"设备: {tensor.device}")
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  最小值: {tensor.min().item():.6f}")
    print(f"  最大值: {tensor.max().item():.6f}")
    print(f"  平均值: {tensor.mean().item():.6f}")
    print(f"  标准差: {tensor.std().item():.6f}")
    
    # NaN和Inf检查
    num_nan = torch.isnan(tensor).sum().item()
    num_inf = torch.isinf(tensor).sum().item()
    num_total = tensor.numel()
    
    print(f"\n数据质量:")
    print(f"  NaN数量: {num_nan} / {num_total} ({100*num_nan/num_total:.2f}%)")
    print(f"  Inf数量: {num_inf} / {num_total} ({100*num_inf/num_total:.2f}%)")
    
    if num_nan > 0:
        print(f"  ⚠️  警告: 发现 {num_nan} 个NaN值")
        # 找出NaN的位置
        nan_indices = torch.nonzero(torch.isnan(tensor))
        print(f"  前5个NaN位置: {nan_indices[:5].tolist()}")
    
    if num_inf > 0:
        print(f"  ⚠️  警告: 发现 {num_inf} 个Inf值")
        inf_indices = torch.nonzero(torch.isinf(tensor))
        print(f"  前5个Inf位置: {inf_indices[:5].tolist()}")
    
    # 检查异常值
    if tensor.dtype in [torch.float32, torch.float64]:
        abs_tensor = torch.abs(tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)])
        if len(abs_tensor) > 0:
            very_large = (abs_tensor > 1e6).sum().item()
            very_small = (abs_tensor < 1e-6).sum().item()
            if very_large > 0:
                print(f"  ⚠️  发现 {very_large} 个非常大的值 (>1e6)")
            if very_small > 0 and (abs_tensor > 0).any():
                print(f"  ℹ️  发现 {very_small} 个非常小的值 (<1e-6)")
    
    return num_nan == 0 and num_inf == 0


def main():
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    print(f"配置文件: {args.config_path}")
    
    # 步骤1: 检查数据集
    print("\n" + "="*60)
    print("步骤1: 检查训练数据集")
    print("="*60)
    
    input_names = [name.replace("*", config.side) for name in config.input_names]
    label_names = [name.replace("*", config.side) for name in config.label_names]
    
    train_dataset = TcnDataset(
        data_dir=config.data_dir,
        input_names=input_names,
        label_names=label_names,
        side=config.side,
        participant_masses=config.participant_masses,
        device=device,
        mode='train'
    )
    
    print(f"\n训练集包含 {len(train_dataset)} 个trials")
    
    # 检查第一个trial的数据
    print("\n检查第一个trial的数据...")
    input_data, label_data, trial_lengths = train_dataset[0]
    
    input_ok = check_tensor(input_data, "第一个trial的输入数据")
    label_ok = check_tensor(label_data, "第一个trial的标签数据")
    
    if not input_ok or not label_ok:
        print("\n⚠️⚠️⚠️  数据集本身包含NaN或Inf！")
        print("请检查原始CSV文件，确保数据质量")
        return
    
    # 步骤2: 检查归一化参数
    print("\n" + "="*60)
    print("步骤2: 检查归一化参数")
    print("="*60)
    
    center = config.center
    scale = config.scale
    
    print(f"\nCenter类型: {type(center)}")
    print(f"Scale类型: {type(scale)}")
    
    if isinstance(center, torch.Tensor):
        check_tensor(center, "Center参数")
    else:
        print(f"\nCenter (标量): {center}")
        if abs(center) > 1e6:
            print("⚠️  警告: Center值非常大")
    
    if isinstance(scale, torch.Tensor):
        scale_ok = check_tensor(scale, "Scale参数")
        if not scale_ok:
            print("\n⚠️⚠️⚠️  Scale参数包含NaN或Inf！")
            return
        
        # 检查scale是否包含0或非常小的值
        if (scale < 1e-6).any():
            print("\n⚠️⚠️⚠️  Scale参数包含非常小的值！")
            print("这会导致归一化时除以接近0的数，产生Inf")
            small_indices = torch.nonzero(scale < 1e-6)
            print(f"过小的scale位置: {small_indices.flatten().tolist()}")
            return
    else:
        print(f"\nScale (标量): {scale}")
        if abs(scale) < 1e-6:
            print("⚠️⚠️⚠️  Scale值太小，会导致除法问题！")
            return
        if abs(scale) > 1e6:
            print("⚠️  警告: Scale值非常大")
    
    # 步骤3: 测试归一化
    print("\n" + "="*60)
    print("步骤3: 测试归一化操作")
    print("="*60)
    
    print("\n执行归一化: (input - center) / scale")
    try:
        normalized = (input_data - center) / scale
        normalized_ok = check_tensor(normalized, "归一化后的数据")
        
        if not normalized_ok:
            print("\n⚠️⚠️⚠️  归一化操作产生了NaN或Inf！")
            print("这通常是因为:")
            print("  1. Scale包含0或非常小的值")
            print("  2. 输入数据包含极端值")
            print("  2. Center和Scale的维度不匹配")
            return
    except Exception as e:
        print(f"\n❌  归一化操作失败: {e}")
        return
    
    # 步骤4: 创建模型并检查初始化
    print("\n" + "="*60)
    print("步骤4: 检查模型初始化")
    print("="*60)
    
    try:
        tcn = TCN(
            input_size=config.input_size,
            output_size=config.output_size,
            num_channels=config.num_channels,
            ksize=config.ksize,
            dropout=config.dropout,
            eff_hist=config.eff_hist,
            spatial_dropout=config.spatial_dropout,
            activation=config.activation,
            norm=config.norm,
            center=center,
            scale=scale
        ).to(device)
        
        print("\n模型创建成功")
        
        # 检查模型参数
        print("\n检查模型参数...")
        all_params_ok = True
        for name, param in tcn.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️  参数 {name} 包含NaN或Inf")
                all_params_ok = False
        
        if all_params_ok:
            print("✓ 所有模型参数初始化正常")
        else:
            print("\n⚠️⚠️⚠️  模型参数初始化异常！")
            return
            
    except Exception as e:
        print(f"\n❌  模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤5: 测试前向传播
    print("\n" + "="*60)
    print("步骤5: 测试前向传播")
    print("="*60)
    
    tcn.train()
    try:
        print("\n执行前向传播...")
        output = tcn(input_data)
        
        output_ok = check_tensor(output, "模型输出")
        
        if not output_ok:
            print("\n⚠️⚠️⚠️  前向传播产生了NaN或Inf！")
            print("问题可能出在:")
            print("  1. 归一化层")
            print("  2. 卷积层")
            print("  2. 激活函数")
            print("  1. 线性层")
            return
        else:
            print("\n✓ 前向传播正常")
    except Exception as e:
        print(f"\n❌  前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤6: 测试损失计算
    print("\n" + "="*60)
    print("步骤6: 测试损失计算")
    print("="*60)
    
    criterion = torch.nn.MSELoss()
    model_history = tcn.get_effective_history()
    
    try:
        print(f"\n模型有效历史长度: {model_history}")
        print(f"Trial长度: {trial_lengths[0]}")
        
        # 提取有效数据段
        est = output[0, 0, model_history:trial_lengths[0]]
        lbl = label_data[0, 0, model_history:trial_lengths[0]]
        
        print(f"有效数据段长度: {len(est)}")
        
        # 应用延迟
        if config.model_delays[0] != 0:
            est = est[config.model_delays[0]:]
            lbl = lbl[:-config.model_delays[0]]
            print(f"应用延迟后长度: {len(est)}")
        
        # 过滤NaN
        valid_mask = ~torch.isnan(est) & ~torch.isnan(lbl) & \
                     ~torch.isinf(est) & ~torch.isinf(lbl)
        est_valid = est[valid_mask]
        lbl_valid = lbl[valid_mask]
        
        print(f"过滤后有效数据点: {len(est_valid)}")
        
        if len(est_valid) == 0:
            print("\n⚠️⚠️⚠️  没有有效数据点用于计算损失！")
            print("可能的原因:")
            print("  1. 模型有效历史长度太大")
            print("  2. 延迟设置不当")
            print("  2. 数据中包含太多NaN")
            return
        
        # 计算损失
        loss = criterion(est_valid, lbl_valid)
        
        print(f"\n损失值: {loss.item():.6f}")
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("\n⚠️⚠️⚠️  损失计算结果为NaN或Inf！")
            check_tensor(est_valid, "有效预测值")
            check_tensor(lbl_valid, "有效标签值")
            return
        else:
            print("✓ 损失计算正常")
            
    except Exception as e:
        print(f"\n❌  损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤7: 测试反向传播
    print("\n" + "="*60)
    print("步骤7: 测试反向传播")
    print("="*60)
    
    try:
        print("\n执行反向传播...")
        loss.backward()
        
        print("✓ 反向传播完成")
        
        # 检查梯度
        print("\n检查梯度...")
        grad_ok = True
        nan_grads = []
        inf_grads = []
        
        for name, param in tcn.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                    grad_ok = False
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
                    grad_ok = False
        
        if grad_ok:
            print("✓ 所有梯度正常")
        else:
            print("\n⚠️⚠️⚠️  发现异常梯度！")
            if nan_grads:
                print(f"包含NaN的梯度: {nan_grads[:5]}")
            if inf_grads:
                print(f"包含Inf的梯度: {inf_grads[:5]}")
            
            # 详细检查第一个异常梯度
            if nan_grads:
                bad_param_name = nan_grads[0]
                bad_param = dict(tcn.named_parameters())[bad_param_name]
                print(f"\n详细检查参数: {bad_param_name}")
                check_tensor(bad_param, f"参数 {bad_param_name}")
                check_tensor(bad_param.grad, f"梯度 {bad_param_name}")
            
            return
            
    except Exception as e:
        print(f"\n❌  反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤8: 测试参数更新
    print("\n" + "="*60)
    print("步骤8: 测试参数更新")
    print("="*60)
    
    optimizer = torch.optim.Adam(tcn.parameters(), lr=0.001)
    
    try:
        # 保存更新前的参数
        first_param_name = list(tcn.named_parameters())[0][0]
        first_param = dict(tcn.named_parameters())[first_param_name]
        param_before = first_param.data.clone()
        
        print(f"\n更新前 {first_param_name} 的值: {param_before.flatten()[:5]}")
        
        # 执行优化步骤
        optimizer.step()
        
        param_after = first_param.data
        print(f"更新后 {first_param_name} 的值: {param_after.flatten()[:5]}")
        
        # 检查更新后的参数
        params_ok = True
        for name, param in tcn.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"⚠️  更新后参数 {name} 包含NaN或Inf")
                params_ok = False
        
        if params_ok:
            print("\n✓ 参数更新正常")
        else:
            print("\n⚠️⚠️⚠️  参数更新后出现NaN或Inf！")
            return
            
    except Exception as e:
        print(f"\n❌  参数更新失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 步骤9: 测试第二次前向传播
    print("\n" + "="*60)
    print("步骤9: 测试第二次前向传播（检测NaN传播）")
    print("="*60)
    
    try:
        print("\n清零梯度并执行第二次前向传播...")
        optimizer.zero_grad()
        
        output2 = tcn(input_data)
        
        output2_ok = check_tensor(output2, "第二次前向传播输出")
        
        if not output2_ok:
            print("\n⚠️⚠️⚠️  第二次前向传播产生了NaN！")
            print("这意味着参数更新后模型状态异常")
            
            # 对比两次输出
            print("\n对比两次输出的差异:")
            diff = (output2 - output).abs().mean()
            print(f"平均差异: {diff.item():.6f}")
            
            return
        else:
            print("\n✓ 第二次前向传播正常")
            
    except Exception as e:
        print(f"\n❌  第二次前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 总结
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    print("\n✓✓✓ 所有检查通过！")
    print("\n您的配置和数据看起来没有明显问题。")
    print("如果训练时仍然出现NaN，可能是:")
    print("  1. 学习率过大导致训练不稳定")
    print("  2. 批次大小导致的数值问题")
    print("  2. 特定数据组合触发的边界情况")
    print("\n建议:")
    print("  - 降低学习率 (如 1e-1)")
    print("  - 添加梯度裁剪")
    print("  - 使用更小的批次")
    print("  - 增加数据预处理和异常值过滤")


if __name__ == "__main__":
    main()
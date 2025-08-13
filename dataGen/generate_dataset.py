import numpy as np
import torch
from dataGen.traditional_solver import HelmholtzSolver, generate_random_source
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

def convert_to_torch_format(u_data, w_data):
    """
    将numpy数据转换为pytorch格式，适配你的Neural Operator
    
    参数:
    u_data: (num_samples, nx, ny)
    w_data: (num_samples, nx, ny)
    
    返回:
    u_torch: (num_samples, nx*ny, 1) - 展平的输入
    w_torch: (num_samples, nx*ny, 1) - 展平的输出
    x_coords: (nx*ny, 1) - x坐标
    y_coords: (nx*ny, 1) - y坐标
    """
    num_samples, nx, ny = u_data.shape
    
    # 展平空间维度
    u_torch = torch.tensor(u_data.reshape(num_samples, -1, 1), dtype=torch.float32)
    w_torch = torch.tensor(w_data.reshape(num_samples, -1, 1), dtype=torch.float32)
    
    # 生成坐标网格
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    x_coords = torch.tensor(X.reshape(-1, 1), dtype=torch.float32)
    y_coords = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32)
    
    return u_torch, w_torch, x_coords, y_coords


def generate_diverse_dataset(num_samples, nx=64, ny=64, delta=1.0, 
                           train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
                           save_dir="./neural_operator_data"):
    """
    生成多样化的训练数据集，支持4维数据格式
    
    参数:
    num_samples: 总样本数
    nx, ny: 网格分辨率
    delta: Helmholtz方程参数
    train_ratio, val_ratio, test_ratio: 训练/验证/测试集比例
    save_dir: 保存目录
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 使用traditional_solver中的generate_training_data函数
    print(f"开始生成 {num_samples} 个样本 (分辨率: {nx}×{ny})")
    print(f"使用4维数据格式: [u值, w值, x坐标, y坐标]")
    
    # 调用修改后的生成函数
    from dataGen.traditional_solver import generate_training_data
    
    # 生成数据 - 现在返回4维combined_data
    combined_data, u_data, w_data = generate_training_data(
        num_samples=num_samples,
        nx=nx, ny=ny, 
        delta=delta,
        save_path=None  # 我们稍后自己保存
    )
    
    print(f"数据生成完成! 形状:")
    print(f"  - combined_data: {combined_data.shape}")
    print(f"  - u_data: {u_data.shape}")
    print(f"  - w_data: {w_data.shape}")
    
    # 从4维数据中提取网格坐标（所有样本的坐标都相同）
    X = combined_data[0, :, :, 2]  # x坐标
    Y = combined_data[0, :, :, 3]  # y坐标
    
    print(f"数据生成完成! 形状: u={u_data.shape}, w={w_data.shape}")
    
    # 计算数据集统计信息
    print("\n数据集统计信息:")
    print(f"u - 最小值: {u_data.min():.4f}, 最大值: {u_data.max():.4f}, 均值: {u_data.mean():.4f}, 标准差: {u_data.std():.4f}")
    print(f"w - 最小值: {w_data.min():.4f}, 最大值: {w_data.max():.4f}, 均值: {w_data.mean():.4f}, 标准差: {w_data.std():.4f}")
    
    # 划分数据集
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val
    
    indices = np.random.permutation(num_samples)
    
    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train+num_val]
    test_idx = indices[num_train+num_val:]
    
    # 保存原始数据（包含4维格式）
    np.savez(os.path.join(save_dir, 'full_dataset.npz'), 
             combined_data=combined_data,  # 新增4维数据
             u=u_data, w=w_data, 
             X=X, Y=Y,
             nx=nx, ny=ny, delta=delta)
    
    # 保存分割后的数据（包含4维格式）
    np.savez(os.path.join(save_dir, 'train_data.npz'), 
             combined_data=combined_data[train_idx],
             u=u_data[train_idx], w=w_data[train_idx])
    np.savez(os.path.join(save_dir, 'val_data.npz'), 
             combined_data=combined_data[val_idx],
             u=u_data[val_idx], w=w_data[val_idx])
    np.savez(os.path.join(save_dir, 'test_data.npz'), 
             combined_data=combined_data[test_idx],
             u=u_data[test_idx], w=w_data[test_idx])
    
    # 转换为PyTorch格式并保存
    print("\n转换为PyTorch格式...")
    u_torch, w_torch, x_coords, y_coords = convert_to_torch_format(u_data, w_data)
    
    # 转换4维数据为PyTorch格式
    combined_torch = torch.tensor(combined_data, dtype=torch.float32)
    
    torch.save({
        # 传统格式（向后兼容）
        'u_train': u_torch[train_idx],
        'w_train': w_torch[train_idx],
        'u_val': u_torch[val_idx],
        'w_val': w_torch[val_idx],
        'u_test': u_torch[test_idx],
        'w_test': w_torch[test_idx],
        'x_coords': x_coords,
        'y_coords': y_coords,
        
        # 新的4维格式
        'combined_train': combined_torch[train_idx],
        'combined_val': combined_torch[val_idx], 
        'combined_test': combined_torch[test_idx],
        'combined_full': combined_torch,
        
        # 元数据
        'nx': nx,
        'ny': ny,
        'delta': delta,
        'data_format': '[u, w, x, y]'  # 说明4维数据的含义
    }, os.path.join(save_dir, 'pytorch_dataset.pt'))
    
    print(f"\n数据集保存完成!")
    print(f"训练集: {num_train} 样本")
    print(f"验证集: {num_val} 样本") 
    print(f"测试集: {num_test} 样本")
    print(f"\n数据格式:")
    print(f"  传统格式 - u: {u_torch.shape}, w: {w_torch.shape}")
    print(f"  坐标数据 - x: {x_coords.shape}, y: {y_coords.shape}")
    print(f"  4维格式 - combined: {combined_torch.shape} [u, w, x, y]")
    
    # 生成可视化样例
    visualize_samples(u_data, w_data, X, Y, save_dir, num_viz=6)
    
    return u_data, w_data, (train_idx, val_idx, test_idx)


def visualize_samples(u_data, w_data, X, Y, save_dir, num_viz=6):
    """可视化一些样本"""
    with plt.rc_context({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False}):
        indices = np.random.choice(len(u_data), min(num_viz, len(u_data)), replace=False)
        
        fig, axes = plt.subplots(2, num_viz, figsize=(3*num_viz, 6))
        
        for i, idx in enumerate(indices):
            # 输入u
            im1 = axes[0,i].contourf(X, Y, u_data[idx], levels=20, cmap='RdBu_r')
            axes[0,i].set_title(f'样本{idx+1}: u(x,y)')
            axes[0,i].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0,i])
            
            # 输出w
            im2 = axes[1,i].contourf(X, Y, w_data[idx], levels=20, cmap='viridis')
            axes[1,i].set_title(f'样本{idx+1}: w(x,y)')
            axes[1,i].set_aspect('equal')
            plt.colorbar(im2, ax=axes[1,i])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'sample_visualization.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"样本可视化保存到: {os.path.join(save_dir, 'sample_visualization.png')}")


def load_pytorch_dataset(data_path):
    """
    加载PyTorch格式的数据集，支持新的4维格式
    
    返回:
    data_dict: 包含所有数据的字典
    """
    data = torch.load(data_path)
    print(f"加载数据集: {data_path}")
    print(f"\n传统格式:")
    print(f"  训练集: u={data['u_train'].shape}, w={data['w_train'].shape}")
    print(f"  验证集: u={data['u_val'].shape}, w={data['w_val'].shape}")
    print(f"  测试集: u={data['u_test'].shape}, w={data['w_test'].shape}")
    print(f"  坐标: x={data['x_coords'].shape}, y={data['y_coords'].shape}")
    
    # 检查是否有新的4维格式
    if 'combined_train' in data:
        print(f"\n4维格式 {data.get('data_format', '[u, w, x, y]')}:")
        print(f"  训练集: {data['combined_train'].shape}")
        print(f"  验证集: {data['combined_val'].shape}")
        print(f"  测试集: {data['combined_test'].shape}")
        print(f"  完整数据: {data['combined_full'].shape}")
    
    return data


def create_neural_operator_dataloader(data_path, batch_size=32, split='train'):
    """
    创建适用于Neural Operator训练的DataLoader
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    data = torch.load(data_path)
    
    u_data = data[f'u_{split}']
    w_data = data[f'w_{split}']
    x_coords = data['x_coords']
    y_coords = data['y_coords']
    
    # 为每个样本复制坐标
    num_samples = u_data.shape[0]
    num_points = x_coords.shape[0]
    
    # 扩展坐标到所有样本
    x_expanded = x_coords.unsqueeze(0).repeat(num_samples, 1, 1)
    y_expanded = y_coords.unsqueeze(0).repeat(num_samples, 1, 1)
    
    dataset = TensorDataset(u_data, x_expanded, y_expanded, w_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    
    return dataloader


def create_combined_dataloader(data_path, batch_size=32, split='train'):
    """
    创建使用4维combined数据格式的DataLoader
    
    参数:
    data_path: 数据文件路径
    batch_size: 批大小
    split: 数据集分割 ('train', 'val', 'test')
    
    返回:
    dataloader: 返回combined数据的DataLoader
    数据格式: (batch_size, nx, ny, 4) where 4 = [u, w, x, y]
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    data = torch.load(data_path)
    
    if f'combined_{split}' not in data:
        raise ValueError(f"数据中没有找到 'combined_{split}' 格式，请使用更新后的数据生成器")
    
    combined_data = data[f'combined_{split}']
    
    print(f"加载4维数据格式: {combined_data.shape} - {data.get('data_format', '[u, w, x, y]')}")
    
    dataset = TensorDataset(combined_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成Neural Operator训练数据')
    parser.add_argument('--num_samples', type=int, default=1000, help='样本数量')
    parser.add_argument('--nx', type=int, default=64, help='x方向网格点数')
    parser.add_argument('--ny', type=int, default=64, help='y方向网格点数')
    parser.add_argument('--delta', type=float, default=1.0, help='Helmholtz方程参数')
    parser.add_argument('--save_dir', type=str, default='./neural_operator_data', help='保存目录')
    
    args = parser.parse_args()
    
    # 生成数据集
    u_data, w_data, splits = generate_diverse_dataset(
        num_samples=args.num_samples,
        nx=args.nx,
        ny=args.ny,
        delta=args.delta,
        save_dir=args.save_dir
    )
    
    # 测试数据加载
    print("\n测试数据加载...")
    data = load_pytorch_dataset(os.path.join(args.save_dir, 'pytorch_dataset.pt'))
    
    # 测试传统DataLoader
    print("\n测试传统DataLoader...")
    train_loader = create_neural_operator_dataloader(
        os.path.join(args.save_dir, 'pytorch_dataset.pt'), 
        batch_size=4, 
        split='train'
    )
    
    for batch_idx, (u_batch, x_batch, y_batch, w_batch) in enumerate(train_loader):
        print(f"  批次 {batch_idx}: u={u_batch.shape}, x={x_batch.shape}, y={y_batch.shape}, w={w_batch.shape}")
        if batch_idx >= 1:  # 只显示前2个批次
            break
    
    # 测试新的4维DataLoader
    print("\n测试4维DataLoader...")
    combined_loader = create_combined_dataloader(
        os.path.join(args.save_dir, 'pytorch_dataset.pt'),
        batch_size=4,
        split='train'
    )
    
    for batch_idx, (combined_batch,) in enumerate(combined_loader):
        print(f"  批次 {batch_idx}: combined={combined_batch.shape}")
        print(f"    u通道: {combined_batch[0, :, :, 0].shape}")
        print(f"    w通道: {combined_batch[0, :, :, 1].shape}")  
        print(f"    x通道: {combined_batch[0, :, :, 2].shape}")
        print(f"    y通道: {combined_batch[0, :, :, 3].shape}")
        if batch_idx >= 1:  # 只显示前2个批次
            break
    
    print("\n数据生成和测试完成!")

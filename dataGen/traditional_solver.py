import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
import torch

class HelmholtzSolver:
    """
    求解Helmholtz方程: -Δw + δw = u
    使用有限差分方法在2D矩形区域上求解
    边界条件: w = 0 (Dirichlet边界条件)
    """
    def __init__(self, nx, ny, x_domain=[0, 1], y_domain=[0, 1], delta=1.0):
        """
        参数:
        nx, ny: x和y方向的网格点数
        x_domain, y_domain: 计算区域
        delta: Helmholtz方程中的参数δ
        """
        self.nx = nx
        self.ny = ny
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.delta = delta
        
        # 计算网格步长
        self.dx = (x_domain[1] - x_domain[0]) / (nx - 1)
        self.dy = (y_domain[1] - y_domain[0]) / (ny - 1)
        
        # 创建网格
        self.x = np.linspace(x_domain[0], x_domain[1], nx)
        self.y = np.linspace(y_domain[0], y_domain[1], ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # 构建系数矩阵
        self._build_matrix()
    
    def _build_matrix(self):
        """构建有限差分系数矩阵"""
        # 内部点的数量 (去掉边界点)
        self.nx_inner = self.nx - 2
        self.ny_inner = self.ny - 2
        N = self.nx_inner * self.ny_inner
        
        # 创建系数矩阵 A，使得 A*w = u
        row = []
        col = []
        data = []
        
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2
        
        for j in range(self.ny_inner):  # y方向索引
            for i in range(self.nx_inner):  # x方向索引
                # 当前点的全局索引
                idx = j * self.nx_inner + i
                
                # 五点差分格式的系数
                center_coeff = 2.0/dx2 + 2.0/dy2 + self.delta
                
                # 中心点
                row.append(idx)
                col.append(idx)
                data.append(center_coeff)
                
                # 左邻点 (i-1, j)
                if i > 0:
                    row.append(idx)
                    col.append(idx - 1)
                    data.append(-1.0/dx2)
                
                # 右邻点 (i+1, j)
                if i < self.nx_inner - 1:
                    row.append(idx)
                    col.append(idx + 1)
                    data.append(-1.0/dx2)
                
                # 下邻点 (i, j-1)
                if j > 0:
                    row.append(idx)
                    col.append(idx - self.nx_inner)
                    data.append(-1.0/dy2)
                
                # 上邻点 (i, j+1)
                if j < self.ny_inner - 1:
                    row.append(idx)
                    col.append(idx + self.nx_inner)
                    data.append(-1.0/dy2)
        
        self.A = sp.csr_matrix((data, (row, col)), shape=(N, N))
    
    def solve(self, u_func):
        """
        求解Helmholtz方程
        
        参数:
        u_func: 右端项函数 u(x,y) 或者numpy数组
        
        返回:
        w: 解向量，形状为(nx, ny)
        """
        # 如果u_func是函数，则在网格上求值
        if callable(u_func):
            u_grid = u_func(self.X, self.Y)
        else:
            u_grid = u_func
        
        # 提取内部点的u值
        u_inner = u_grid[1:-1, 1:-1].flatten()
        
        # 求解线性系统 A*w = u
        w_inner = spsolve(self.A, u_inner)
        
        # 将解放回完整网格（边界为0）
        w_full = np.zeros((self.nx, self.ny))
        w_full[1:-1, 1:-1] = w_inner.reshape(self.nx_inner, self.ny_inner)
        
        return w_full
    
    def get_grid_points(self):
        """返回网格坐标点"""
        return self.X, self.Y
    
    def verify_solution(self, w, u_func):
        """验证解的精度，计算残差"""
        if callable(u_func):
            u_exact = u_func(self.X, self.Y)
        else:
            u_exact = u_func
        
        # 计算数值拉普拉斯算子
        laplacian = np.zeros_like(w)
        
        # 内部点的拉普拉斯算子（五点差分）
        for i in range(1, self.nx-1):
            for j in range(1, self.ny-1):
                d2w_dx2 = (w[i+1,j] - 2*w[i,j] + w[i-1,j]) / self.dx**2
                d2w_dy2 = (w[i,j+1] - 2*w[i,j] + w[i,j-1]) / self.dy**2
                laplacian[i,j] = d2w_dx2 + d2w_dy2
        
        # 计算残差: -Δw + δw - u
        residual = -laplacian + self.delta * w - u_exact
        
        # 只在内部点计算残差
        residual_inner = residual[1:-1, 1:-1]
        
        max_residual = np.max(np.abs(residual_inner))
        rms_residual = np.sqrt(np.mean(residual_inner**2))
        
        return max_residual, rms_residual, residual


def generate_training_data(num_samples, nx=64, ny=64, delta=1.0, save_path=None):
    """
    批量生成训练数据，包含坐标信息
    
    参数:
    num_samples: 训练样本数量
    nx, ny: 网格分辨率
    delta: Helmholtz方程参数
    save_path: 保存路径
    
    返回:
    combined_data: 形状为(num_samples, nx, ny, 3)的数组
                  最后一维：[u值, w值, x坐标, y坐标]
    u_data: 输入数据，形状为(num_samples, nx, ny) - 兼容性保留
    w_data: 输出数据，形状为(num_samples, nx, ny) - 兼容性保留
    """
    solver = HelmholtzSolver(nx, ny, delta=delta)
    X, Y = solver.get_grid_points()
    
    # 初始化数据数组
    combined_data = np.zeros((num_samples, nx, ny, 4))  # u, w, x, y
    u_data = []
    w_data = []
    
    print(f"正在生成 {num_samples} 个训练样本...")
    print(f"数据格式: [batch_size={num_samples}, nx={nx}, ny={ny}, features=4]")
    print(f"特征维度: [u值, w值, x坐标, y坐标]")
    
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"已生成 {i}/{num_samples} 个样本")
        
        # 生成随机右端项 u(x,y)
        u_sample = generate_random_source(X, Y, method='fourier')
        
        # 求解得到对应的w
        w_sample = solver.solve(u_sample)
        
        # 组合数据: [u, w, x, y]
        combined_data[i, :, :, 0] = u_sample  # u值
        combined_data[i, :, :, 1] = w_sample  # w值
        combined_data[i, :, :, 2] = X         # x坐标
        combined_data[i, :, :, 3] = Y         # y坐标
        
        # 保留原始格式以向后兼容
        u_data.append(u_sample)
        w_data.append(w_sample)
    
    u_data = np.array(u_data)
    w_data = np.array(w_data)
    
    if save_path:
        np.savez(save_path, 
                combined_data=combined_data,
                u=u_data, w=w_data, 
                X=X, Y=Y,
                nx=nx, ny=ny, delta=delta)
        print(f"训练数据已保存到 {save_path}")
        print(f"包含以下数据:")
        print(f"  - combined_data: {combined_data.shape}")
        print(f"  - u: {u_data.shape}")
        print(f"  - w: {w_data.shape}")
        print(f"  - X, Y: {X.shape}")
    
    return combined_data, u_data, w_data


# def generate_training_data_3d(num_samples, nx=64, ny=64, delta=1.0, coord_type='x', save_path=None):
#     """
#     生成3维训练数据: [bs, nx, ny, 3] = [u值, w值, 坐标值]
    
#     参数:
#     num_samples: 训练样本数量
#     nx, ny: 网格分辨率
#     delta: Helmholtz方程参数
#     coord_type: 坐标类型 ('x', 'y', 'r' - 距离中心的距离)
#     save_path: 保存路径
    
#     返回:
#     data_3d: 形状为(num_samples, nx, ny, 3)的数组
#              最后一维：[u值, w值, 坐标值]
#     """
#     solver = HelmholtzSolver(nx, ny, delta=delta)
#     X, Y = solver.get_grid_points()
    
#     # 选择坐标表示
#     if coord_type == 'x':
#         coord_values = X
#         coord_name = "x坐标"
#     elif coord_type == 'y':
#         coord_values = Y
#         coord_name = "y坐标"
#     elif coord_type == 'r':
#         # 距离中心的距离
#         center_x, center_y = 0.5, 0.5
#         coord_values = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
#         coord_name = "距中心距离"
#     else:
#         raise ValueError("coord_type必须是 'x', 'y' 或 'r'")
    
#     # 初始化数据数组
#     data_3d = np.zeros((num_samples, nx, ny, 3))
    
#     print(f"正在生成 {num_samples} 个3维训练样本...")
#     print(f"数据格式: [batch_size={num_samples}, nx={nx}, ny={ny}, features=3]")
#     print(f"特征维度: [u值, w值, {coord_name}]")
    
#     for i in range(num_samples):
#         if i % 100 == 0:
#             print(f"已生成 {i}/{num_samples} 个样本")
        
#         # 生成随机右端项 u(x,y)
#         u_sample = generate_random_source(X, Y, method='fourier')
        
#         # 求解得到对应的w
#         w_sample = solver.solve(u_sample)
        
#         # 组合数据: [u, w, coord]
#         data_3d[i, :, :, 0] = u_sample      # u值
#         data_3d[i, :, :, 1] = w_sample      # w值
#         data_3d[i, :, :, 2] = coord_values  # 坐标值
    
#     if save_path:
#         np.savez(save_path, 
#                 data_3d=data_3d,
#                 X=X, Y=Y, coord_values=coord_values,
#                 nx=nx, ny=ny, delta=delta, coord_type=coord_type)
#         print(f"3维训练数据已保存到 {save_path}")
#         print(f"数据形状: {data_3d.shape}")
#         print(f"坐标类型: {coord_name}")
    
#     return data_3d


# def generate_custom_training_data(num_samples, nx=64, ny=64, delta=1.0, 
#                                  features=['u', 'w', 'x'], save_path=None):
#     """
#     生成自定义训练数据，可以选择任意3个特征组合
    
#     参数:
#     num_samples: 训练样本数量
#     nx, ny: 网格分辨率  
#     delta: Helmholtz方程参数
#     features: 特征列表，从['u', 'w', 'x', 'y']中选择3个
#     save_path: 保存路径
    
#     返回:
#     custom_data: 形状为(num_samples, nx, ny, 3)的数组
#     """
#     if len(features) != 3:
#         raise ValueError("必须选择正好3个特征")
    
#     available_features = ['u', 'w', 'x', 'y'] 
#     for feat in features:
#         if feat not in available_features:
#             raise ValueError(f"无效特征 '{feat}'，可选特征: {available_features}")
    
#     solver = HelmholtzSolver(nx, ny, delta=delta)
#     X, Y = solver.get_grid_points()
    
#     # 初始化数据数组
#     custom_data = np.zeros((num_samples, nx, ny, 3))
    
#     print(f"正在生成 {num_samples} 个自定义训练样本...")
#     print(f"数据格式: [batch_size={num_samples}, nx={nx}, ny={ny}, features=3]")
#     print(f"选择的特征: {features}")
    
#     for i in range(num_samples):
#         if i % 100 == 0:
#             print(f"已生成 {i}/{num_samples} 个样本")
        
#         # 生成随机右端项 u(x,y)
#         u_sample = generate_random_source(X, Y, method='fourier')
        
#         # 求解得到对应的w
#         w_sample = solver.solve(u_sample)
        
#         # 创建特征字典
#         feature_dict = {
#             'u': u_sample,
#             'w': w_sample,
#             'x': X,
#             'y': Y
#         }
        
#         # 按照选择的特征填充数据
#         for j, feat in enumerate(features):
#             custom_data[i, :, :, j] = feature_dict[feat]
    
#     if save_path:
#         np.savez(save_path, 
#                 custom_data=custom_data,
#                 features=features,
#                 X=X, Y=Y,
#                 nx=nx, ny=ny, delta=delta)
#         print(f"自定义训练数据已保存到 {save_path}")
#         print(f"数据形状: {custom_data.shape}")
#         print(f"特征顺序: {features}")
    
#     return custom_data


def generate_random_source(X, Y, method='fourier'):
    """
    生成随机的源项函数 u(x,y)
    
    参数:
    X, Y: 网格坐标
    method: 生成方法 ('fourier', 'gaussian', 'polynomial')                                                                                                                       
    """
    if method == 'fourier':
        # 傅里叶级数展开
        u = np.zeros_like(X)
        num_modes = np.random.randint(3, 8)  # 随机选择3-7个模态
        
        for _ in range(num_modes):
            m = np.random.randint(1, 6)  # 波数
            n = np.random.randint(1, 6)
            amplitude = np.random.uniform(-2, 2)
            phase_x = np.random.uniform(0, 2*np.pi)
            phase_y = np.random.uniform(0, 2*np.pi)
            
            u += amplitude * np.sin(m * np.pi * X + phase_x) * np.sin(n * np.pi * Y + phase_y)
    
    elif method == 'gaussian':
        # 高斯函数的组合
        u = np.zeros_like(X)
        num_gaussians = np.random.randint(2, 5)
        
        for _ in range(num_gaussians):
            # 随机中心
            center_x = np.random.uniform(0.2, 0.8)
            center_y = np.random.uniform(0.2, 0.8)
            # 随机标准差
            sigma_x = np.random.uniform(0.1, 0.3)
            sigma_y = np.random.uniform(0.1, 0.3)
            # 随机幅度
            amplitude = np.random.uniform(-3, 3)
            
            gaussian = amplitude * np.exp(-((X - center_x)**2 / (2*sigma_x**2) + 
                                          (Y - center_y)**2 / (2*sigma_y**2)))
            u += gaussian
    
    elif method == 'polynomial':
        # 多项式
        u = np.zeros_like(X)
        max_degree = 4
        
        for i in range(max_degree):
            for j in range(max_degree - i):
                coeff = np.random.uniform(-1, 1)
                u += coeff * (X**i) * (Y**j)
    
    return u


def test_solver():
    """测试求解器的精度"""
    print("测试Helmholtz求解器...")
    
    # 创建求解器
    solver = HelmholtzSolver(nx=65, ny=65, delta=1.0)
    X, Y = solver.get_grid_points()
    
    # 测试案例1: 已知解析解
    def u_analytical(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    def w_analytical(x, y):
        # 对于 -Δw + δw = u，当u = sin(πx)sin(πy)时
        # 解析解为 w = u / (2π² + δ)
        return u_analytical(x, y) / (2 * np.pi**2 + solver.delta)
    
    # 生成右端项
    u_test = u_analytical(X, Y)
    
    # 数值求解
    w_numerical = solver.solve(u_test)
    
    # 解析解
    w_exact = w_analytical(X, Y)
    
    # 计算误差
    error = np.abs(w_numerical - w_exact)
    max_error = np.max(error)
    rms_error = np.sqrt(np.mean(error**2))
    
    print(f"最大误差: {max_error:.2e}")
    print(f"RMS误差: {rms_error:.2e}")
    
    # 验证残差
    max_res, rms_res, _ = solver.verify_solution(w_numerical, u_test)
    print(f"最大残差: {max_res:.2e}")
    print(f"RMS残差: {rms_res:.2e}")
    
    # 可视化结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    im1 = axes[0,0].contourf(X, Y, u_test, levels=20)
    axes[0,0].set_title('输入 u(x,y)')
    plt.colorbar(im1, ax=axes[0,0])
    
    im2 = axes[0,1].contourf(X, Y, w_exact, levels=20)
    axes[0,1].set_title('解析解 w(x,y)')
    plt.colorbar(im2, ax=axes[0,1])
    
    im3 = axes[0,2].contourf(X, Y, w_numerical, levels=20)
    axes[0,2].set_title('数值解 w(x,y)')
    plt.colorbar(im3, ax=axes[0,2])
    
    im4 = axes[1,0].contourf(X, Y, error, levels=20)
    axes[1,0].set_title(f'误差 (max={max_error:.2e})')
    plt.colorbar(im4, ax=axes[1,0])
    
    # 生成随机样本测试
    u_random = generate_random_source(X, Y, method='fourier')
    w_random = solver.solve(u_random)
    max_res_rand, rms_res_rand, residual = solver.verify_solution(w_random, u_random)
    
    im5 = axes[1,1].contourf(X, Y, u_random, levels=20)
    axes[1,1].set_title('随机输入')
    plt.colorbar(im5, ax=axes[1,1])
    
    im6 = axes[1,2].contourf(X, Y, w_random, levels=20)
    axes[1,2].set_title(f'随机解 (残差={rms_res_rand:.2e})')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig('solver_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return solver


if __name__ == "__main__":
    # 测试求解器
    solver = test_solver()
    
    # 生成小批量训练数据作为示例
    print("\n生成训练数据...")
    u_data, w_data = generate_training_data(
        num_samples=1000, 
        nx=64, 
        ny=64, 
        delta=-50.0,
        save_path="training_data_small.npz"
    )
    
    print(f"训练数据形状: u={u_data.shape}, w={w_data.shape}")

import torch
import numpy as np

def sample_interior(n_points):
    """均匀采样内部点 [0,1]x[0,1]"""
    return torch.tensor(np.random.rand(n_points, 2),device=device)


def get_points_inner(x_domain: list, y_domain: list, dist_points_num: int, device = torch.device("cpu"), requires_grad=True):
    soboleng = torch.quasirandom.SobolEngine(dimension=2)
    points_unit = soboleng.draw(dist_points_num)  # shape: (dist_points_num, 2)
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    x = x_min + (x_max - x_min) * points_unit[:, 0:1]
    y = y_min + (y_max - y_min) * points_unit[:, 1:2]
    x = x.to(device).clone().detach().requires_grad_(requires_grad)
    y = y.to(device).clone().detach().requires_grad_(requires_grad)
    return x, y



def get_points_bd(x_domain: list, y_domain: list, num_points: int, device = torch.device("cpu"), requires_grad=True):
    """
    在二维矩形边界均匀采样num_points个点，返回x、y两个tensor。
    Args:
        x_domain (list): [x_min, x_max]
        y_domain (list): [y_min, y_max]
        num_points (int): 边界总采样点数
        device: torch.device
        requires_grad: 是否对tensor开启梯度
    Returns:
        x_bd, y_bd: shape=(num_points, 1)
    """
    x_min, x_max = x_domain
    y_min, y_max = y_domain

    # 四条边等分点数，多余补到前几条边
    num_each = [num_points // 4] * 4
    for i in range(num_points % 4):
        num_each[i] += 1

    # bottom (x_min->x_max, y_min)
    x_b = torch.linspace(x_min, x_max, steps=num_each[0], device=device).unsqueeze(1)
    y_b = torch.full((num_each[0], 1), y_min, device=device)

    # right (x_max, y_min->y_max)
    y_r = torch.linspace(y_min, y_max, steps=num_each[1], device=device).unsqueeze(1)
    x_r = torch.full((num_each[1], 1), x_max, device=device)

    # top (x_max->x_min, y_max)
    x_t = torch.linspace(x_max, x_min, steps=num_each[2], device=device).unsqueeze(1)
    y_t = torch.full((num_each[2], 1), y_max, device=device)

    # left (x_min, y_max->y_min)
    y_l = torch.linspace(y_max, y_min, steps=num_each[3], device=device).unsqueeze(1)
    x_l = torch.full((num_each[3], 1), x_min, device=device)

    # 合并所有边
    x_bd = torch.cat([x_b, x_r, x_t, x_l], dim=0).clone().detach().requires_grad_(requires_grad)
    y_bd = torch.cat([y_b, y_r, y_t, y_l], dim=0).clone().detach().requires_grad_(requires_grad)

    return x_bd, y_bd


def get_points(x_domain: list, y_domain: list, dist_points_num: int , device = torch.device("cpu"), requires_grad=True):
    """This generates a 2D grid of points.
      dist_points_num (short for distinct points number) is the number of points in the interval [start,end].
      The generated grid consists of dist_points_num**2 number of points"""
    
    x_raw = torch.linspace(x_domain[0], x_domain[1], steps=dist_points_num, requires_grad=requires_grad)
    y_raw = torch.linspace(y_domain[0], y_domain[1], steps=dist_points_num, requires_grad=requires_grad)
    grids = torch.meshgrid(x_raw, y_raw, indexing='ij')
    x = grids[0].reshape(-1, 1).to(device)
    y = grids[1].reshape(-1, 1).to(device)
    return x, y



def df(output: torch.Tensor, input: torch.Tensor, order: int = 1) -> torch.Tensor:
    df_value = output
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            input,
            grad_outputs=torch.ones_like(input),
            create_graph=True,
            retain_graph=True,
        )[0]
    return df_value


# PDE损失函数：计算残差
def pde_loss(u_now, w_now, sigma, x_points, y_points):
    psi_dx = df(w_now, x_points, 1)
    psi_dy = df(w_now, y_points, 1)
    psi_ddx= df(w_now, x_points, 2)
    psi_ddy= df(w_now, y_points, 2)
    
    pde_loss_ = -psi_ddx - psi_ddy - u_now - sigma * w_now
    
    grad_u = psi_dx+psi_dy # 计算梯度）
    numerator = torch.sum(grad_u**2)  # 计算梯度的平方
    denominator = torch.sum(u_now**2)  # 计算u的平方
    return torch.mean(pde_loss_ ** 2), numerator / denominator
    


def rayleigh_loss(lambda_now, lambda_pre):
    res = torch.abs(lambda_now-lambda_pre)
    return res # 返回Rayleigh商的差异


# 停止条件：检查是否满足停止条件
def stop_criteria(rayleigh_diff, u_diff, max_iters, current_iter, residual, epsilon_1=1e-5, epsilon_2=1e-5, epsilon_3=1e-5):
    if rayleigh_diff < epsilon_1 and u_diff < epsilon_2 and residual < epsilon_3:  # 如果 Rayleigh 商差异小于阈值
        return True
    # if u_diff < epsilon_2:  # 如果特征向量差异小于阈值
    #     return True
    # if residual < epsilon_3:  # 如果残差小于阈值
    #     return True
    # if current_iter >= max_iters:  # 如果超过最大迭代次数
    #     return True
    return False

#-----------------------未使用的函数---------------------#

def rayleigh_value(u_now):
    # 预测解 
    grad_u = torch.gradient(u_now)  # 计算梯度（仅对一个维度计算）
    numerator = torch.sum(grad_u**2)  # 计算梯度的平方
    denominator = torch.sum(u_now**2)  # 计算u的平方
    return numerator / denominator

    
def compute_rayleigh(u):
    """
    计算 Rayleigh 商
    参数：
    u -- 特征向量，形状为 (N, 1)，每个空间点的解
    dx -- 网格步长，用于计算梯度
    
    返回：
    lambda_hat -- Rayleigh 商的估计值
    """

    
    # 计算梯度 (自动微分)
    grad_u = torch.autograd.grad(u.sum(), u, create_graph=True)[0]  # 对 u 求导
    
    # 计算 L2 范数 ||∇u||^2
    grad_u_sq = grad_u**2  # 梯度的平方
    numerator = torch.sum(grad_u_sq)  # 积分的近似值：∫Ω ||∇u||^2
    
    # 计算 u^2
    u_sq = u**2  # u 的平方
    denominator = torch.sum(u_sq)  # 积分的近似值：∫Ω u^2
    
    # 计算 Rayleigh 商
    lambda_hat = numerator / denominator
    
    return lambda_hat



def pde_loss2(u_x, w_now, sigma, x_points, y_points):
    # 确保坐标可导
    x_points.requires_grad_(True)
    y_points.requires_grad_(True)

    # d^2/dx^2（保图，因后面还要对 w_now 求 d/dy）
    du_dx  = torch.autograd.grad(w_now, x_points, grad_outputs=torch.ones_like(w_now),
                                 create_graph=True, retain_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_points, grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True, retain_graph=True)[0]

    # d^2/dy^2（最后一条分支，可以不保）
    du_dy  = torch.autograd.grad(w_now, y_points, grad_outputs=torch.ones_like(w_now),
                                 create_graph=True, retain_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y_points, grad_outputs=torch.ones_like(du_dy),
                                  create_graph=True, retain_graph=False)[0]

    residual = d2u_dx2 + d2u_dy2 + u_x + sigma * w_now
    return (residual ** 2).mean()

def rayleigh_quotient_and_loss(u, coords, w_now, sigma):
    # grad_coords, = torch.autograd.grad(
    #     outputs=u,
    #     inputs=coords,
    #     grad_outputs=torch.ones_like(u),
    #     create_graph=True,
    #     retain_graph=True,
    #     only_inputs=True
    # )
    grads = torch.autograd.grad(
        outputs=u, inputs=coords, grad_outputs=torch.ones_like(u), 
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    lap = torch.autograd.grad(
        grads, coords, grad_outputs=torch.ones_like(grads), 
        create_graph=True, only_inputs=True
    )[0]
    laplacian = lap.sum(dim=1)
    
    loss = laplacian + u + sigma * w_now
    # grad_coords: (N,2) -> |∇u|^2
    grad_norm2 = (grads**2).sum(dim=1, keepdim=True)
    numerator = grad_norm2.sum()
    denominator = (u**2).sum() + 1e-12
    
    return numerator / denominator, loss

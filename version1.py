import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import copy
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import random
from datetime import datetime
from util import *

#定义神经网络（MLP）
class NeuralOperator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralOperator, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, u, x_points, y_points):
        x = torch.cat((u, x_points, y_points), dim=1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)
    
#     def rayleigh_quotient(self, u, x_points, y_points):
        
#         # coords: (N, 2), u: (N,1)
#         #coords = torch.cat((x_points, y_points), dim=1)

#         # grad_u_x, grad_u_y = torch.autograd.grad(
#         #     outputs=u, inputs=[x_points, y_points], grad_outputs=torch.ones_like(u), 
#         #     create_graph=True, retain_graph=True, only_inputs=True
#         # )
#         grad_u_x, grad_u_y = torch.autograd.grad(
#         outputs=u,
#         inputs=[x_points, y_points],
#         grad_outputs=torch.ones_like(u),
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True
#     )
#         grad_norm2 = grad_u_x**2 + grad_u_y**2
#         numerator = grad_norm2.sum(dim=0)
#         denominator = (u ** 2).sum(dim=0)
#         return numerator / denominator

# class NeuralOperator(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.layer1 = nn.Linear(input_dim, hidden_dim)
#         self.layer2 = nn.Linear(hidden_dim, hidden_dim)
#         self.layer3 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x, coords):
#         x = torch.cat((x, coords), dim=1)  # (N, 1+2)
#         x = torch.relu(self.layer1(x))
#         x = torch.relu(self.layer2(x))
#         return self.layer3(x)

#     def rayleigh_quotient(self, u, coords):
#         grad_coords, = torch.autograd.grad(
#             outputs=u,
#             inputs=coords,
#             grad_outputs=torch.ones_like(u),
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True
#         )
#         # grad_coords: (N,2) -> |∇u|^2
#         grad_norm2 = (grad_coords**2).sum(dim=1, keepdim=True)
#         numerator = grad_norm2.sum()
#         denominator = (u**2).sum() + 1e-12
#         return numerator / denominator

# 神经网络训练过程
def train_neural_operator(model, u_now, sigma, lambda_val, num_epochs=100, lr=1e-3, max_iters=100):
    
    x_points, y_points = get_points(x_domain, y_domain, dist_points_num=100, requires_grad=True, device=device)
    interior = sample_interior
    #x_points_bd, y_points_bd = get_points_bd(x_domain, y_domain, points_bound, requires_grad=True, device=device)
    
    lambda_before = torch.tensor(0,dtype=torch.float32)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    with tqdm(total=num_epochs, desc="Training", ncols=150) as pbar:
        for epoch in range(num_epochs):
            w_now = model(u_now, x_points, y_points)
            #lambda_now = model.rayleigh_quotient(w_now, x_points, y_points)
            
            #coords = torch.cat((x_points, y_points), dim=1)  # (N,2), requires_grad=True
            #w_now = model(u_now, coords)
            
            #lambda_now, pde_loss_value = rayleigh_quotient_and_loss(u = u_now, coords=coords, w_now= w_now ,sigma=sigma)
            lambda_now, pde_loss_value = pde_loss(u_now, w_now, sigma, x_points, y_points)
            
            total_loss_value = pde_loss_value
            
            optimizer.zero_grad()
            total_loss_value.backward()  # 反向传播
            optimizer.step()  # 更新参数    
            
            
            # with torch.no_grad():
            #     lambda_now = model.rayleigh_quotient(w_now, coords)
                
            #u_new = w_now / torch.norm(w_now) #外层循环：更新特征向量     
            # 计算 PDE 损失
            with torch.no_grad():
                u_new = w_now / (w_now.norm() + 1e-12)
            u_diff = torch.norm(u_new - u_now)  # 计算特征向量的差异
            u_now = u_new.detach()


            # 外层循环更新特征向量
            # 只能是后一次循环比较前一次循环
            rayleigh_diff = rayleigh_loss(lambda_now, lambda_before)
            lambda_before = lambda_now
            

            # 检查停止条件
            if stop_criteria(rayleigh_diff, u_diff, max_iters, current_iter= epoch, residual=total_loss_value.item()):
                print(f"{rayleigh_diff:4f} | {u_diff:4f} | {total_loss_value.item():4f}")
                print(f"Stopping criteria met at epoch {epoch+1}")
                break
            
            if epoch % 100 == 0:
                se_info = f"rayleigh: {rayleigh_diff.item():.4f}"
                norm_info = f"u_diff: {u_diff.item():.4f}"
                #bias_info = f"bias: {bias_loss.item():.4f}"
                total_info = f"total: {total_loss_value.item():.4f}"
                lambda_info = f"λ: {lambda_now:.4f}"
                pbar.set_postfix_str(f"{se_info} | {norm_info} | {total_info} | {lambda_info}")
            pbar.update(1)


# 示例：定义输入输出维度，模型和数据
input_dim = 3
hidden_dim = 64
output_dim = 1
sample_num = 100

# 区域定义
x = np.linspace(0, 1, sample_num)
y = np.linspace(0, 1, sample_num)
X, Y = np.meshgrid(x, y)
dx = x[1] - x[0]
dy = y[1] - y[0]

# 特征函数: 
m=1
n=2
u_gt = np.sin(m * np.pi * X) * np.sin(n* np.pi * Y)
lambda_val = m**2 + n**2  # 5

x_domain = [0, 1]      
y_domain = [0, 1] 


#Hyperparameter
weight_bd = 1.0
sigma = 50.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#需要迭代的u, 随机特征函数
u_init = torch.randn((sample_num, sample_num), device=device,requires_grad=True)
u_init = u_init.reshape(-1,1)


model = NeuralOperator(input_dim, hidden_dim, output_dim)
model.to(device)


train_neural_operator(model=model, u_now=u_init, sigma=sigma, lambda_val=lambda_val, num_epochs=10000)

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:12:59 2025

@author: Big33
"""

import numpy as np
import cvxopt
from cvxopt import matrix, solvers

def setup_qp_and_solve(vec):
    """
    改进1&5: 高效QP求解器实现
    目标函数: 最小化 ||d - 平均梯度||² ，其中 d = sum(w_i * g_i), w_i为权重
    约束条件:
        1. 权重非负且和为1 (w >= 0, sum(w) = 1)
        2. 公平性角度约束（隐式通过 vec 输入包含公平性梯度修正项）
    """
    # ------------------------- 参数初始化 -------------------------
    m, n = vec.shape  # m=客户端数, n=参数维度
    cvxopt.solvers.options['show_progress'] = False  # 关闭求解日志

    # ------------------------- 构造QP问题 -------------------------
    # 目标函数: 0.5 * w^T * P * w + q^T * w
    # 等价于最小化 ||d - 平均梯度||² = ||vec.T @ w - mean_grad||²
    # 其中 mean_grad = (1/m) * sum(vec[i])
    # 展开后 P = vec @ vec.T, q = -vec @ mean_grad
    mean_grad = np.mean(vec, axis=0)  # 平均梯度
    P = np.dot(vec, vec.T)            # P矩阵 (m x m)
    q = -np.dot(vec, mean_grad)       # q向量 (m x 1)

    # 约束条件:
    # 1. 权重非负: G * w <= h → -w <= 0 → G = -I, h = 0
    G = -np.eye(m)                   
    h = np.zeros(m)
    
    # 2. 权重和为1: A * w = b → A = [1, 1, ..., 1], b = [1]
    A = np.ones((1, m))              
    b = np.array([1.0])

    # ------------------------- 转换为cvxopt格式 -------------------------
    P_cvx = matrix(P.astype(np.double))
    q_cvx = matrix(q.astype(np.double))
    G_cvx = matrix(G.astype(np.double))
    h_cvx = matrix(h.astype(np.double))
    A_cvx = matrix(A.astype(np.double))
    b_cvx = matrix(b.astype(np.double))

    # ------------------------- 求解QP问题 -------------------------
    try:
        sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
        if sol['status'] == 'optimal':
            w = np.array(sol['x']).flatten()  # 权重向量 (m x 1)
        else:
            # 求解失败时回退为平均权重
            w = np.ones(m) / m  
    except:
        # 数值不稳定时使用均匀权重
        w = np.ones(m) / m  

    return w, None  # 返回权重和占位符（与原代码兼容）
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:22:37 2025

@author: Big33
"""

import torch
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from .qp_solver import setup_qp_and_solve

import torch
import numpy as np
from .qp_solver import setup_qp_and_solve

def get_fedmdfgm_d(
  grads, 
  value, 
  add_grads, 
  alpha, 
  fair_guidance_vec, 
  force_active, 
  device, 
  k_sparse=0.2, 
  use_projection=False, 
  projected_dim=128
):
  """改进后的梯度聚合方向生成函数，解决维度不匹配问题"""
  m, original_dim = grads.shape  # 原始维度: 模型参数总数
  k = max(1, int(k_sparse * m))  # 至少保留1个客户端
  
  # === 1. 稀疏梯度选择 ===
  impact_scores = torch.abs(fair_guidance_vec @ grads)
  topk_indices = torch.topk(impact_scores, k=k).indices
  topk_indices = torch.clamp(topk_indices, 0, m-1)
  grads_sparse = grads[topk_indices] # (k, original_dim)

  # === 2. 随机投影降维 ===
  if use_projection:
    if not hasattr(get_fedmdfgm_d, 'proj_matrix'):
      proj = torch.randn(original_dim, projected_dim, device=device)
      proj /= torch.norm(proj, dim=0) # 列归一化
      get_fedmdfgm_d.proj_matrix = proj
    else:
      proj = get_fedmdfgm_d.proj_matrix
    grads_proj = grads_sparse @ proj # (k, projected_dim)
  else:
    grads_proj = grads_sparse
    projected_dim = original_dim

  # === 3. 公平性修正 ===
  fair_guidance_vec_sparse = fair_guidance_vec[topk_indices]
  value_sparse = value[topk_indices]
  value_norm = value_sparse / torch.norm(value_sparse)
  cos = torch.clamp(value_norm @ fair_guidance_vec_sparse, -1, 1)
  bias = torch.acos(cos) * 180 / np.pi
  pref_active = (bias > alpha) | force_active

  if pref_active:
    h_vec = (fair_guidance_vec_sparse @ value_norm * value_norm - fair_guidance_vec_sparse)
    h_vec /= torch.norm(h_vec)
    fair_grad = h_vec @ grads_proj # (projected_dim,)
    vec = torch.cat([grads_proj, fair_grad.unsqueeze(0)], dim=0) # (k+1, projected_dim)
  else:
    vec = grads_proj # (k, projected_dim)
    fair_grad = None

  # === 4. 融合历史梯度 ===
  if add_grads is not None:
    if use_projection:
      add_grads_proj = add_grads @ proj # 历史梯度投影
    else:
      add_grads_proj = add_grads
    vec = torch.cat([vec, add_grads_proj])
  # 求解QP
  sol, _ = setup_qp_and_solve(vec.cpu().numpy())
  d = torch.from_numpy(sol).float().to(device) @ vec

  return d, vec, int(pref_active), fair_grad if pref_active else None
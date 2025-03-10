# -*- coding: utf-8 -*-
import fedplat as fp
import copy
import math
import numpy as np
import torch
from fedplat.algorithm.common.utils import get_fedmdfg_d
from .FedMDFGM import get_fedmdfgm_d

"""
Code of FedMDFG.

"""

class ImprovedFedMDFG(fp.Algorithm):
    def __init__(self,
                 name='ImprovedFedMDFG_personal',
                 data_loader=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 metric_list=None,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 update_client=False,
                 write_log=True,
                 params=None,
                 theta=11.25,  # tolerable fairness angle. theta=11.25 is corresponding to theta=pi/16 in the paper. In the codes we use the degree unit.
                 s=5,  # line search parmeter
                 *args,
                 **kwargs):

        if params is not None:
            theta = params['theta']
            s = params['s']
        if save_name is None:
            save_name = name + ' ' + model.name + ' E' + str(epochs) + ' lr' + str(train_setting['optimizer'].defaults['lr']) + ' decay' + str(train_setting['lr_decay']) + ' theta' + str(theta) + ' s' + str(s)
        super().__init__(name, data_loader, model, device, train_setting, client_num, client_list, metric_list,
                         max_comm_round, max_training_num, epochs, save_name, outFunc, update_client, write_log)
        # check parameters
        if theta <= 0 or theta >= 90:
            raise RuntimeError('illegal parameter setting.')
        if s < 0:
            raise RuntimeError('illegal parameter setting.')
        self.theta = theta
        self.s = s
        self.last_client_id_list = None
        self.last_g_locals = None
        self.last_d = None
        self.client_expected_loss = [None] * self.data_loader.pool_size
        self.client_join_count = [0] * self.data_loader.pool_size
        self.same_user_flag = True
        self.prefer_active = 0
        self.client_sizes = None
        self.client_weights = None
        self.alpha = 0.9
    def calculate_client_sizes(self):
        # Calculate client sizes based on data volume, update speed, and predefined weights
        data_volumes = self.send_require_attr('data_volume')
        update_speeds = self.send_require_attr('update_speed')
        predefined_weights = self.send_require_attr('predefined_weight')

        # Normalize each factor
        data_volumes = np.array(data_volumes) / np.max(data_volumes)
        update_speeds = np.array(update_speeds) / np.max(update_speeds)
        predefined_weights = np.array(predefined_weights) / np.sum(predefined_weights)

        # Combine factors to determine client sizes
        self.client_sizes = (data_volumes + update_speeds + predefined_weights) / 3

    def update_client_weights(self, l_locals):
        if self.client_weights is None:
            self.client_weights = self.client_sizes / np.sum(self.client_sizes)

        # Calculate the inverse of losses to give higher weight to clients with lower loss
        inverse_losses = 1 / (l_locals.cpu().numpy() + 1e-8)  # Adding small epsilon to avoid division by zero

        # Normalize the inverse losses
        normalized_inverse_losses = inverse_losses / np.sum(inverse_losses)

        # Update weights using a combination of predefined weights and dynamic adjustment
        updated_weights = self.alpha * self.client_weights + (1 - self.alpha) * normalized_inverse_losses
        
        # Normalize the updated weights
        self.client_weights = updated_weights / np.sum(updated_weights)

    def calculate_fair_guidance_vec(self, l_locals):
        self.update_client_weights(l_locals)
        
        # Calculate relative fairness
        mean_loss = torch.mean(l_locals)
        relative_fairness = (l_locals - mean_loss) / mean_loss

        # Combine relative fairness with client weights
        fair_guidance_vec = torch.tensor(self.client_weights).to(self.device) * (1 + relative_fairness)

        # Normalize the fair guidance vector
        fair_guidance_vec = fair_guidance_vec / torch.sum(fair_guidance_vec)

        return fair_guidance_vec.float()

    def update_model(self, model, d, lr):
        self.optimizer = self.train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        for i, p in enumerate(model.parameters()):
            p.grad = d[self.model.Loc_reshape_list[i]]
        self.optimizer.step()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert d.shape[0] == total_params
        return model

    def line_search(self, g_locals, d, fair_guidance_vec, fair_grad, base_lr, l_locals_0, live_idx, scale):#线性搜索
        old_loss_norm = float(torch.sum(l_locals_0))
        fair_guidance_vec_norm = torch.norm(fair_guidance_vec)
        old_cos = l_locals_0 @ fair_guidance_vec / (old_loss_norm * fair_guidance_vec_norm)
        beta = 1e-4
        c = -(g_locals@d)
        if self.same_user_flag:
            lr = float(2**self.s * base_lr)
        else:
            lr = float(base_lr)
        old_model = copy.deepcopy(self.model.state_dict())
        min_lr = float(0.5**self.s * base_lr / scale)
        lr_storage = []
        norm_storage = []
        while lr >= min_lr:
            self.model.load_state_dict(old_model)
            temp_model = self.update_model(self.model, d, lr)
            # evaluate temporary model
            # Note that here we use such a way just for convenient, that we reuse the framework to copy the model to all clients.
            # In fact, we don't need this step, just send the direction d^t to clients before the step size line search,
            # and then just send lr to clients and let clients update a local temporary model by theirselves instead.
            self.send_sync_model(update_count=False, model=temp_model)
            self.send_cal_all_batches_loss_order()
            l_locals = self.send_require_cal_all_batches_loss_result()
            l_locals = l_locals[live_idx]
            # store the loss norm
            l_locals_norm = float(torch.sum(l_locals))
            lr_storage.append(lr)
            norm_storage.append(l_locals_norm)
            # stop criterion
            if self.prefer_active == 0 and torch.all(l_locals_0 - l_locals >= lr * beta * c):
                lr_storage = []
                norm_storage = []
                break
            elif self.prefer_active == 1 and torch.all(l_locals_0 - l_locals >= lr * beta * c) and (l_locals @ fair_guidance_vec) / (torch.norm(l_locals) * fair_guidance_vec_norm) - old_cos > 0:
                lr_storage = []
                norm_storage = []
                break
            lr /= 2
        if len(norm_storage) > 0:
            for idx, l_locals_norm in enumerate(norm_storage):
                lr = lr_storage[idx]
                if lr > base_lr and self.same_user_flag == False:
                    continue
                if l_locals_norm < old_loss_norm:
                    norm_storage = []
                    break
        if len(norm_storage) > 0:
            best_idx = np.argmin(norm_storage)
            lr = lr_storage[best_idx]
        self.model.load_state_dict(old_model)
        self.model = self.update_model(self.model, d, lr)

    def train_a_round(self):#训练一轮
        self.model.train() # 本地训练
        self.prefer_active = 0
        self.send_sync_model()  # Here we reuse the framework to copy the model to all clients. In fact, we only need to send the global model to those new-come clients.
        if self.client_sizes is None:
            self.calculate_client_sizes()
        self.send_cal_all_batches_gradient_loss_order()
        g_locals, l_locals = self.send_require_all_batches_gradient_loss_result()# 获取梯度与损失，g_locals客户端梯度，l_locals客户端损失
        # print(g_locals)
        client_id_list = self.send_require_attr('id')
        force_active = False
        increase_count = 0
        for i, client_id in enumerate(client_id_list):
            if self.client_join_count[client_id] == 0:
                self.client_expected_loss[client_id] = l_locals[i]
            else:
                if l_locals[i] <= self.client_expected_loss[client_id]:
                    self.client_expected_loss[client_id] = (self.client_expected_loss[client_id] * self.client_join_count[client_id] + l_locals[i]) / (self.client_join_count[client_id] + 1)
                else:
                    if l_locals[i] > self.client_expected_loss[client_id]:
                        increase_count += 1
            self.client_join_count[client_id] += 1
        if increase_count > 0 and increase_count < self.client_num:
            force_active = True
        # historical fairness
        if self.last_client_id_list is not None:
            add_idx = []
            for idx, last_client_id in enumerate(self.last_client_id_list):
                if last_client_id not in client_id_list:
                    add_idx.append(idx)
            if len(add_idx) > 0:
                add_grads = self.last_g_locals[add_idx, :]
                self.same_user_flag = False
            else:
                add_grads = None
                self.same_user_flag = True
        else:
            add_grads = None
            self.same_user_flag = True
        grad_local_norm = torch.norm(g_locals, dim=1)# 归一化梯度
        live_idx = torch.where(grad_local_norm > 1e-6)[0] # 获取有效客户端模型索引位置列表
        if len(live_idx) == 0:
            return
        if len(live_idx) > 0:
            g_locals = g_locals[live_idx, :]
            l_locals = l_locals[live_idx]
            grad_local_norm = torch.norm(g_locals, dim=1)
        # scale the outliers of all gradient norms
        miu = torch.mean(grad_local_norm)
        g_locals = g_locals / grad_local_norm.reshape(-1, 1) * miu
        # 原算法
        # fair_guidance_vec = torch.Tensor([1.0] * len(live_idx)).to(self.device)
        # 第一次改进
        # fair_guidance_vec = l_locals[live_idx]/torch.mean(l_locals[live_idx])
        # 第二次改进
        # fair_guidance_vec = (l_locals[live_idx]-l_locals[live_idx].min())/(l_locals[live_idx].max() - l_locals[live_idx].min())
        # 下面这个算法也是曾经的改进方案，最后*100可以改成其他的倍率
        # fair_guidance_vec = 1 + (fair_guidance_vec - 1)*100
        # for i in range(len(live_idx)):
        #     vec = l_locals
        fair_guidance_vec = self.calculate_fair_guidance_vec(l_locals[live_idx])
        # 下面这个函数应该就是计算q的部分了，但是没有完全理解
        # calculate d
        # d, vec, p_active_flag, fair_grad = get_fedmdfg_d(g_locals, l_locals, add_grads, self.theta, fair_guidance_vec, force_active, self.device)
        d, vec, p_active_flag, fair_grad = get_fedmdfg_d(g_locals, l_locals, add_grads, self.theta, fair_guidance_vec, force_active, self.device)
        # print("d:",d,"vec:",vec,"p_active_flag",p_active_flag,"fair_grad",fair_grad)
        if p_active_flag == 1:
            self.prefer_active = 1
        # Update parameters of the model
        # line search
        weights = torch.Tensor([1 / len(live_idx)] * len(live_idx)).float().to(self.device)
        g_norm = torch.norm(weights @ g_locals)
        d_norm = torch.norm(d)
        min_lr = self.lr
        d_old = copy.deepcopy(d)
        d = d / d_norm * g_norm
        # prevent the effects of the float or double type, it can be sikpped theoretically
        while torch.max(-(vec @ d)) > 1e-6:
            if torch.norm(d) > d_norm * 2:
                d /= 2
            else:
                d = d_old
                break
        scale = torch.norm(d) / torch.norm(d_old)
        self.line_search(g_locals, d, fair_guidance_vec, fair_grad, min_lr, l_locals, live_idx, scale)# 线性搜索
        self.current_training_num += 1
        self.last_client_id_list = self.send_require_attr('id')
        self.last_client_id_list = [self.last_client_id_list[live_idx[i]] for i in range(len(live_idx))]
        self.last_g_locals = copy.deepcopy(g_locals)
        self.last_d = d

    def run(self):
        while not self.terminated():
            self.train_a_round()


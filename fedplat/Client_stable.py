import fedplat as fp
import torch
import copy
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from .PersonalizedModel import PersonalizedModel
from torch.nn.parallel import DataParallel  

class Client:
    def __init__(self,
                 id=None,
                 model=None,
                 device=None,
                 train_setting=None,
                 metric_list=None,
                 dishonest=None,
                 *args,
                 **kwargs):
        self.id = id
        if model is not None:
            model = model
        # model = PersonalizedModel(model,96)
        self.model = model
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        device_ids = [0]
        self.model = DataParallel(model, device_ids=device_ids).to(device_ids[0])  
        self.model.to(self.device)
        self.train_setting = train_setting
        self.metric_list = metric_list
        self.dishonest = dishonest
        self.local_training_data = None
        self.local_training_number = 0
        self.local_test_data = None
        self.local_test_number = 0
        self.training_batch_num = 0
        self.test_batch_num = 0
        self.metric_history = {'training_loss': [],
                               'test_loss': [],
                               'local_test_number': 0}
        for metric in self.metric_list:
            self.metric_history[metric.name] = []  
            if metric.name == 'correct':
                self.metric_history['test_accuracy'] = []  
        self.model_weights = None  
        self.model_loss = None  
        self.info_msg = {}  
        self.update_speed = self.id/100 + 1
        self.predefined_weight = 1
        self.quantization_level = 65536
        self.compression_ratio = 1.0
        self.initial_lr = float(train_setting['optimizer'].defaults['lr'])
        self.lr = self.initial_lr
        self.optimizer = train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
        self.optimizer.defaults = copy.deepcopy(train_setting['optimizer'].defaults)
        self.criterion = self.train_setting['criterion'].to(self.device)
        self.old_model = copy.deepcopy(self.model)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        self.init_weights()
        
    def init_weights(self):
        def init_func(m):
            if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        self.model.apply(init_func)

    def update_data(self, id, local_training_data, local_training_number, local_test_data, local_test_number):
        self.id = id
        self.local_training_data = local_training_data
        self.local_training_number = local_training_number
        self.local_test_data = local_test_data
        self.local_test_number = local_test_number
        self.training_batch_num = len(local_training_data)
        self.test_batch_num = len(local_test_data)
        self.data_volume = len(self.local_training_data)

    def batch_compress(self, return_grad):
        # 将梯度列表转换为二维张量 [batch_size, features]
        stacked_grad = torch.stack([g.view(-1) for g in return_grad])  # [B, N]
        
        # 批量计算Top-k索引
        k = int(self.compression_ratio * stacked_grad.size(1))
        _, topk_indices = torch.topk(stacked_grad.abs(), k, dim=1)     # [B, k]
        
        # 批量生成掩码
        batch_indices = torch.arange(stacked_grad.size(0), device=stacked_grad.device)[:, None]
        mask = torch.zeros_like(stacked_grad)
        mask[batch_indices, topk_indices] = 1  # 向量化散射操作
        
        # 批量稀疏化
        sparse_grad = stacked_grad * mask
        
        # 批量量化
        max_vals = sparse_grad.abs().max(dim=1, keepdim=True).values  # [B, 1]
        max_vals = max_vals.clamp_min(1e-8)  # 防止除零
        
        quantized = (sparse_grad / max_vals * (self.quantization_level - 1))
        quantized = quantized.round_().mul_(max_vals / (self.quantization_level - 1))
        
        # 恢复原始形状
        compressed_g_locals = [q.view_as(g) for q, g in zip(quantized, return_grad)]
        masks = [m.view_as(g) for m, g in zip(mask, return_grad)]
        
        return compressed_g_locals, masks
    def simulate_compression(self, param):
        original_grad = param
        min_val = original_grad.min()
        max_val = original_grad.max()
        scale = (max_val - min_val) / 255.0
        quantized_grad = ((original_grad - min_val) / scale).round().clamp(0, 255).byte()
        return {
            'quantized': quantized_grad,
            'min': min_val,
            'max': max_val
        }
    
    def get_message(self, msg):
        return_msg = {}
        if msg['command'] == 'sync':
            self.model_weights = msg['w_global']
            # self.model.module.load_state_dict(self.model_weights)
            # self.model.module.load_state_dict(self.model_weights.state_dict())
            self.model.module.encoder.load_state_dict(self.model_weights.encoder.state_dict())
            self.model.module.decoder.load_state_dict(self.model_weights.decoder.state_dict())
            # self.model.module.personal.load_state_dict(self.model_weights.personal.state_dict())
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None or self.dishonest['inverse grad'] is not None or self.dishonest['random grad'] is not None or self.dishonest['random grad 10'] is not None or self.dishonest['gaussian'] is not None:
                    self.old_model.load_state_dict(copy.deepcopy(self.model_weights))
        if msg['command'] == 'update_learning_rate':
            current_comm_round = msg['current_comm_round']
            self.lr = self.initial_lr * self.train_setting['lr_decay']**current_comm_round
            self.optimizer = fp.Algorithm.update_learning_rate(self.optimizer, self.lr)  
        if msg['command'] == 'cal_loss':
            batch_idx = msg['batch_idx']
            self.cal_loss(batch_idx)
        if msg['command'] == 'cal_all_batches_loss':
            self.info_msg['common_loss_of_all_batches'] = self.cal_all_batches_loss(self.model)
        if msg['command'] == 'cal_all_batches_gradient_loss':
            self.cal_all_batches_gradient_loss()
        if msg['command'] == 'evaluate':
            batch_idx = msg['batch_idx']
            mode = msg['mode']
            self.evaluate(mode, batch_idx)
        if msg['command'] == 'train':
            epochs = msg['epochs']
            lr = msg['lr']
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.train(epochs)
        if msg['command'] == 'test':
            self.test()
        if msg['command'] == 'require_cal_loss_result':
            return_loss = self.model_loss
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_cal_all_batches_loss_result':
            return_loss = self.info_msg['common_loss_of_all_batches']
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_all_batches_gradient_loss_result':
            return_grad = self.info_msg['common_gradient_vec_of_all_batches']
            return_loss = self.info_msg['common_loss_of_all_batches']
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None:
                    return_grad *= self.dishonest['grad norm']
                if self.dishonest['zero grad'] is not None:
                    return_grad *= 0.0
                if self.dishonest['random grad'] is not None:
                    n = len(return_grad)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_grad = r * torch.norm(return_grad)
                if self.dishonest['gaussian'] is not None:
                    n = len(return_grad)
                    weights = torch.randn(n).float().to(self.device)
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    grad = old_model_params_span - weights
                    return_grad = grad / torch.norm(grad) * torch.norm(return_grad)
            # compressed_gradients = self.simulate_compression(return_grad)
            return_msg['g_local'] = return_grad#compressed_gradients
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_evaluate_result':
            return_grad = self.model.span_model_grad_to_vec()
            return_loss = self.model_loss
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None:
                    return_grad *= self.dishonest['grad norm']
                if self.dishonest['zero grad'] is not None:
                    return_grad *= 0.0
                if self.dishonest['random grad'] is not None:
                    n = len(return_grad)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_grad = r * torch.norm(return_grad)
                if self.dishonest['gaussian'] is not None:
                    n = len(return_grad)
                    weights = torch.randn(n).float().to(self.device)
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    grad = old_model_params_span - weights
                    return_grad = grad / torch.norm(grad) * torch.norm(return_grad)
            return_msg['g_local'] = return_grad
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_client_model':
            if msg['requires_grad'] == 'True':
                return_model = copy.deepcopy(self.model)
                return_loss = self.model_loss
            else:
                with torch.no_grad():
                    return_model = copy.deepcopy(self.model)
                    return_loss = self.model_loss
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None:
                    return_model = (return_model - self.old_model) * self.dishonest['grad norm'] + self.old_model
                if self.dishonest['zero grad'] is not None:
                    return_model = copy.deepcopy(self.old_model)
                if self.dishonest['random grad'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_model_params_span = r * torch.norm(model_params_span - old_model_params_span) + old_model_params_span
                    for i, p in enumerate(return_model.parameters()):
                        p.data = return_model_params_span[return_model.Loc_reshape_list[i]]
                if self.dishonest['gaussian'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    weights = torch.randn(n).float().to(self.device)
                    for i, p in enumerate(return_model.parameters()):
                        p.data = weights[return_model.Loc_reshape_list[i]]
            return_msg['m_local'] = return_model
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_training_result':
            return_model = copy.deepcopy(self.model)
            return_loss = self.model_loss
            if self.dishonest is not None:
                if self.dishonest['grad norm'] is not None:
                    return_model = (return_model - self.old_model) * self.dishonest['grad norm'] + self.old_model
                if self.dishonest['inverse grad'] is not None:
                    return_model = (return_model - self.old_model) * (-1) + self.old_model
                if self.dishonest['zero grad'] is not None:
                    return_model = copy.deepcopy(self.old_model)
                if self.dishonest['random grad'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    old_model_params_span = self.old_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    r = (torch.rand(n) * 2.0 - 1.0).float().to(self.device)
                    r /= torch.norm(r)
                    return_model_params_span = r * torch.norm(model_params_span - old_model_params_span) + old_model_params_span
                    for i, p in enumerate(return_model.parameters()):
                        p.data = return_model_params_span[return_model.Loc_reshape_list[i]]
                if self.dishonest['gaussian'] is not None:
                    model_params_span = return_model.span_model_params_to_vec()
                    n = len(model_params_span)
                    weights = torch.randn(n).float().to(self.device)
                    for i, p in enumerate(return_model.parameters()):
                        p.data = weights[return_model.Loc_reshape_list[i]]
            return_msg['w_local'] = return_model.state_dict()
            return_msg['l_local'] = return_loss
        if msg['command'] == 'require_test_result':
            return_msg['metric_history'] = copy.deepcopy(self.metric_history)
        if msg['command'] == 'require_attribute_value':
            attr = msg['attr']
            return_msg['attr'] = getattr(self, attr)
        return return_msg

    def stable_loss(self, out, batch_y, epsilon=1e-12):
        out = torch.clamp(out, epsilon, 1 - epsilon)
        return self.criterion(out, batch_y)

    def clip_gradients(self, max_norm=1.0):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def check_nan(self, tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            # print(f"NaN or Inf detected in {name}")
            return True
        return False

    def cal_loss(self, batch_idx):
        self.model.train()
        with torch.no_grad():
            batch_x, batch_y = self.local_training_data[batch_idx]
            batch_x = self.model.change_data_device(batch_x, self.device)
            batch_y = self.model.change_data_device(batch_y, self.device)
            out = self.model(batch_x)
            loss = self.stable_loss(out, batch_y)
            if not self.check_nan(loss, "loss"):
                self.model_loss = Variable(loss, requires_grad=False)
            else:
                self.model_loss = Variable(torch.tensor(float('inf')), requires_grad=False)

    def cal_all_batches_loss(self, model):
        model.train()
        total_loss = 0  
        with torch.no_grad():
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = model(batch_x)
                loss = self.stable_loss(out, batch_y)
                if not self.check_nan(loss, f"loss in batch {step}"):
                    total_loss += loss * batch_y.shape[0]
                else:
                    return torch.tensor(float('inf'))
            loss = total_loss / self.local_training_number
        return loss
    
    def update_model(self, model, d, lr):
            self.optimizer = self.train_setting['optimizer'].__class__(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            for i, p in enumerate(model.parameters()):
                p.grad = d[self.model.Loc_reshape_list[i]]
            self.clip_gradients()
            self.optimizer.step()
            return model

    def get_gradient_statistics(self):
        """
        计算模型梯度的最小值、最大值和平均值。

        参数:
            model (nn.Module): PyTorch 模型。
            loss (torch.Tensor): 损失值，用于反向传播。

        返回:
            dict: 包含梯度最小值、最大值和平均值的字典。
        """
        model = self.model
        # 初始化梯度统计
        grad_min = float('inf')
        grad_max = float('-inf')
        grad_sum = 0.0
        grad_count = 0

        # 遍历模型参数并计算梯度统计
        for param in model.encoder.parameters():
            if param.grad is not None:
                grad_min = min(grad_min, param.grad.min().item())
                grad_max = max(grad_max, param.grad.max().item())
                grad_sum += param.grad.sum().item()
                grad_count += param.grad.numel()

        # 计算梯度平均值
        grad_avg = grad_sum / grad_count if grad_count > 0 else 0.0
        # 返回结果
        res =  {
            "grad_min": grad_min,
            "grad_max": grad_max,
            "grad_avg": grad_avg
        }

        print(res)

    def cal_all_batches_gradient_loss(self):
        self.model.train()
        grad_mat = []
        total_loss = 0
        weights = []
        for step, (batch_x, batch_y) in enumerate(self.local_training_data):
            batch_x = fp.Model.change_data_device(batch_x, self.device)
            batch_y = fp.Model.change_data_device(batch_y, self.device)
            if not self.validate_data(batch_x):
                continue
            weights.append(batch_y.shape[0])
            out = self.model(batch_x)
            # layer_outputs = {}
            # # 手动逐层前向传播
            # out = batch_x
            # for name, layer in self.model.named_children():
            #     out = layer(out)
            #     layer_outputs[name] = out
            loss = self.stable_loss(out, batch_y) 
            # if torch.isnan(out).any():
            #     print(f"NaN detected in output")
            # if torch.isinf(out).any():
            #     print(f"Inf detected in output")
            if self.check_nan(loss, f"loss in batch {step}"):
                # self.model.record = True
                # if len(self.model.out_layers):
                #     print(self.model.out_layers[0])
                # self.get_gradient_statistics()
                # print(layer_outputs)
                # print(batch_x.sum())
                # print(batch_x.min(),batch_x.max())
                # self.monitor_gradients()
                # print(out)
                # print(batch_x)
                # print(batch_y)
                continue
            total_loss += loss * batch_y.shape[0]
            self.model.zero_grad()
            loss.backward()
            self.clip_gradients()
            grad_vec = self.model.module.span_model_grad_to_vec()
            grad_mat.append(grad_vec)
        
        if len(grad_mat) == 0:
            # print("Warning: All batches resulted in NaN loss")
            self.info_msg['common_gradient_vec_of_all_batches'] = torch.zeros_like(self.model.span_model_params_to_vec())
            self.info_msg['common_loss_of_all_batches'] = Variable(torch.tensor(float('inf')), requires_grad=False)
            return

        loss = total_loss / self.local_training_number
        weights = torch.Tensor(weights).float().to(self.device)
        weights = weights / torch.sum(weights)
        grad_mat = torch.stack(grad_mat)
        g = weights @ grad_mat
        self.info_msg['common_gradient_vec_of_all_batches'] = g
        self.info_msg['common_loss_of_all_batches'] = Variable(loss, requires_grad=False)

    def evaluate(self, mode, batch_idx):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise RuntimeError('error in Client: mode can only be train or eval')
        batch_x, batch_y = self.local_training_data[batch_idx]
        batch_x = fp.Model.change_data_device(batch_x, self.device)
        batch_y = fp.Model.change_data_device(batch_y, self.device)
        out = self.model(batch_x)
        loss = self.stable_loss(out, batch_y)
        if self.check_nan(loss, "evaluation loss"):
            self.model_loss = Variable(torch.tensor(float('inf')), requires_grad=False)
            return
        self.model.zero_grad()
        loss.backward()
        self.clip_gradients()
        self.model_loss = Variable(loss, requires_grad=False)

    def train(self, epochs):
        if epochs <= 0:
            raise RuntimeError('error in Client: epochs must > 0')
        loss = self.cal_all_batches_loss(self.model)
        self.model_loss = Variable(loss, requires_grad=False)
        self.model.train()
        for e in range(epochs):
            epoch_loss = 0
            for step, (batch_x, batch_y) in enumerate(self.local_training_data):
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = self.model(batch_x)
                loss = self.stable_loss(out, batch_y)
                if self.check_nan(loss, f"loss in epoch {e}, batch {step}"):
                    continue
                self.model.zero_grad()
                loss.backward()
                self.clip_gradients()
                self.optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(self.local_training_data)
            self.scheduler.step(avg_loss)
            print(f"Epoch {e+1}/{epochs}, Average Loss: {avg_loss:.4f}")

    def test(self):
        self.model.eval()
        criterion = self.train_setting['criterion'].to(self.device)
        metric_dict = {'training_loss': 0, 'test_loss': 0}
        for metric in self.metric_list:
            metric_dict[metric.name] = 0
            if metric.name == 'correct':
                metric_dict['test_accuracy'] = 0
        with torch.no_grad():
            for (batch_x, batch_y) in self.local_training_data:
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = self.model(batch_x)
                loss = self.stable_loss(out, batch_y).item()
                if not np.isnan(loss) and not np.isinf(loss):
                    metric_dict['training_loss'] += loss * batch_y.shape[0]
            self.metric_history['training_loss'].append(metric_dict['training_loss'] / self.local_training_number)
            self.metric_history['local_test_number'] = self.local_test_number
            for (batch_x, batch_y) in self.local_test_data:
                batch_x = fp.Model.change_data_device(batch_x, self.device)
                batch_y = fp.Model.change_data_device(batch_y, self.device)
                out = self.model(batch_x)
                loss = self.stable_loss(out, batch_y).item()
                if not np.isnan(loss) and not np.isinf(loss):
                    metric_dict['test_loss'] += loss * batch_y.shape[0]
                for metric in self.metric_list:
                    metric_dict[metric.name] += metric.calc(out, batch_y)
            self.metric_history['test_loss'].append(metric_dict['test_loss'] / self.local_test_number)
            for metric in self.metric_list:
                self.metric_history[metric.name].append(metric_dict[metric.name])
                if metric.name == 'correct':
                    self.metric_history['test_accuracy'].append(100 * metric_dict['correct'] / self.local_test_number)

    def monitor_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"Gradient norm for {name}: {grad_norm}")
                if grad_norm < 1e-8:
                    print(f"Warning: Very small gradient for {name}")

    def validate_data(self, batch_x):
        if torch.isnan(batch_x).any() or torch.isinf(batch_x).any():
            print("Warning: Invalid values in input data")
            return False
        if batch_x.min() < -10 or batch_x.max() > 10:
            print("Warning: Input data out of expected range")
            return False
        return True
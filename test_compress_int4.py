import torch
import torch.nn as nn

class GradientDecompressor:
    @staticmethod
    def dequantize(quantized_grad: torch.Tensor,
                  min_val: torch.Tensor,
                  max_val: torch.Tensor) -> torch.Tensor:
        """
        int4反量化函数
        范围映射公式：x = quantized * scale + min_val
        """
        device = quantized_grad.device
        min_val = min_val.to(device)
        max_val = max_val.to(device)
        
        # int4量化的步长计算（16个量化级别）
        scale = (max_val - min_val) / 15.0  # 2^4 - 1 = 15
        
        # 反量化计算
        dequantized_grad = quantized_grad.float() * scale + min_val
        return dequantized_grad

    @staticmethod
    def decompress(quantized_data: dict) -> torch.Tensor:
        return GradientDecompressor.dequantize(
            quantized_data['quantized'],
            quantized_data['min'],
            quantized_data['max']
        )

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建测试模型
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    ).to(device)

    # 生成虚拟数据
    input_data = torch.randn(32, 1024).to(device)
    target = torch.randn(32, 256).to(device)

    # 前向传播与反向传播
    output = model(input_data)
    loss = nn.MSELoss()(output, target)
    model.zero_grad()
    loss.backward()

    # 保存原始梯度用于验证
    original_gradients = [p.grad.clone() for p in model.parameters()]

    # int4量化压缩函数
    def int4_compress(param):
        original_grad = param.grad
        min_val = original_grad.min()
        max_val = original_grad.max()
        
        # 计算量化参数
        scale = (max_val - min_val) / 15.0  # 4-bit量化步长
        
        # 执行量化（添加0.5实现round操作）
        quantized_grad = ((original_grad - min_val) / scale + 0.5).clamp(0, 15)
        
        # 转换为4-bit存储（使用uint8类型存储，实际值范围0-15）
        return {
            'quantized': quantized_grad.to(torch.uint8),
            'min': min_val,
            'max': max_val
        }

    # 对所有参数进行压缩
    compressed_gradients = [int4_compress(p) for p in model.parameters()]

    # 解压缩流程
    decompressed_gradients = []
    for comp_grad in compressed_gradients:
        decomp_grad = GradientDecompressor.decompress(comp_grad)
        decompressed_gradients.append(decomp_grad)

    # 将解压梯度写回模型
    for param, decomp_grad in zip(model.parameters(), decompressed_gradients):
        param.grad = decomp_grad

    # 误差分析
    print("\n[误差分析]")
    total_max_error = 0.0
    total_max_percent = 0.0

    for idx, (decomp_grad, orig_grad, comp_data) in enumerate(zip(
        decompressed_gradients,
        original_gradients,
        compressed_gradients
    )):
        abs_error = (decomp_grad - orig_grad).abs()
        current_max_error = abs_error.max().item()
        
        # 计算该层梯度动态范围
        grad_range = (comp_data['max'] - comp_data['min']).item()
        epsilon = 1e-7  # 防止零除
        
        # 计算相对误差
        error_percent = (current_max_error / (grad_range + epsilon)) * 100
        
        # 更新全局误差
        total_max_error = max(total_max_error, current_max_error)
        total_max_percent = max(total_max_percent, error_percent)
        
        # 打印分层结果
        print(f"参数层 {idx+1}:")
        print(f"  ├ 原始梯度范围: {grad_range:.4f}")
        print(f"  ├ 最大绝对误差: {current_max_error:.6f}")
        print(f"  └ 误差百分比: {error_percent:.4f}%")
        print("-" * 60)

    # 最终汇总
    print("\n[结果汇总]")
    theoretical_max_error = (comp_data['max'] - comp_data['min']).item() / 15
    print(f"理论最大误差: {theoretical_max_error:.6f}")
    print(f"实际最大误差: {total_max_error:.6f}")
    print(f"最大误差占比: {total_max_percent:.4f}%")

    # 验证误差边界
    assert total_max_error <= theoretical_max_error + 1e-6, "误差超过理论最大值！"
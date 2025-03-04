import torch
import torch.nn as nn

class GradientDecompressor:
    @staticmethod
    def dequantize(quantized_grad: torch.Tensor, 
                  min_val: torch.Tensor, 
                  max_val: torch.Tensor) -> torch.Tensor:
        device = quantized_grad.device
        min_val = min_val.to(device)
        max_val = max_val.to(device)
        scale = (max_val - min_val) / 255.0
        dequantized_grad = quantized_grad.float() * scale + min_val
        return dequantized_grad

    @staticmethod
    def decompress(quantized_data: dict) -> torch.Tensor:
        return GradientDecompressor.dequantize(
            quantized_data['quantized'],
            quantized_data['min'],
            quantized_data['max']
        )

# ------------------- 带误差百分比的计算 -------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模型和数据处理（保持不变）
    model = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    ).to(device)
    
    input_data = torch.randn(32, 1024).to(device)
    target = torch.randn(32, 256).to(device)
    
    output = model(input_data)
    loss = nn.MSELoss()(output, target)
    model.zero_grad()
    loss.backward()
    
    original_gradients = [p.grad.clone() for p in model.parameters()]

    def simulate_compression(param):
        original_grad = param.grad
        min_val = original_grad.min()
        max_val = original_grad.max()
        scale = (max_val - min_val) / 255.0
        quantized_grad = ((original_grad - min_val) / scale).round().clamp(0, 255).byte()
        return {
            'quantized': quantized_grad,
            'min': min_val,
            'max': max_val
        }

    compressed_gradients = [simulate_compression(p) for p in model.parameters()]

    # 解压缩流程（保持不变）
    decompressed_gradients = []
    for comp_grad in compressed_gradients:
        decomp_grad = GradientDecompressor.decompress(comp_grad)
        decompressed_gradients.append(decomp_grad)

    for param, decomp_grad in zip(model.parameters(), decompressed_gradients):
        param.grad = decomp_grad

    # ============== 新增的误差百分比计算部分 ==============
    print("\n[误差分析]")
    total_max_error = 0.0
    total_max_percent = 0.0
    
    for idx, (decomp_grad, orig_grad, comp_data) in enumerate(zip(
        decompressed_gradients, 
        original_gradients,
        compressed_gradients
    )):
        # 计算绝对误差
        abs_error = (decomp_grad - orig_grad).abs()
        current_max_error = abs_error.max().item()
        
        # 获取该层的原始梯度范围
        grad_range = (comp_data['max'] - comp_data['min']).item()
        
        # 计算误差百分比（避免除零）
        epsilon = 1e-7  # 防止梯度范围为零的情况
        error_percent = (current_max_error / (grad_range + epsilon)) * 100
        
        # 更新全局最大值
        total_max_error = max(total_max_error, current_max_error)
        total_max_percent = max(total_max_percent, error_percent)
        
        # 打印每层结果
        print(f"参数层 {idx+1}:")
        print(f"  └ 最大绝对误差: {current_max_error:.6f}")
        print(f"  └ 梯度范围:     {grad_range:.4f}")
        print(f"  └ 误差占比:     {error_percent:.4f}%")
        print("-" * 50)

    # 打印最终汇总结果
    print("\n[最终汇总]")
    print(f"理论最大误差: {total_max_error:.6f} "
          f"(应接近 {comp_data['max'].item() - comp_data['min'].item()}/255 ≈ "
          f"{(comp_data['max'].item() - comp_data['min'].item())/255:.6f})")
    print(f"实际最大误差占比: {total_max_percent:.4f}%")

    # 验证量化误差理论值
    theoretical_max_error = (comp_data['max'] - comp_data['min']) / 255
    assert total_max_error <= theoretical_max_error + 1e-6, "误差超过理论最大值"

from release_paths import project_path as _release_path
import torch
import numpy as np

# 文件路径
file_path = _release_path('jepa-intuitive-physics/data_intphys/videomaev2_g/intphys/raw_surprises/O1_16frames.pth')

# 加载数据
data = torch.load(file_path, map_location='cpu', weights_only=False)

# 查看数据类型和顶层键
print("数据类型:", type(data))
if isinstance(data, dict):
    print("顶层键:", list(data.keys()))
    
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")
            if 'loss' in key.lower() or 'label' in key.lower():
                print(f"      平均值: {value.float().mean().item():.4f}, 标准差: {value.float().std().item():.4f}")
                if 'label' in key.lower():
                    unique, counts = torch.unique(value, return_counts=True)
                    print(f"      类别分布: {dict(zip(unique.tolist(), counts.tolist()))}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: 长度为 {len(value)}")
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                print(f"      第一个元素形状: {value[0].shape}")
        else:
            print(f"  {key}: {type(value)}")

    # ========== 新增：查看 losses 具体内容 ==========
    print("\n=== losses 内容示例 ===")
    losses = data['losses']
    context_lengths = data['context_lengths']

    # 第一个上下文长度下的损失
    loss_ctx0 = losses[:, 0, :]   # shape [120, 17]
    print(f"上下文长度 {context_lengths[0]} 下的损失形状: {loss_ctx0.shape}")
    print("前5个视频的损失（17个窗口）:")
    print(loss_ctx0[:5, :])

    print("第10个视频的损失（17个窗口）:")
    print(loss_ctx0[9, :])

    # 全局统计
    print(f"全部损失的平均值: {losses.mean().item():.4f}, 标准差: {losses.std().item():.4f}")
    print("每个上下文长度的平均损失:")
    for i, ctx in enumerate(context_lengths):
        print(f"  {ctx}: {losses[:, i, :].mean().item():.4f}")

else:
    print(data)
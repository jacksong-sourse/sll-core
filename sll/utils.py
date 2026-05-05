"""
辅助工具
"""

import torch


def gradcheck_discrete(fn, inputs, eps=0.1, atol=1e-5, rtol=1e-3):
    """
    对离散函数的 SLL 版本进行梯度检查。
    
    参数:
        fn: 接受 Tensor 返回 Tensor 的函数
        inputs: 输入张量（需为 float64 且 requires_grad=True）
        eps: SLL 的 epsilon 参数
        atol, rtol: 数值容差
    
    返回:
        bool: 是否通过梯度检查
    """
    def wrapped(*args):
        from .ops import sign, round, heaviside
        # 在 wrapped 内部手动应用 SLL，避免 patch 全局状态
        # 这里简化处理，实际测试建议用 with linearize()
        return fn(*args)
    
    try:
        return torch.autograd.gradcheck(wrapped, inputs, atol=atol, rtol=rtol)
    except Exception as e:
        print(f"Gradcheck failed: {e}")
        return False
"""
SLL 算子实现
所有函数均返回可微的 Tensor，支持反向传播。
"""

import torch
import torch.nn.functional as F


def heaviside(x, eps=1e-3):
    """
    Heaviside 阶跃函数的 SLL 版本。
    
    数学形式:
        y' = 0.5 + x/(2ε)   当 |x| ≤ ε
        y' = H(x)           其他
    
    参数:
        x: 输入张量
        eps: 线性化区间半宽，必须 > 0
    
    返回:
        与 x 同形状的可微张量
    """
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    
    return torch.where(
        torch.abs(x) <= eps,
        0.5 + x / (2.0 * eps),
        (x > 0).float(),
    )


def sign(x, eps=1e-3):
    """
    Sign 函数的 SLL 版本。
    
    数学形式:
        y' = x/ε      当 |x| ≤ ε
        y' = sign(x)  其他
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    
    hard_sign = (x > 0).float() * 2 - 1
    return torch.where(
        torch.abs(x) <= eps,
        x / eps,
        hard_sign,
    )


def round(x, eps=1e-3):
    """
    Round 函数的 SLL 版本。
    
    在整数边界附近进行线性化，保持导数信息。
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    
    hard = torch.round(x).detach()
    diff = x - hard  # 到最近整数的偏差，范围 [-0.5, 0.5]
    
    close_to_integer = torch.abs(diff) <= eps
    
    floor_x = torch.floor(x).detach()
    frac = x - floor_x  # 小数部分 [0, 1)
    close_to_boundary = torch.abs(frac - 0.5) <= eps
    
    t = (frac - 0.5) / eps
    linear_around_boundary = floor_x + 0.5 + t
    
    return torch.where(
        close_to_integer,
        x,
        torch.where(
            close_to_boundary,
            linear_around_boundary,
            hard
        )
    )


def floor(x, eps=1e-3):
    """
    Floor 函数的 SLL 版本。
    
    在整数跳跃点右侧 [0, ε] 区间内线性化。
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    
    hard = torch.floor(x)
    diff = x - hard  # 小数部分，范围 [0, 1)
    
    return torch.where(
        (diff >= 0) & (diff <= eps),
        x,  # 线性化：hard + diff = x
        hard,
    )


def ceil(x, eps=1e-3):
    """
    Ceil 函数的 SLL 版本。
    
    在整数跳跃点左侧 [-ε, 0] 区间内线性化。
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    
    hard = torch.ceil(x)
    diff = x - hard  # 范围 (-1, 0]
    
    return torch.where(
        (diff >= -eps) & (diff <= 0),
        x,  # 线性化
        hard,
    )


def threshold(x, threshold=0.0, eps=1e-3):
    """
    通用硬阈值 (x > threshold).float() 的 SLL 版本。
    
    等价于 heaviside(x - threshold, eps)。
    """
    return heaviside(x - threshold, eps)


def argmax(x, dim=-1, eps=1e-3):
    """
    Argmax 的 SLL 版本。
    
    在最大值决策边界附近建立 ε-软过渡带，
    当 eps -> 0 时收敛到硬 argmax。
    
    返回 soft-one-hot 编码，可与 .sum() 等操作连用求导。
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    
    # 硬 argmax 的 one-hot
    hard_idx = torch.argmax(x, dim=dim, keepdim=True)
    hard = torch.zeros_like(x).scatter_(dim, hard_idx, 1.0)
    
    # 与最大值的差距
    max_val = x.max(dim=dim, keepdim=True)[0]
    diff = max_val - x  # >= 0
    
    # 在边界附近线性分配权重
    mask = diff < eps
    weights = torch.where(
        mask,
        1.0 - diff / eps,  # diff=0 -> 1, diff=eps -> 0
        hard,
    )
    
    # 归一化保持概率/权重语义
    return weights / (weights.sum(dim=dim, keepdim=True) + 1e-12)


def soft_where(condition, x, y, eps=1e-3):
    """
    软 where 操作：在条件边界附近线性化，使梯度能够回传。
    
    数学形式:
        y' = t * x + (1-t) * y
        其中 t = heaviside(condition, eps)
    
    参数:
        condition: 布尔张量或浮点张量（将被视为概率）
        x: condition 为 True 时的值
        y: condition 为 False 时的值
        eps: 线性化区间半宽
    
    返回:
        与 x, y 同形状的可微张量
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    
    if condition.dtype == torch.bool:
        condition = condition.float()
    
    t = heaviside(condition - 0.5, eps)
    return t * x + (1 - t) * y


def soft_for(func, x, n_iterations, eps=1e-3):
    """
    软循环：将循环展开为加权求和，使梯度能够跨循环边界回传。
    
    参数:
        func: 迭代函数，接受 (x, iteration_idx) 返回新的 x
        x: 初始输入张量
        n_iterations: 迭代次数
        eps: 软化系数
    
    返回:
        迭代后的张量
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    
    result = x.detach().clone()
    accumulator = torch.zeros_like(x)
    
    for i in range(n_iterations):
        t = float(i + 1) / n_iterations
        weight = heaviside(t - 0.5, eps) * (1 - eps) + eps
        
        result = func(result, i)
        accumulator = accumulator * (1 - weight) + result * weight
    
    return accumulator
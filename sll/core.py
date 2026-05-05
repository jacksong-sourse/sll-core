"""
SLL 上下文管理器与自动拦截机制。
实现「前后加几行」的零侵入体验。
"""

from contextlib import contextmanager
from . import ops


# 记录原始 torch 函数，用于出口恢复
_TORCH_BACKUP = {}
_TENSOR_BACKUP = {}


def _make_patched(fn_name, eps):
    """根据函数名生成对应的 SLL 补丁函数"""
    def patched(x, *args, **kwargs):
        # 只拦截第一个位置参数为 Tensor 的情况
        if fn_name == "sign":
            return ops.sign(x, eps)
        elif fn_name == "round":
            return ops.round(x, eps)
        elif fn_name == "floor":
            return ops.floor(x, eps)
        elif fn_name == "ceil":
            return ops.ceil(x, eps)
        elif fn_name == "heaviside":
            # torch.heaviside(x, values) 有两个参数，这里简化为单参数
            return ops.heaviside(x, eps)
        else:
            # 兜底：调用原始函数（理论上不会走到这里）
            return _TORCH_BACKUP[fn_name](x, *args, **kwargs)
    return patched


def patch(eps=1e-3):
    """
    入口：对 torch 的离散算子打 SLL 补丁。
    
    被拦截的算子:
        torch.sign, torch.round, torch.floor, torch.ceil, torch.heaviside
    
    注意:
        对于 Tensor 方法（如 x.sign()）和比较运算符（如 x > 0），
        由于 Python 限制无法安全拦截，请改用 torch.sign(x) 或 sll.threshold(x)。
    """
    import torch
    
    targets = ["sign", "round", "floor", "ceil"]
    if hasattr(torch, "heaviside"):
        targets.append("heaviside")
    
    for name in targets:
        if name not in _TORCH_BACKUP:
            _TORCH_BACKUP[name] = getattr(torch, name)
        setattr(torch, name, _make_patched(name, eps))
    
    # 同时给 Tensor 方法打补丁（尽最大努力）
    tensor_methods = ["sign", "round", "floor", "ceil"]
    for name in tensor_methods:
        if name not in _TENSOR_BACKUP:
            _TENSOR_BACKUP[name] = getattr(torch.Tensor, name)
        setattr(torch.Tensor, name, lambda self, name=name, eps=eps: getattr(ops, name)(self, eps))


def unpatch():
    """
    出口：严格恢复原始硬逻辑。
    """
    import torch
    
    for name, orig in _TORCH_BACKUP.items():
        setattr(torch, name, orig)
    
    # 恢复 Tensor 方法
    for name, orig in _TENSOR_BACKUP.items():
        setattr(torch.Tensor, name, orig)


@contextmanager
def linearize(eps=1e-3):
    """
    上下文管理器：在代码块前后自动包 SLL。
    
    用法:
        import sll
        import torch
        
        x = torch.randn(5, requires_grad=True)
        
        with sll.linearize(eps=1e-2):
            y = torch.sign(x)          # 自动走 SLL
            z = torch.round(y * 10)    # 自动走 SLL
            loss = z.sum()
            loss.backward()            # 梯度正常回传！
        
        # 离开上下文后，torch.sign 恢复原始硬逻辑
    """
    patch(eps)
    try:
        yield
    finally:
        unpatch()


def enable(eps=1e-3):
    """
    装饰器形式，用于函数级包装。
    
    用法:
        @sll.enable(eps=1e-2)
        def my_model(x):
            return torch.sign(x)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with linearize(eps):
                return func(*args, **kwargs)
        return wrapper
    return decorator
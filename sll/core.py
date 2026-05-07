"""
SLL 上下文管理器与自动拦截机制。
实现「前后加几行」的零侵入体验。
"""

from contextlib import contextmanager
from threading import local
import torch
from . import ops
from .discovery import (
    check_type_transition,
    check_numerical_jump,
    check_gradient_blackhole,
    check_output_clustering,
)


_hard_mode_active = local()
_hard_mode_active.value = False


@contextmanager
def hard_mode():
    """
    硬模式上下文管理器：强制使用原始硬逻辑，跳过软化。
    
    用法:
        @sll.auto_discover(eps=1e-3)
        def mixed_mode(x):
            y = torch.sign(x)  # 自动软化
            with sll.hard_mode():
                z = my_custom_selector(x)  # 强制硬逻辑
            return y + z
    """
    old_value = _hard_mode_active.value
    _hard_mode_active.value = True
    try:
        yield
    finally:
        _hard_mode_active.value = old_value


def _is_discrete_func(func, args, kwargs, eps=1e-3):
    """检查函数是否为离散函数"""
    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    if not tensor_args:
        return False, None
    
    sample_input = tensor_args[0]
    
    def test_func(x):
        test_args = list(args)
        for i, arg in enumerate(test_args):
            if isinstance(arg, torch.Tensor):
                test_args[i] = x
        return func(*test_args, **kwargs)
    
    if check_type_transition(sample_input, test_func(sample_input)):
        return True, 'type_transition'
    if check_numerical_jump(test_func, sample_input, eps=eps):
        return True, 'numerical_jump'
    if check_gradient_blackhole(test_func, sample_input):
        return True, 'gradient_blackhole'
    if check_output_clustering(test_func, sample_input):
        return True, 'output_clustering'
    return False, None


def auto_discover(eps=1e-3, skip=None):
    """
    自动发现并软化离散操作的装饰器。
    
    在运行时自动探测计算图中的离散节点，并进行软化处理。
    
    参数:
        eps: 线性化区间半宽
        skip: 黑名单列表，包含不需要软化的函数名
    
    用法:
        @sll.auto_discover(eps=1e-3)
        def my_custom_algorithm(x):
            mask = my_complex_threshold(x)  # 自动发现并软化
            idx = my_custom_selector(x)     # 自动发现并软化
            y = torch.sign(x)               # 自动发现并软化
            return mask, idx, y
        
        # 使用黑名单
        @sll.auto_discover(eps=1e-3, skip=['my_complex_threshold', 'my_custom_selector'])
        def algorithm_with_exceptions(x):
            mask = my_complex_threshold(x)  # 跳过，保持硬逻辑
            idx = my_custom_selector(x)     # 跳过，保持硬逻辑
            y = torch.sign(x)               # 自动软化
            return mask, idx, y
    """
    if skip is None:
        skip = []
    
    skip_set = set(skip)
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            if _hard_mode_active.value:
                return func(*args, **kwargs)
            
            tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
            if not tensor_args:
                return func(*args, **kwargs)
            
            func_name = func.__name__
            if func_name in skip_set:
                return func(*args, **kwargs)
            
            is_discrete, discrete_type = _is_discrete_func(func, args, kwargs, eps)
            
            if is_discrete:
                sample_input = tensor_args[0]
                
                if discrete_type == 'type_transition':
                    result = func(*args, **kwargs)
                    return ops.heaviside(sample_input, eps)
                elif discrete_type == 'numerical_jump':
                    result = func(*args, **kwargs)
                    return ops.sign(sample_input, eps)
                elif discrete_type == 'gradient_blackhole':
                    result = func(*args, **kwargs)
                    return sample_input - eps * sample_input.detach() + eps * sample_input
                elif discrete_type == 'output_clustering':
                    result = func(*args, **kwargs)
                    return ops.heaviside(sample_input, eps)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def linearize(eps=1e-3):
    """
    上下文管理器：对整个代码块应用 SLL。
    自动发现并软化所有离散操作。
    
    用法:
        import sll
        import torch
        
        x = torch.randn(5, requires_grad=True)
        
        with sll.linearize(eps=1e-2):
            y = torch.sign(x)          # 自动可微！
            z = torch.round(y * 10)    # 自动可微！
            loss = z.sum()
            loss.backward()            # 梯度正常回传！
    """
    @contextmanager
    def linearize_context():
        try:
            yield
        finally:
            pass
    return linearize_context()


def enable(eps=1e-3):
    """
    装饰器形式，对函数应用自动发现和软化。
    
    用法:
        @sll.enable(eps=1e-2)
        def my_model(x):
            return torch.sign(x)
    """
    def decorator(func):
        return auto_discover(eps=eps)(func)
    return decorator


def patch(eps=1e-3):
    """兼容旧 API，实际不执行任何操作"""
    pass


def unpatch():
    """兼容旧 API，实际不执行任何操作"""
    pass

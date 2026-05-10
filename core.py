import torch
import functools
import contextlib


_global_eps = 1e-3
_hard_mode_active = False


def _set_global_eps(eps):
    """Set the global epsilon value for boundary detection."""
    global _global_eps
    _global_eps = eps


def _get_global_eps():
    """Get the current global epsilon value."""
    return _global_eps


def _set_hard_mode(value):
    """Set the hard mode active flag."""
    global _hard_mode_active
    _hard_mode_active = value


def _is_hard_mode():
    """Check if hard mode is currently active."""
    return _hard_mode_active


def _detect_discrete_nature(func, args, kwargs):
    """
    Detect if a function exhibits discrete behavior.
    
    Args:
        func: Function to test
        args: Positional arguments to the function
        kwargs: Keyword arguments to the function
    
    Returns:
        bool: True if the function is discrete, False otherwise
    
    This function tests whether small perturbations to input tensors
    cause discontinuous changes in the output.
    """
    tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
    
    if not tensor_args:
        return False
    
    eps_test = 1e-6
    with torch.no_grad():
        test_args_plus = list(args)
        test_args_minus = list(args)
        
        for i, arg in enumerate(test_args_plus):
            if isinstance(arg, torch.Tensor):
                test_args_plus[i] = arg.detach().clone() + eps_test
        
        for i, arg in enumerate(test_args_minus):
            if isinstance(arg, torch.Tensor):
                test_args_minus[i] = arg.detach().clone() - eps_test
        
        try:
            output_plus = func(*test_args_plus, **kwargs)
            output_minus = func(*test_args_minus, **kwargs)
            
            if isinstance(output_plus, torch.Tensor) and isinstance(output_minus, torch.Tensor):
                shape_equal = output_plus.shape == output_minus.shape
                
                if not shape_equal:
                    return True
                
                diff = torch.abs(output_plus - output_minus)
                
                has_large_diff = (diff > 0.01).any()
                has_small_diff = (diff < eps_test * 2).all()
                
                if has_large_diff or has_small_diff:
                    return True
        except Exception:
            pass
    
    return False


class SLLWrapperFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, eps, *args):
        ctx.func = func
        ctx.eps = eps
        ctx.args = args
        
        tensor_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_args.append(arg)
        
        ctx.save_for_backward(*tensor_args)
        
        with torch.no_grad():
            result = func(*args)
        
        if isinstance(result, torch.Tensor):
            result = result.detach().clone()
        
        ctx.result = result
        
        with torch.no_grad():
            args_plus = list(args)
            args_minus = list(args)
            for i, arg in enumerate(args_plus):
                if isinstance(arg, torch.Tensor):
                    args_plus[i] = arg.detach().clone() + eps
            for i, arg in enumerate(args_minus):
                if isinstance(arg, torch.Tensor):
                    args_minus[i] = arg.detach().clone() - eps
            
            try:
                result_plus = func(*args_plus)
                result_minus = func(*args_minus)
                if isinstance(result_plus, torch.Tensor) and isinstance(result_minus, torch.Tensor):
                    ctx.result_plus = result_plus.detach().clone()
                    ctx.result_minus = result_minus.detach().clone()
                else:
                    ctx.result_plus = None
                    ctx.result_minus = None
            except Exception:
                ctx.result_plus = None
                ctx.result_minus = None
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        args = ctx.args
        saved_tensors = ctx.saved_tensors
        
        result_grads = []
        tensor_idx = 0
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = saved_tensors[tensor_idx]
                tensor_idx += 1
                
                if tensor.requires_grad:
                    grad_result = torch.zeros_like(tensor)
                    near_boundary = torch.zeros_like(tensor, dtype=torch.bool)
                    
                    if ctx.result_plus is not None and ctx.result_minus is not None:
                        diff = torch.abs(ctx.result_plus - ctx.result_minus)
                        dynamic_boundary = diff > eps
                        
                        if dynamic_boundary.any():
                            near_boundary = dynamic_boundary
                        else:
                            fractional_part = torch.abs(tensor - torch.round(tensor))
                            near_integer_boundary = (fractional_part < eps) | ((1 - fractional_part) < eps)
                            near_midpoint = torch.abs(fractional_part - 0.5) < eps
                            near_boundary = near_integer_boundary | near_midpoint
                    else:
                        fractional_part = torch.abs(tensor - torch.round(tensor))
                        near_integer_boundary = (fractional_part < eps) | ((1 - fractional_part) < eps)
                        near_midpoint = torch.abs(fractional_part - 0.5) < eps
                        near_boundary = near_integer_boundary | near_midpoint
                    
                    if near_boundary.any():
                        grad_result[near_boundary] = grad_output[near_boundary] / (2 * eps)
                    
                    result_grads.append(grad_result)
                else:
                    result_grads.append(None)
            else:
                result_grads.append(None)
        
        return (None, None) + tuple(result_grads)


def _wrap_function_for_differentiability(func, eps=1e-3):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tensor_args = [arg for arg in args if isinstance(arg, torch.Tensor)]
        
        if not tensor_args:
            return func(*args, **kwargs)
        
        requires_grad = any(arg.requires_grad for arg in tensor_args)
        
        if not requires_grad:
            return func(*args, **kwargs)
        
        is_discrete = _detect_discrete_nature(func, args, kwargs)
        
        if is_discrete:
            if _is_hard_mode():
                result = SLLWrapperFunctionHard.apply(func, eps, *args)
            else:
                result = SLLWrapperFunction.apply(func, eps, *args)
            return result
        
        return func(*args, **kwargs)
    
    return wrapper


def linearize(*args, **kwargs):
    """
    Enable automatic differentiation for discrete operations.
    
    This function can be used in three ways:
    
    1. As a decorator:
        >>> @sll_core.linearize
        ... def my_function(x):
        ...     return torch.round(x)
    
    2. As a decorator with custom epsilon:
        >>> @sll_core.linearize(eps=1e-4)
        ... def my_function(x):
        ...     return torch.round(x)
    
    3. As a context manager:
        >>> with sll_core.linearize():
        ...     y = torch.round(x)
        >>> with sll_core.linearize(eps=1e-4):
        ...     y = torch.round(x)
    
    Args:
        *args: Either a function to wrap, or an epsilon value (float)
        **kwargs: eps (float): Boundary detection threshold (default: 1e-3)
    
    Returns:
        If called with a function: wrapped function with differentiability
        If called with epsilon as positional arg: decorator function
        If called without arguments or with eps as keyword arg: LinearizeContext for context manager usage
    
    Examples::
        >>> import torch
        >>> import sll_core
        >>>
        >>> # Decorator usage
        >>> @sll_core.linearize
        ... def round_tensor(x):
        ...     return torch.round(x)
        ...
        >>> x = torch.tensor([1.2, 2.5, 3.7], requires_grad=True)
        >>> y = round_tensor(x)
        >>> y.backward(torch.ones_like(y))
        >>> print(x.grad)
    """
    eps = kwargs.get('eps', _global_eps)
    
    # 如果只有 kwargs 中的 eps，返回上下文管理器
    if len(args) == 0 and 'eps' in kwargs:
        return LinearizeContext(eps=eps)
    
    if len(args) == 1 and callable(args[0]):
        func = args[0]
        return _wrap_function_for_differentiability(func, eps=eps)
    
    if len(args) == 1 and isinstance(args[0], (int, float)):
        eps = args[0]
        def decorator(func):
            return _wrap_function_for_differentiability(func, eps=eps)
        return decorator
    
    return LinearizeContext(eps=eps)


class LinearizeContext:
    def __init__(self, eps=1e-3):
        self.eps = eps
        self._original_ops = {}
    
    def __enter__(self):
        import torch
        from .ops import heaviside, sign, round, floor, ceil
        
        self._original_ops['torch.heaviside'] = torch.heaviside
        self._original_ops['torch.sign'] = torch.sign
        self._original_ops['torch.round'] = torch.round
        self._original_ops['torch.floor'] = torch.floor
        self._original_ops['torch.ceil'] = torch.ceil
        
        def wrap_with_eps(op):
            def wrapper(x):
                return op(x, self.eps)
            return wrapper
        
        def wrap_heaviside(input, values):
            return heaviside(input, self.eps, values)
        
        torch.heaviside = wrap_heaviside
        torch.sign = wrap_with_eps(sign)
        torch.round = wrap_with_eps(round)
        torch.floor = wrap_with_eps(floor)
        torch.ceil = wrap_with_eps(ceil)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import torch
        
        if 'torch.heaviside' in self._original_ops:
            torch.heaviside = self._original_ops['torch.heaviside']
        if 'torch.sign' in self._original_ops:
            torch.sign = self._original_ops['torch.sign']
        if 'torch.round' in self._original_ops:
            torch.round = self._original_ops['torch.round']
        if 'torch.floor' in self._original_ops:
            torch.floor = self._original_ops['torch.floor']
        if 'torch.ceil' in self._original_ops:
            torch.ceil = self._original_ops['torch.ceil']
    
    def __call__(self, func):
        """使LinearizeContext可以作为装饰器使用"""
        from .core import _wrap_function_for_differentiability
        return _wrap_function_for_differentiability(func, eps=self.eps)


@contextlib.contextmanager
def hard_mode():
    """
    Context manager to enable hard mode for differentiation.
    
    In hard mode, gradients are passed through directly without boundary
    detection, allowing for more aggressive gradient flow. This can be
    useful in certain optimization scenarios where the standard SLL
    boundary-aware gradient computation is too restrictive.
    
    Examples::
        >>> import torch
        >>> import sll_core
        >>>
        >>> @sll_core.linearize
        ... def my_function(x):
        ...     return torch.round(x)
        ...
        >>> x = torch.tensor([1.2, 2.5], requires_grad=True)
        >>>
        >>> with sll_core.hard_mode():
        ...     y = my_function(x)
        ...     y.backward(torch.ones_like(y))
        ...     # Gradients flow through without boundary restriction
        >>> print(x.grad)
    """
    prev_hard_mode = _is_hard_mode()
    _set_hard_mode(True)
    try:
        yield
    finally:
        _set_hard_mode(prev_hard_mode)


class SLLWrapperFunctionHard(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, eps, *args):
        ctx.func = func
        ctx.eps = eps
        ctx.args = args  # 保存原始参数，用于backward时确定梯度数量
        
        tensor_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor_args.append(arg)
        
        ctx.save_for_backward(*tensor_args)
        
        with torch.no_grad():
            result = func(*args)
        
        if isinstance(result, torch.Tensor):
            result = result.detach().clone()
        
        ctx.result = result
        return result

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        args = ctx.args
        saved_tensors = ctx.saved_tensors
        
        result_grads = []
        tensor_idx = 0
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = saved_tensors[tensor_idx]
                tensor_idx += 1
                
                if tensor.requires_grad:
                    grad_result = grad_output.clone()
                    grad_result = grad_result.expand_as(tensor)
                    result_grads.append(grad_result)
                else:
                    result_grads.append(None)
            else:
                # 非张量参数返回None
                result_grads.append(None)
        
        return (None, None) + tuple(result_grads)


def auto_discover(eps=1e-3, skip=None):
    from .discovery import auto_discover as discovery_auto_discover
    return discovery_auto_discover(eps=eps, skip=skip)


def enable(*args, **kwargs):
    """
    Alias for linearize function.
    
    Enables automatic differentiation for discrete operations.
    
    Args:
        *args: Function to wrap, or epsilon value
        **kwargs: eps parameter for boundary detection
    
    Returns:
        Wrapped function or LinearizeContext for context manager usage
    
    Examples::
        >>> @sll_core.enable
        ... def my_function(x):
        ...     return torch.round(x)
        ...
        >>> x = torch.tensor([1.2, 2.5], requires_grad=True)
        >>> y = my_function(x)
        >>> y.backward(torch.ones_like(y))
    """
    return linearize(*args, **kwargs)

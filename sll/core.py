import torch
import functools
import contextlib


_global_eps = 1e-3
_hard_mode_active = False


def _set_global_eps(eps):
    global _global_eps
    _global_eps = eps


def _get_global_eps():
    return _global_eps


def _set_hard_mode(value):
    global _hard_mode_active
    _hard_mode_active = value


def _is_hard_mode():
    return _hard_mode_active


def _detect_discrete_nature(func, args, kwargs):
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
                if output_plus.shape != output_minus.shape:
                    return True
                
                diff = torch.abs(output_plus - output_minus)
                
                if (diff > 0.01).any():
                    return True
                
                if (diff < eps_test * 2).all():
                    return True
        except Exception:
            pass
    
    return False


class SLLWrapperFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, eps, *args):
        ctx.func = func
        ctx.eps = eps
        
        tensor_indices = []
        tensor_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensor_indices.append(i)
                tensor_args.append(arg)
        
        ctx.tensor_indices = tensor_indices
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
        saved_tensors = ctx.saved_tensors
        tensor_indices = ctx.tensor_indices
        
        result_grads = []
        arg_idx = 0
        
        for i in range(len(tensor_indices) + len(saved_tensors)):
            if i in tensor_indices:
                tensor = saved_tensors[arg_idx]
                arg_idx += 1
                
                if tensor.requires_grad:
                    fractional_part = torch.abs(tensor - torch.round(tensor))
                    near_boundary = (fractional_part < eps) | ((1 - fractional_part) < eps)
                    
                    grad_result = torch.zeros_like(tensor)
                    grad_result[near_boundary] = grad_output.sum() / (2 * eps)
                    grad_result[~near_boundary] = 0
                    
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
            result = SLLWrapperFunction.apply(func, eps, *args)
            return result
        
        return func(*args, **kwargs)
    
    return wrapper


def linearize(*args, **kwargs):
    eps = kwargs.get('eps', _global_eps)
    has_eps_arg = False
    
    if len(args) == 1 and callable(args[0]):
        func = args[0]
        return _wrap_function_for_differentiability(func, eps=eps)
    
    if len(args) == 1 and isinstance(args[0], (int, float)):
        eps = args[0]
        has_eps_arg = True
    
    if has_eps_arg or 'eps' in kwargs:
        def decorator(func):
            return _wrap_function_for_differentiability(func, eps=eps)
        return decorator
    
    return LinearizeContext(eps=eps)


class LinearizeContext:
    def __init__(self, eps=1e-3):
        self.eps = eps
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@contextlib.contextmanager
def hard_mode():
    prev_hard_mode = _is_hard_mode()
    _set_hard_mode(True)
    try:
        yield
    finally:
        _set_hard_mode(prev_hard_mode)


def auto_discover(eps=1e-3, skip=None):
    skip = skip or []
    
    def decorator(func):
        return _wrap_function_for_differentiability(func, eps=eps)
    
    return decorator


def enable(*args, **kwargs):
    return linearize(*args, **kwargs)

import torch
import functools
import inspect


def auto_discover(eps=1e-3, skip=None):
    skip = skip or []
    
    def decorator(func):
        from .core import _wrap_function_for_differentiability, _detect_discrete_nature
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            
            if func_name in skip:
                return func(*args, **kwargs)
            
            is_discrete = _detect_discrete_nature(func, args, kwargs)
            
            if is_discrete:
                wrapped_func = _wrap_function_for_differentiability(func, eps=eps)
                return wrapped_func(*args, **kwargs)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def discover_module(module, eps=1e-3, skip=None):
    skip = skip or []
    
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and name not in skip:
            wrapped = auto_discover(eps=eps, skip=skip)(obj)
            setattr(module, name, wrapped)
    
    return module

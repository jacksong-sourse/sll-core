import torch
import functools


def auto_discover(eps=1e-3, skip=None):
    skip = skip or []
    
    def decorator(func):
        from .core import _wrap_function_for_differentiability
        return _wrap_function_for_differentiability(func, eps=eps)
    
    return decorator

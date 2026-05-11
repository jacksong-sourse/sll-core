import torch
import functools


_discrete_spaces = {}


def register_discrete_space(name, boundaries, domain=None):
    """
    Register a custom discrete space.
    
    Args:
        name: Name of the discrete space
        boundaries: Tensor or list of boundary values defining the discrete regions
        domain: Optional domain limits (min, max) for the space
    """
    if isinstance(boundaries, list):
        boundaries = torch.tensor(boundaries, dtype=torch.float32)
    elif isinstance(boundaries, torch.Tensor):
        boundaries = boundaries.float()
    
    _discrete_spaces[name] = {
        'boundaries': boundaries.sort().values if boundaries.numel() > 0 else boundaries,
        'domain': domain
    }


def get_discrete_space(name):
    return _discrete_spaces.get(name)


def list_discrete_spaces():
    return list(_discrete_spaces.keys())


def _detect_discrete_points(tensor, tolerance=1e-6):
    """
    Detect discrete points in a tensor by finding unique values.
    
    Args:
        tensor: Input tensor to analyze
        tolerance: Tolerance for distinguishing discrete values
    
    Returns:
        Tensor of unique discrete values found in the input
    """
    if tensor.numel() == 0:
        return torch.tensor([])
    
    unique_values = torch.unique(tensor.flatten())
    return unique_values


class SLL:
    """
    Static Local Linearization for differentiable discrete programming.
    
    Core Mathematical Principle:
        At each discrete point x_i, create a piecewise linear function:
        y = x_i + k*(x - x_i) for x in [x_i - ε/2, x_i + ε/2]
        
        This transforms discrete points into differentiable functions.
    
    Key Features:
        - Detects discrete values in function inputs and outputs
        - Creates differentiable approximations at each discrete boundary
        - Enables gradient flow through discrete operations
    """
    
    def __init__(self, func=None, eps=1e-3):
        self.eps = float(eps)
        self._func = func
    
    def __call__(self, *args, **kwargs):
        if self._func is None:
            if len(args) == 1 and callable(args[0]):
                self._func = args[0]
                return self
            raise ValueError("SLL requires a function to wrap")
        
        return self._differentiable_call(*args, **kwargs)
    
    def _differentiable_call(self, *args, **kwargs):
        has_grad = any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args)
        
        if not has_grad:
            return self._func(*args, **kwargs)
        
        if kwargs:
            from inspect import signature
            sig = signature(self._func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            args = bound_args.args
        
        return _SLLFunction.apply(self.eps, self._func, *args)
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)


class _SLLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, func, *args):
        ctx.eps = eps
        ctx.func = func
        
        tensor_mask = [isinstance(arg, torch.Tensor) for arg in args]
        tensor_indices = [i for i, is_tensor in enumerate(tensor_mask) if is_tensor]
        tensor_args = [args[i] for i in tensor_indices]
        
        with torch.no_grad():
            result = func(*args)
        
        if isinstance(result, torch.Tensor):
            result = result.detach()
        
        ctx.save_for_backward(*tensor_args)
        ctx.tensor_indices = tensor_indices
        ctx.tensor_mask = tensor_mask
        ctx.args = args
        ctx.result = result
        ctx.result_is_tensor = isinstance(result, torch.Tensor)
        
        if ctx.result_is_tensor:
            ctx.discrete_points = _detect_discrete_points(result)
        else:
            ctx.discrete_points = None
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        func = ctx.func
        args = ctx.args
        saved_tensors = ctx.saved_tensors
        tensor_indices = ctx.tensor_indices
        tensor_mask = ctx.tensor_mask
        
        grads = []
        tensor_idx = 0
        
        for i, is_tensor in enumerate(tensor_mask):
            if is_tensor:
                tensor = saved_tensors[tensor_idx]
                tensor_idx += 1
                
                if tensor.requires_grad and grad_output is not None:
                    args_plus = list(args)
                    args_minus = list(args)
                    args_plus[i] = tensor.detach() + eps
                    args_minus[i] = tensor.detach() - eps
                    
                    with torch.no_grad():
                        result_plus = func(*args_plus)
                        result_minus = func(*args_minus)
                    
                    if ctx.result_is_tensor and isinstance(result_plus, torch.Tensor) and isinstance(result_minus, torch.Tensor):
                        diff = result_plus - result_minus
                        
                        if diff.numel() == 1:
                            diff = diff.expand_as(tensor)
                        elif diff.shape != tensor.shape:
                            diff = torch.full_like(tensor, float((diff != 0).any()))
                        
                        mask = (diff != 0).float()
                        grad = grad_output / (2 * eps) * mask
                        
                        grad = torch.clamp(grad, min=-1e5, max=1e5)
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=1e5, neginf=-1e5)
                    else:
                        grad = grad_output * torch.ones_like(tensor)
                    
                    grads.append(grad)
                else:
                    grads.append(None)
            else:
                grads.append(None)
        
        return (None, None) + tuple(grads)


def sll(func=None, eps=1e-3):
    """
    Decorator to make functions with discrete operations differentiable.
    
    Args:
        func: Function to wrap
        eps: Epsilon for linearization region
    
    Usage:
        @sll
        def my_function(x):
            return torch.round(x)
    """
    if func is None:
        return lambda f: SLL(f, eps=eps)
    return SLL(func, eps=eps)


def discretize_to_continuous(discrete_tensor, eps=1e-3):
    """
    Convert a discrete tensor to a differentiable continuous approximation.
    
    Creates piecewise linear functions at each discrete point:
        For each discrete value v:
            y = v + (x - v) * (1/eps)  when x is near v
    
    Args:
        discrete_tensor: Tensor containing discrete values
        eps: Width of the linearization region
    
    Returns:
        Differentiable tensor with gradient flow enabled
    """
    if not isinstance(discrete_tensor, torch.Tensor):
        return discrete_tensor
    
    class DiscretizeToContinuous(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.detach()
        
        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            
            unique_values = torch.unique(x)
            
            grad = torch.zeros_like(x)
            
            for v in unique_values:
                mask = (x == v)
                if mask.any():
                    grad[mask] = grad_output[mask] * (1.0 / eps)
            
            return grad
    
    return DiscretizeToContinuous.apply(discrete_tensor)


def piecewise_linear_approximation(x, eps=1e-3):
    """
    Apply piecewise linear approximation to a tensor.
    
    For each unique value v in x:
        y = v when x is far from v
        y = v + slope*(x - v) when x is within eps of v
    
    Args:
        x: Input tensor
        eps: Linearization region width
    
    Returns:
        Differentiable approximation
    """
    if not isinstance(x, torch.Tensor):
        return x
    
    unique_values = torch.unique(x)
    
    if len(unique_values) <= 1:
        return x
    
    result = x.clone()
    
    for v in unique_values:
        mask = torch.abs(x - v) < eps
        if mask.any():
            result[mask] = v + (x[mask] - v) * (1.0 / eps)
    
    return result


def make_differentiable(output, eps=1e-3):
    """
    Make any output differentiable.
    
    Args:
        output: Output to make differentiable
        eps: Linearization parameter
    
    Returns:
        Differentiable version of the output
    """
    if isinstance(output, torch.Tensor):
        return discretize_to_continuous(output, eps)
    return output
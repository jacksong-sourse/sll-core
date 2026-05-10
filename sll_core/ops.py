import torch
import torch.nn.functional as F
from torch.autograd import Function


_original_torch_sign = torch.sign
_original_torch_round = torch.round
_original_torch_floor = torch.floor
_original_torch_ceil = torch.ceil


def create_sll_operator(forward_func, boundary_func, gradient_func, operator_name):
    """
    Factory function for creating SLL (Static Local Linearization) operators.
    
    Args:
        forward_func: Forward pass function, takes (x, eps) and returns result
        boundary_func: Boundary detection function, takes (x, eps) and returns mask
        gradient_func: Gradient computation function, takes (grad_output, mask, eps) and returns gradient
        operator_name: Name of the operator for class name generation
    
    Returns:
        Wrapped function that takes (x, eps) and returns differentiable result
    """
    class_name = f'{operator_name.capitalize()}SLL'
    
    class SLLOperator(Function):
        @staticmethod
        def forward(ctx, x, eps=1e-3):
            ctx.save_for_backward(x)
            ctx.eps = eps
            return forward_func(x, eps)

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            eps = ctx.eps
            mask = boundary_func(x, eps)
            return gradient_func(grad_output, mask, eps), None
    
    SLLOperator.__name__ = class_name
    
    def wrapper(x, eps: float = 1e-3):
        return SLLOperator.apply(x, eps)
    
    wrapper.__name__ = operator_name
    return wrapper


def _heaviside_forward(x, eps):
    return (x >= 0).float()

def _heaviside_boundary(x, eps):
    return (x.abs() <= eps)

def _heaviside_gradient(grad_output, mask, eps):
    grad_x = grad_output.clone()
    grad_x[mask] = grad_output[mask] / (2 * eps)
    grad_x[~mask] = 0
    grad_x = torch.clamp(grad_x, min=-1e5, max=1e5)
    return grad_x


def _sign_forward(x, eps):
    return _original_torch_sign(x)

def _sign_boundary(x, eps):
    return (x.abs() <= eps)

def _sign_gradient(grad_output, mask, eps):
    grad_x = grad_output.clone()
    grad_x[mask] = grad_output[mask] / (2 * eps)
    grad_x[~mask] = 0
    grad_x = torch.clamp(grad_x, min=-1e5, max=1e5)
    return grad_x


def _round_forward(x, eps):
    return _original_torch_round(x)

def _round_boundary(x, eps):
    x_floor = _original_torch_floor(x)
    distance = x - x_floor
    near_integer = ((distance <= eps) | ((1 - distance) <= eps))
    near_midpoint = torch.abs(distance - 0.5) <= eps
    return near_integer | near_midpoint

def _round_gradient(grad_output, mask, eps):
    grad_x = grad_output.clone()
    grad_x[mask] = grad_output[mask] / (2 * eps)
    grad_x[~mask] = 0
    grad_x = torch.clamp(grad_x, min=-1e5, max=1e5)
    return grad_x


def _floor_forward(x, eps):
    return _original_torch_floor(x)

def _floor_boundary(x, eps):
    x_floor = _original_torch_floor(x)
    distance = x - x_floor
    return (distance <= eps)

def _floor_gradient(grad_output, mask, eps):
    grad_x = grad_output.clone()
    grad_x[mask] = grad_output[mask] / eps
    grad_x[~mask] = 0
    grad_x = torch.clamp(grad_x, min=-1e5, max=1e5)
    return grad_x


def _ceil_forward(x, eps):
    return _original_torch_ceil(x)

def _ceil_boundary(x, eps):
    x_ceil = _original_torch_ceil(x)
    distance = x_ceil - x
    return (distance <= eps)

def _ceil_gradient(grad_output, mask, eps):
    grad_x = grad_output.clone()
    grad_x[mask] = grad_output[mask] / eps
    grad_x[~mask] = 0
    grad_x = torch.clamp(grad_x, min=-1e5, max=1e5)
    return grad_x


_heaviside_sll = create_sll_operator(
    _heaviside_forward,
    _heaviside_boundary,
    _heaviside_gradient,
    'heaviside'
)


def heaviside(x, eps: float = 1e-3, values=None):
    """
    Differentiable heaviside function with API compatibility.
    
    Args:
        x: Input tensor
        eps: Boundary detection epsilon (default: 1e-3)
        values: Value to use at x=0 (for API compatibility with torch.heaviside)
    
    Returns:
        Tensor with 1.0 where x > 0, 0.0 where x < 0, and values (or 1.0) where x == 0
    
    Examples::
        >>> import torch
        >>> from sll_core.ops import heaviside
        >>>
        >>> x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = heaviside(x)
        >>> y.backward(torch.ones_like(y))
        
        >>> # With values parameter (compatible with torch.heaviside)
        >>> y = heaviside(x, values=torch.tensor(0.5))
    """
    result = _heaviside_sll(x, eps)
    
    if values is not None:
        mask = (x == 0)
        result = torch.where(mask, values, result)
    
    return result

sign = create_sll_operator(
    _sign_forward,
    _sign_boundary,
    _sign_gradient,
    'sign'
)

round = create_sll_operator(
    _round_forward,
    _round_boundary,
    _round_gradient,
    'round'
)

floor = create_sll_operator(
    _floor_forward,
    _floor_boundary,
    _floor_gradient,
    'floor'
)

ceil = create_sll_operator(
    _ceil_forward,
    _ceil_boundary,
    _ceil_gradient,
    'ceil'
)


def threshold(x, threshold=0.0, eps=1e-3):
    """
    Differentiable threshold function.
    
    Args:
        x: Input tensor
        threshold: Threshold value (default: 0.0)
        eps: Boundary detection epsilon (default: 1e-3)
    
    Returns:
        Tensor with 1.0 where x >= threshold, 0.0 otherwise
    
    Examples::
        >>> import torch
        >>> from sll_core.ops import threshold
        >>>
        >>> x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
        >>> y = threshold(x, threshold=0.5)
        >>> y.backward(torch.ones_like(y))
    """
    return heaviside(x - threshold, eps)


_discrete_ops = {
    'heaviside': heaviside,
    'sign': sign,
    'round': round,
    'floor': floor,
    'ceil': ceil,
    'threshold': threshold,
}


def get_discrete_op(name):
    """
    Get a discrete operator by name.
    
    Args:
        name: Name of the operator ('heaviside', 'sign', 'round', 'floor', 'ceil', 'threshold')
    
    Returns:
        The operator function if found, None otherwise
    """
    return _discrete_ops.get(name)


def is_discrete_op(name):
    """
    Check if a name corresponds to a registered discrete operator.
    
    Args:
        name: Name to check
    
    Returns:
        True if the name is a registered operator, False otherwise
    """
    return name in _discrete_ops
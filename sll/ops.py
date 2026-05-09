import torch
import torch.nn.functional as F
from torch.autograd import Function


_original_torch_sign = torch.sign
_original_torch_round = torch.round
_original_torch_floor = torch.floor
_original_torch_ceil = torch.ceil


def create_sll_operator(forward_func, boundary_func, gradient_func, operator_name):
    """
    自动化创建 SLL (Static Local Linearization) 算子的工厂函数
    
    Args:
        forward_func: 前向传播函数，接受 (x, eps) 返回结果
        boundary_func: 边界检测函数，接受 (x, eps) 返回边界掩码
        gradient_func: 梯度计算函数，接受 (grad_output, mask, eps) 返回梯度
        operator_name: 算子名称，用于生成类名
    
    Returns:
        包装函数，接受 (x, eps) 返回可微结果
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
    
    def wrapper(x, eps=1e-3):
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
    return grad_x


def _sign_forward(x, eps):
    return _original_torch_sign(x)

def _sign_boundary(x, eps):
    return (x.abs() <= eps)

def _sign_gradient(grad_output, mask, eps):
    grad_x = grad_output.clone()
    grad_x[mask] = grad_output[mask] / (2 * eps)
    grad_x[~mask] = 0
    return grad_x


def _round_forward(x, eps):
    return _original_torch_round(x)

def _round_boundary(x, eps):
    x_floor = _original_torch_floor(x)
    distance = x - x_floor
    return ((distance <= eps) | ((1 - distance) <= eps))

def _round_gradient(grad_output, mask, eps):
    grad_x = grad_output.clone()
    grad_x[mask] = grad_output[mask] / (2 * eps)
    grad_x[~mask] = 0
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
    return grad_x


heaviside = create_sll_operator(
    _heaviside_forward,
    _heaviside_boundary,
    _heaviside_gradient,
    'heaviside'
)

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
    return _discrete_ops.get(name)


def is_discrete_op(name):
    return name in _discrete_ops
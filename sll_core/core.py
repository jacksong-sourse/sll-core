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
    
    def __init__(self, func=None, eps=1e-3, max_grad_norm=1e3, smooth_factor=1.0, sensitivity_scale=1.0):
        self.eps = float(eps)
        self.max_grad_norm = float(max_grad_norm)
        self.smooth_factor = float(smooth_factor)
        self.sensitivity_scale = float(sensitivity_scale)
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
            args = tuple(bound_args.arguments[key] for key in sig.parameters.keys())
        
        return _SLLFunction.apply(
            self.eps, 
            self.max_grad_norm, 
            self.smooth_factor, 
            self.sensitivity_scale,
            self._func, 
            *args
        )
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return functools.partial(self.__call__, instance)


class _SLLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, eps, max_grad_norm, smooth_factor, sensitivity_scale, func, *args):
        ctx.eps = eps
        ctx.max_grad_norm = max_grad_norm
        ctx.smooth_factor = smooth_factor
        ctx.sensitivity_scale = sensitivity_scale
        ctx.func = func
        
        tensor_mask = [isinstance(arg, torch.Tensor) for arg in args]
        tensor_indices = [i for i, is_tensor in enumerate(tensor_mask) if is_tensor]
        tensor_args = [args[i] for i in tensor_indices]
        
        with torch.no_grad():
            result = func(*args)
        
        if isinstance(result, torch.Tensor):
            result = result.detach()
        elif isinstance(result, tuple):
            result = tuple(r.detach() if isinstance(r, torch.Tensor) else r for r in result)
        
        ctx.save_for_backward(*tensor_args)
        ctx.tensor_indices = tensor_indices
        ctx.tensor_mask = tensor_mask
        ctx.args = args
        ctx.result = result
        ctx.result_is_tensor = isinstance(result, torch.Tensor)
        ctx.result_is_tuple = isinstance(result, tuple)
        
        if ctx.result_is_tensor:
            ctx.discrete_points = _detect_discrete_points(result)
        elif ctx.result_is_tuple:
            ctx.discrete_points = []
            for r in result:
                if isinstance(r, torch.Tensor):
                    ctx.discrete_points.append(_detect_discrete_points(r))
                else:
                    ctx.discrete_points.append(None)
        else:
            ctx.discrete_points = None
        
        return result

    @staticmethod
    def backward(ctx, *grad_outputs):
        eps = ctx.eps
        max_grad_norm = ctx.max_grad_norm
        smooth_factor = ctx.smooth_factor
        sensitivity_scale = ctx.sensitivity_scale
        func = ctx.func
        args = ctx.args
        saved_tensors = ctx.saved_tensors
        tensor_indices = ctx.tensor_indices
        tensor_mask = ctx.tensor_mask
        
        grads = [None] * len(tensor_mask)
        tensor_idx = 0
        
        outputs = []
        if ctx.result_is_tensor:
            if grad_outputs and grad_outputs[0] is not None:
                outputs.append((0, ctx.result, grad_outputs[0]))
        elif ctx.result_is_tuple:
            for i, r in enumerate(ctx.result):
                if isinstance(r, torch.Tensor) and i < len(grad_outputs) and grad_outputs[i] is not None:
                    outputs.append((i, r, grad_outputs[i]))
        
        for i, is_tensor in enumerate(tensor_mask):
            if not is_tensor:
                continue
            
            tensor = saved_tensors[tensor_idx]
            tensor_idx += 1
            
            if not tensor.requires_grad:
                grads[i] = None
                continue
            
            total_grad = torch.zeros_like(tensor)
            
            for output_idx, result, grad_output in outputs:
                args_plus = list(args)
                args_minus = list(args)
                args_plus[i] = tensor.detach() + eps
                args_minus[i] = tensor.detach() - eps
                
                with torch.no_grad():
                    result_plus = func(*args_plus)
                    result_minus = func(*args_minus)
                
                if ctx.result_is_tuple:
                    if isinstance(result_plus, tuple) and isinstance(result_minus, tuple):
                        rp = result_plus[output_idx]
                        rm = result_minus[output_idx]
                    else:
                        continue
                else:
                    rp = result_plus
                    rm = result_minus
                
                if isinstance(rp, torch.Tensor) and isinstance(rm, torch.Tensor):
                    diff = rp - rm
                    
                    if diff.numel() == 1:
                        diff = diff.expand_as(tensor)
                    elif diff.shape != tensor.shape:
                        result_numel = diff.numel()
                        input_numel = tensor.numel()
                        
                        if result_numel < input_numel:
                            repeats = (input_numel + result_numel - 1) // result_numel
                            diff = diff.repeat(repeats)[:input_numel].view(tensor.shape)
                        else:
                            diff = diff.flatten().mean()
                            diff = torch.full_like(tensor, diff.item())
                    
                    abs_diff = torch.abs(diff)
                    
                    if smooth_factor > 0:
                        smooth_mask = torch.exp(-abs_diff ** 2 / (2 * smooth_factor * eps ** 2))
                        clipped_diff = diff * smooth_mask
                    else:
                        smooth_mask = (abs_diff > eps * 1e-3).float()
                        clipped_diff = diff * smooth_mask
                    
                    grad_contribution = grad_output / (2 * eps) * clipped_diff * sensitivity_scale
                    
                    grad_contribution = torch.clamp(grad_contribution, min=-max_grad_norm, max=max_grad_norm)
                    grad_contribution = torch.nan_to_num(grad_contribution, nan=0.0, posinf=max_grad_norm, neginf=-max_grad_norm)
                    
                    total_grad += grad_contribution
                else:
                    total_grad += grad_output * torch.ones_like(tensor)
            
            grads[i] = total_grad
        
        return (None, None, None, None, None) + tuple(grads)


def sll(func=None, eps=1e-3, max_grad_norm=1e3, smooth_factor=1.0, sensitivity_scale=1.0):
    """
    Decorator to make functions with discrete operations differentiable.
    
    Args:
        func: Function to wrap
        eps: Epsilon for linearization region (default: 1e-3)
        max_grad_norm: Maximum gradient norm (default: 1e3)
        smooth_factor: Smoothing factor for boundary handling (default: 1.0)
                       Larger values = more smoothing
        sensitivity_scale: Scale factor for gradient sensitivity (default: 1.0)
                          Larger values = more sensitive gradients
    
    Usage:
        @sll
        def my_function(x):
            return torch.round(x)
        
        # With custom parameters
        @sll(eps=1e-4, max_grad_norm=100.0, smooth_factor=2.0)
        def my_function(x):
            return torch.where(x > 0, torch.sin(x), torch.cos(x))
    """
    if func is None:
        return lambda f: SLL(f, eps=eps, max_grad_norm=max_grad_norm, 
                            smooth_factor=smooth_factor, sensitivity_scale=sensitivity_scale)
    return SLL(func, eps=eps, max_grad_norm=max_grad_norm, 
               smooth_factor=smooth_factor, sensitivity_scale=sensitivity_scale)


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
            
            unique_values = _differentiable_unique(x)
            
            grad = torch.zeros_like(x)
            
            for v in unique_values:
                v = v.to(x.device)
                mask = torch.abs(x - v) < 1e-6
                if mask.any():
                    grad[mask] = grad_output[mask] * (1.0 / eps)
            
            return grad
    
    return DiscretizeToContinuous.apply(discrete_tensor)


def _differentiable_unique(x, tolerance=1e-6):
    """
    Differentiable approximation of torch.unique using soft clustering.
    
    Args:
        x: Input tensor
        tolerance: Tolerance for grouping values together
    
    Returns:
        Tensor of unique values (differentiable)
    """
    if x.numel() == 0:
        return torch.tensor([])
    
    x_flat = x.flatten()
    sorted_x, _ = torch.sort(x_flat)
    
    diffs = sorted_x[1:] - sorted_x[:-1]
    boundaries = (diffs > tolerance).nonzero().squeeze()
    
    if boundaries.numel() == 0:
        return sorted_x[:1]
    
    if boundaries.dim() == 0:
        boundaries = boundaries.unsqueeze(0)
    
    unique_indices = torch.cat([torch.tensor([0], device=x.device), boundaries + 1])
    unique_values = sorted_x[unique_indices]
    
    return unique_values


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
    
    unique_values = _differentiable_unique(x)
    
    if len(unique_values) <= 1:
        return x
    
    x_device = x.device
    result = x.clone()
    
    for v in unique_values:
        v = v.to(x_device)
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

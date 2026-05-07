import functools
import torch
import torch.fx


def check_type_transition(sample_input, sample_output):
    """检查类型跃迁：输入是浮点张量，输出变成非浮点类型"""
    if not isinstance(sample_input, torch.Tensor) or not isinstance(sample_output, torch.Tensor):
        return False
    
    input_is_float = sample_input.dtype.is_floating_point
    output_is_float = sample_output.dtype.is_floating_point
    
    if input_is_float and not output_is_float:
        return True
    return False


def check_numerical_jump(func, sample_input, eps=1e-4, delta=0.1):
    """检查数值跳跃：对输入注入 eps 扰动，输出发生不可解释的跳跃"""
    if not isinstance(sample_input, torch.Tensor):
        return False
    
    x = sample_input.detach().clone()
    x.requires_grad_(False)
    
    try:
        y0 = func(x)
        if not isinstance(y0, torch.Tensor):
            return False
        
        x_perturbed = x + eps * torch.randn_like(x)
        y1 = func(x_perturbed)
        
        input_delta = (x_perturbed - x).abs().mean().item()
        output_delta = (y1 - y0).abs().mean().item()
        
        if input_delta > 0:
            ratio = output_delta / input_delta
        else:
            ratio = float('inf')
        
        if output_delta > delta or ratio > 100:
            return True
    except Exception:
        pass
    
    return False


def check_gradient_blackhole(func, sample_input):
    """检查梯度黑洞：前向输出非恒定，但反向传播时梯度为 0 或 None"""
    if not isinstance(sample_input, torch.Tensor):
        return False
    
    x = sample_input.detach().clone()
    x.requires_grad_(True)
    
    try:
        y = func(x)
        if not isinstance(y, torch.Tensor):
            return False
        
        if not y.requires_grad:
            return True
        
        if y.detach().std() < 1e-12:
            return False
        
        y.sum().backward(retain_graph=True)
        
        if x.grad is None or x.grad.abs().max() < 1e-12:
            return True
    except Exception:
        return True
    
    return False


def check_output_clustering(func, sample_input, n_samples=50):
    """检查输出聚集：输出值只出现在少数几个固定点"""
    if not isinstance(sample_input, torch.Tensor):
        return False
    
    outputs = []
    for _ in range(n_samples):
        x = torch.randn_like(sample_input)
        try:
            y = func(x)
            if isinstance(y, torch.Tensor):
                outputs.extend(y.flatten().tolist())
        except Exception:
            continue
    
    if len(outputs) == 0:
        return False
    
    unique_values = len(set([round(v, 6) for v in outputs]))
    
    if unique_values < len(outputs) * 0.1:
        return True
    return False


@functools.lru_cache(maxsize=128)
def probe_function_signature(func_id, input_shape, input_dtype):
    """缓存探测结果，避免重复探测相同函数签名"""
    pass


class DiscreteDiscoveryTracer(torch.fx.Tracer):
    """
    运行时离散性质发现追踪器。
    在构建计算图时，对每个节点进行离散性质探测。
    """
    
    def __init__(self, eps=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self._probe_cache = {}
    
    def _extract_tensor_inputs(self, args, kwargs):
        """从 args 和 kwargs 中提取所有张量输入"""
        inputs = []
        
        def collect(x):
            if isinstance(x, torch.Tensor):
                inputs.append(x)
            elif isinstance(x, (list, tuple)):
                for item in x:
                    collect(item)
            elif isinstance(x, dict):
                for value in x.values():
                    collect(value)
        
        collect(args)
        collect(kwargs)
        return inputs
    
    def _probe_node(self, node, args, kwargs):
        """对节点进行离散性质探测"""
        tensor_inputs = self._extract_tensor_inputs(args, kwargs)
        
        if not tensor_inputs:
            return False, None
        
        sample_input = tensor_inputs[0]
        
        def func(x):
            return node.target(x, *args[1:], **kwargs) if args else node.target(x, **kwargs)
        
        discrete_type = None
        
        if check_type_transition(sample_input, func(sample_input)):
            discrete_type = 'type_transition'
        elif check_numerical_jump(func, sample_input, eps=self.eps):
            discrete_type = 'numerical_jump'
        elif check_gradient_blackhole(func, sample_input):
            discrete_type = 'gradient_blackhole'
        elif check_output_clustering(func, sample_input):
            discrete_type = 'output_clustering'
        
        return discrete_type is not None, discrete_type
    
    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        
        if kind in ('call_function', 'call_method'):
            try:
                is_discrete, discrete_type = self._probe_node(node, args, kwargs)
                node.meta['is_discrete'] = is_discrete
                node.meta['discrete_type'] = discrete_type
                node.meta['eps'] = self.eps
            except Exception:
                node.meta['is_discrete'] = False
                node.meta['discrete_type'] = None
                node.meta['eps'] = self.eps
        else:
            node.meta['is_discrete'] = False
            node.meta['discrete_type'] = None
            node.meta['eps'] = self.eps
        
        return node

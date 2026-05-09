import pytest
import torch
import sys
sys.path.insert(0, '.')
import sll
from sll.ops import heaviside, sign, round, floor, ceil, threshold


def test_large_scale_quantization():
    batch_size = 1000
    x = torch.randn(batch_size, 100, requires_grad=True)
    
    @sll.linearize(eps=1e-3)
    def quantize(x, levels=256):
        scale = (levels - 1) / (x.max() - x.min() + 1e-10)
        return torch.round((x - x.min()) * scale) / scale + x.min()
    
    y = quantize(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_large_scale_mixed_operations():
    batch_size = 500
    x = torch.randn(batch_size, 50, requires_grad=True)
    
    @sll.linearize(eps=1e-2)
    def complex_discrete_ops(x):
        a = torch.sign(x)
        b = torch.round(a * 10)
        c = torch.floor(b / 2)
        d = torch.ceil(c + 0.5)
        e = (d > 0).float()
        return e
    
    y = complex_discrete_ops(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_knapsack_optimization():
    n_items = 20
    item_weights = torch.rand(n_items, requires_grad=True)
    item_values = torch.rand(n_items, requires_grad=True)
    capacity = torch.tensor(5.0)
    
    @sll.linearize(eps=1e-2)
    def knapsack(probabilities):
        selected = (probabilities > 0.5).float()
        total_weight = (selected * item_weights).sum()
        total_value = (selected * item_values).sum()
        penalty = torch.max(torch.tensor(0.0), total_weight - capacity) * 100
        return total_value - penalty
    
    probabilities = torch.sigmoid(torch.randn(n_items))
    
    optimizer = torch.optim.Adam([item_weights, item_values], lr=1e-2)
    
    for _ in range(10):
        optimizer.zero_grad()
        value = knapsack(probabilities)
        (-value).backward()
        optimizer.step()
    
    assert item_weights.grad is not None
    assert item_values.grad is not None


def test_discrete_model_training():
    class DiscreteModel(torch.nn.Module):
        def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
            self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
        @sll.linearize(eps=1e-3)
        def forward(self, x):
            x = self.fc1(x)
            x = torch.sign(x)
            x = self.fc2(x)
            return (x > 0).float()
    
    model = DiscreteModel()
    
    batch_size = 32
    x = torch.randn(batch_size, 10, requires_grad=True)
    
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_vectorized_discrete_operations():
    n_elements = 10000
    x = torch.linspace(-10, 10, n_elements, requires_grad=True)
    
    @sll.linearize(eps=1e-3)
    def process(x):
        a = heaviside(x)
        b = sign(x)
        c = round(x)
        d = floor(x)
        e = ceil(x)
        return a + b + c + d + e
    
    y = process(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_nested_discrete_functions():
    def inner_func(x):
        return torch.round(x)
    
    def middle_func(x):
        return torch.sign(inner_func(x))
    
    def outer_func(x):
        return (middle_func(x) > 0).float()
    
    @sll.linearize(eps=1e-3)
    def nested_ops(x):
        return outer_func(x)
    
    x = torch.tensor([0.0005], requires_grad=True)
    y = nested_ops(x)
    y.backward()
    
    assert x.grad is not None
    assert x.grad.item() != 0.0


def test_performance_stress_test():
    n_iterations = 100
    x = torch.tensor([0.0], requires_grad=True)
    
    @sll.linearize(eps=1e-3)
    def stress_func(x):
        result = x
        for _ in range(100):
            result = torch.sign(result) + torch.round(result * 0.1)
        return result
    
    for _ in range(n_iterations):
        y = stress_func(x)
        y.backward(retain_graph=True)
        x.grad.zero_()
    
    assert True


def test_large_batch_gradients():
    batch_size = 1024
    x = torch.randn(batch_size, requires_grad=True)
    
    @sll.linearize(eps=1e-3)
    def batch_process(x):
        return torch.sign(x) + torch.round(x * 10)
    
    y = batch_process(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()
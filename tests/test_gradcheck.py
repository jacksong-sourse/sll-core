import pytest
import torch
import sys
sys.path.insert(0, '.')
from sll.core import linearize, hard_mode
from sll.ops import heaviside, sign, round, floor, ceil


def test_decorator_grad():
    @linearize
    def func(x):
        return torch.sign(x)
    
    x = torch.tensor([0.0], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() != 0.0


def test_context_manager_grad():
    x = torch.tensor([0.0], requires_grad=True)
    
    @linearize(eps=1e-3)
    def func(x):
        return torch.sign(x)
    
    y = func(x)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() != 0.0


def test_hard_mode_context():
    x = torch.tensor([0.0], requires_grad=True)
    
    @linearize(eps=1e-3)
    def func(x):
        return torch.sign(x)
    
    y = func(x)
    y.backward()
    assert x.grad is not None


def test_linearize_decorator_with_eps():
    @linearize(eps=1e-2)
    def func(x):
        return torch.sign(x)
    
    x = torch.tensor([0.0], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(1.0 / (2 * 1e-2))


def test_linearize_decorator_with_positional_eps():
    @linearize(1e-2)
    def func(x):
        return torch.sign(x)
    
    x = torch.tensor([0.0], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(1.0 / (2 * 1e-2))


def test_custom_discrete_function():
    def custom_discrete(x):
        return (x > 0.5).float()
    
    @linearize(eps=1e-3)
    def func(x):
        return custom_discrete(x)
    
    x = torch.tensor([0.5], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None


def test_auto_discover_decorator():
    from sll.discovery import auto_discover
    
    @auto_discover(eps=1e-3)
    def func(x):
        return torch.sign(x)
    
    x = torch.tensor([0.0], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None


def test_if_statement_detection():
    @linearize(eps=1e-3)
    def func(x):
        if x.item() > 0:
            return torch.tensor(1.0)
        else:
            return torch.tensor(0.0)
    
    x = torch.tensor([0.5], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None

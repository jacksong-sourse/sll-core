import pytest
import torch
import sys
sys.path.insert(0, '.')
from sll.discovery import auto_discover


def test_auto_discover_sign():
    @auto_discover(eps=1e-3)
    def func(x):
        return torch.sign(x)
    
    x = torch.tensor([0.0], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() != 0.0


def test_auto_discover_round():
    @auto_discover(eps=1e-3)
    def func(x):
        return torch.round(x)
    
    x = torch.tensor([1.0005], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() != 0.0


def test_auto_discover_mixed():
    @auto_discover(eps=1e-3)
    def func(x):
        a = torch.sign(x)
        b = torch.round(a * 10)
        return b
    
    x = torch.tensor([0.0], requires_grad=True)
    y = func(x)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() != 0.0


def test_auto_discover_skip():
    original_sign = torch.sign
    
    @auto_discover(eps=1e-3, skip=[])
    def func(x):
        return torch.sign(x)
    
    x = torch.tensor([0.0], requires_grad=True)
    y = func(x)
    y.backward()
    
    assert torch.sign is original_sign


def test_auto_discover_custom_discrete():
    def custom_discrete(x):
        return (x > 0.5).float()
    
    @auto_discover(eps=1e-3)
    def func(x):
        return custom_discrete(x)
    
    x = torch.tensor([0.5], requires_grad=True)
    y = func(x)
    assert y.item() == 0.0 or y.item() == 1.0

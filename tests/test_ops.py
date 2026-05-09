import pytest
import torch
import sys
sys.path.insert(0, '.')
from sll.ops import heaviside, sign, round, floor, ceil, threshold


def test_heaviside_forward():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    result = heaviside(x)
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
    assert torch.allclose(result, expected)


def test_heaviside_gradient():
    x = torch.tensor([0.0], requires_grad=True)
    eps = 1e-3
    y = heaviside(x, eps=eps)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(1.0 / (2 * eps))


def test_sign_forward():
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    result = sign(x)
    expected = torch.tensor([-1.0, -1.0, 0.0, 1.0, 1.0])
    assert torch.allclose(result, expected)


def test_sign_gradient():
    x = torch.tensor([0.0], requires_grad=True)
    eps = 1e-3
    y = sign(x, eps=eps)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(1.0 / (2 * eps))


def test_round_forward():
    x = torch.tensor([1.2, 1.5, 1.8, -1.2, -1.5])
    result = round(x)
    expected = torch.tensor([1.0, 2.0, 2.0, -1.0, -2.0])
    assert torch.allclose(result, expected)


def test_floor_forward():
    x = torch.tensor([1.2, 1.8, -1.2, -1.8])
    result = floor(x)
    expected = torch.tensor([1.0, 1.0, -2.0, -2.0])
    assert torch.allclose(result, expected)


def test_ceil_forward():
    x = torch.tensor([1.2, 1.8, -1.2, -1.8])
    result = ceil(x)
    expected = torch.tensor([2.0, 2.0, -1.0, -1.0])
    assert torch.allclose(result, expected)


def test_threshold():
    x = torch.tensor([-1.0, 0.0, 0.5, 1.0])
    result = threshold(x, threshold=0.5)
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
    assert torch.allclose(result, expected)


def test_sign_gradient_far_from_boundary():
    x = torch.tensor([1.0], requires_grad=True)
    eps = 1e-3
    y = sign(x, eps=eps)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() == 0.0


def test_round_gradient_at_boundary():
    x = torch.tensor([1.0], requires_grad=True)
    eps = 1e-3
    y = round(x, eps=eps)
    y.backward()
    assert x.grad is not None
    assert x.grad.item() == pytest.approx(1.0 / (2 * eps))

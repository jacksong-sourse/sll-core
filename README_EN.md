<div align="center">

# 🔷 SLL-Core: Static Local Linearization

**Zero-Invasion Differentiable Engine for Discrete Programs**

[![PyPI Version](https://img.shields.io/pypi/v/sll-core.svg)](https://pypi.org/project/sll-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sll-core.svg)](https://pypi.org/project/sll-core/)
[![License](https://img.shields.io/github/license/jacksong-sourse/sll-core.svg)](https://github.com/jacksong-sourse/sll-core/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/sll-core)](https://pepy.tech/project/sll-core)

<p align="center">
  <a href="./README.md">中文</a> | <a href="./README_EN.md">English</a>
</p>

</div>

---

## 🎯 Introduction

SLL-Core is a PyTorch library based on **Static Local Linearization** principle, providing **zero-invasion** automatic differentiation for discrete operations.

**Key Advantages**:
- ✅ **Zero Code Changes**: Decorate existing code directly, no model structure modification required
- ✅ **Zero Deployment Overhead**: Differentiable during training, automatically restores hard logic during deployment
- ✅ **Stable Convergence**: Constant gradient design, no vanishing/exploding gradient issues
- ✅ **Mathematical Guarantee**: As ε→0, the optimal solution converges to the original discrete problem

---

## ⚡ Quick Start

```python
import torch
import sll

# Decorate to make discrete operations differentiable
@ sll.linearize(eps=1e-2)
def my_discrete_function(x):
    y = torch.sign(x)      # Automatically differentiable!
    z = torch.round(y * 10)
    return z.sum()

x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
loss = my_discrete_function(x)
loss.backward()

print(x.grad)  # ✅ Gradient flows normally
```

---

## 🚀 Installation

```bash
pip install sll-core
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 1.9.0

---

## 📖 Usage

### Method 1: Decorator (Recommended)

```python
import torch
import sll

@ sll.linearize(eps=1e-3)
def custom_algorithm(x):
    mask = (x > 0.5).float()   # Auto-discovered and softened
    y = torch.sign(x)           # Auto-discovered and softened
    return mask * y

x = torch.tensor([-0.5, 0.5], requires_grad=True)
y = custom_algorithm(x)
y.sum().backward()
```

### Method 2: Context Manager

```python
import torch
import sll

x = torch.tensor([1.2, 2.5], requires_grad=True)

with sll.linearize(eps=1e-3):
    y = torch.round(x)
    y.backward(torch.ones_like(y))

print(x.grad)  # ✅ Gradient flows normally
```

### Method 3: Manual Operators

```python
from sll.ops import heaviside, sign, round, floor, ceil

x = torch.tensor([0.0], requires_grad=True)
y = sll.sign(x, eps=1e-3)
y.backward()
print(x.grad)  # tensor([500.])
```

---

## 🔧 Supported Operators

| Operator | Description | Usage Example |
|----------|-------------|---------------|
| `heaviside` | Heaviside step function | `sll.heaviside(x)` |
| `sign` | Sign function | `sll.sign(x)` |
| `round` | Round to nearest integer | `sll.round(x)` |
| `floor` | Floor function | `sll.floor(x)` |
| `ceil` | Ceiling function | `sll.ceil(x)` |
| `threshold` | General threshold function | `sll.threshold(x, threshold=0.5)` |

---

## 🔬 Applications

### Application 1: Quantization-Aware Training (QAT)

```python
@ sll.linearize(eps=1e-3)
def quantize(x, levels=256):
    scale = (levels - 1) / (x.max() - x.min() + 1e-10)
    return torch.round((x - x.min()) * scale) / scale + x.min()
```

### Application 2: Combinatorial Optimization

```python
@ sll.linearize(eps=1e-2)
def knapsack(probabilities):
    selected = (probabilities > 0.5).float()
    total_weight = (selected * weights).sum()
    total_value = (selected * values).sum()
    penalty = torch.max(torch.tensor(0.0), total_weight - capacity) * 100
    return total_value - penalty
```

### Application 3: Discrete Control Policy

```python
@ sll.linearize(eps=1e-3)
def discrete_controller(state):
    action_prob = torch.sigmoid(state)
    action = (action_prob > 0.5).float()  # Discrete decision
    return action
```

---

## ⚙️ Parameter Description

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps` | float | 1e-3 | Half-width of linearization interval |

**How `eps` works**:
- Input within `eps` of hard boundary: Use linearization approximation (has gradient)
- Input beyond `eps` from hard boundary: Use original hard logic (gradient=0)
- Smaller `eps`: Closer to hard logic, narrower gradient region
- Larger `eps`: Smoother transition, wider approximation region

---

## 📊 Gradient Comparison

| Method | Forward Output | Boundary Gradient | Far from Boundary | Tuning Difficulty |
|--------|---------------|-------------------|-------------------|-------------------|
| Hard Function | Exact | 0 | 0 | - |
| STE | Exact | 1 | 1 | - |
| Sigmoid Relaxation | Approximate | Gaussian peak | 0 | High |
| **SLL** | **Exact** | **1/(2ε)** | **0** | **Low** |

---

## 🏛️ Project Structure

```
sll-core/
├── sll/
│   ├── __init__.py          # Module exports
│   ├── core.py              # Core API (linearize)
│   ├── discovery.py         # Auto-discovery decorator
│   └── ops.py               # SLL operator implementations
├── README.md
├── README_EN.md
├── LICENSE
└── pyproject.toml
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

### Development Environment

```bash
git clone https://github.com/jacksong-sourse/sll-core.git
cd sll-core
pip install -e .[dev]
```

### Running Tests

```bash
pytest tests/ -v
```

---

## 📚 Citation

If you use SLL in your research, please cite:

```bibtex
@software{sll-core,
  title = {SLL-Core: Static Local Linearization for Differentiable Discrete Programming},
  author = {Jackson Guo},
  year = {2024},
  url = {https://github.com/jacksong-sourse/sll-core},
}
```

---

**⭐ If this project helps you, please give it a Star!**
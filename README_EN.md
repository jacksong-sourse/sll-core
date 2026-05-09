<div align="center">

# 🔷 SLL-Core: Static Local Linearization

**Zero-Invasion Differentiable Engine for Discrete Programs**

[![PyPI](https://img.shields.io/pypi/v/sll-core)](https://pypi.org/project/sll-core/)
[![License](https://img.shields.io/github/license/jacksong-sourse/sll-core)](https://github.com/jacksong-sourse/sll-core/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/sll-core)](https://pypi.org/project/sll-core/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/jacksong-sourse/sll-core/ci.yml)](https://github.com/jacksong-sourse/sll-core/actions)
[![Coverage](https://img.shields.io/codecov/c/github/jacksong-sourse/sll-core)](https://codecov.io/gh/jacksong-sourse/sll-core)

<p align="center">
  <a href="./README.md">中文</a> | <a href="./README_EN.md">English</a>
</p>

</div>

---

## 🤯 Problem: Why Can't Discrete Programs Be Differentiated?

In deep learning, discrete decisions are everywhere:

- **Quantization**: `round(x)`, `floor(x)`
- **Thresholding**: `sign(x)`, `x > 0`
- **Classification**: `argmax(x)`

However, these operations have a critical issue: gradients are almost everywhere zero, making standard backpropagation ineffective.

```python
x = torch.tensor([0.5], requires_grad=True)
y = torch.sign(x)   # ❌ Gradient is 0, parameters never update
loss = (y - target).pow(2).sum()
loss.backward()
print(x.grad)       # tensor([0.]) ← Dead
```

### Limitations of Traditional Approaches

| Method | Requires Code Change | Deployment Residue | Gradient Quality | Convergence Stability |
|--------|---------------------|--------------------|------------------|----------------------|
| Hard Function Direct Training | ✅ No Change | ✅ No Residue | ❌ Zero gradient, untrainable | ❌ No convergence |
| Sigmoid/Softmax Relaxation | ❌ Rewrite Model | ❌ Approximation Error | ⚠️ Vanishing/Exploding | ⚠️ Hard to tune |
| Straight-Through Estimator (STE) | ❌ Custom Gradient | ✅ No Residue | ⚠️ Incorrect direction | ⚠️ Oscillation |
| Reparameterization/Gumbel-Softmax | ❌ Change Structure | ❌ Temperature Residue | ⚠️ High variance | ⚠️ Slow |
| ⭐ SLL (Static Local Linearization) | ✅ Zero-Invasion | ✅ Exact Hard Logic | ✅ Constant gradient | ✅ Stable |

**Core Insight**: Instead of approximating over the entire domain, SLL only linearizes locally within an ε-interval near decision boundaries, preserving original hard logic elsewhere. As `ε → 0`, the optimal solution converges to the original discrete problem's optimal solution.

---

## ⚡ One-Line Solution

```python
import torch
import sll

x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

@ sll.linearize(eps=1e-2)
def compute(x):
    y = torch.sign(x)              # Automatically differentiable!
    z = torch.round(y * 10)
    return z.sum()

loss = compute(x)
loss.backward()

print(x.grad)                      # Gradient flows normally ✅
```

Outside the decorator, `torch.sign` automatically restores to original hard logic — differentiable during training, zero overhead during deployment.

---

## 🚀 Installation

```bash
pip install sll-core
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 1.9.0

---

## 🎯 Quick Start

### Method 1: Decorator (Recommended)

```python
import torch
import sll

@ sll.linearize(eps=1e-3)
def my_custom_algorithm(x):
    mask = (x > 0.5).float()       # Auto-discovered and softened
    y = torch.sign(x)               # Auto-discovered and softened
    return mask * y

x = torch.tensor([-0.5, 0.5], requires_grad=True)
y = my_custom_algorithm(x)
y.sum().backward()                  # Gradient flows ✅
```

### Method 2: Auto Discover

Runtime automatic detection and softening of discrete operations:

```python
from sll.discovery import auto_discover

@ auto_discover(eps=1e-3)
def algorithm(x):
    a = torch.sign(x)
    b = torch.round(a * 10)
    return b
```

### Method 3: Manual Operators

Directly use predefined SLL operators:

```python
from sll.ops import heaviside, sign, round, floor, ceil

x = torch.tensor([0.0], requires_grad=True)
y = sll.sign(x, eps=1e-3)
y.backward()
print(x.grad)                      # tensor([500.])
```

---

## 📊 Why SLL is Better?

### Gradient Quality Comparison

|      | Hard Function | STE | Sigmoid Relaxation | SLL |
|------|--------------|-----|---------------------|-----|
| Forward Output | `[-1, 0, 1]` | `[-1, 0, 1]` | Continuous (with error) | Exact hard output |
| Gradient Near Boundary | `0` | `1` (constant) | Gaussian peak (vanishes) | Constant `1/(2ε)` |
| Gradient Far from Boundary | `0` | `1 ≈ 0` | `0` | `0` (hard logic) |
| Requires Temperature Tuning | — | — | Requires `β` | No tuning needed |

### Mathematical Principle

SLL establishes a local linearization interval near discrete decision boundaries:

$$
y(x) = 
  \begin{cases}
    0.5 + x/(2\epsilon) & \text{when } |x| \leq \epsilon \\
    H(x) & \text{otherwise}
  \end{cases}
$$

Where `H(x)` is the original Heaviside function. As `ε → 0`, `y(x) → H(x)`, and the optimal solution converges to the original problem's optimal solution.

---

## 📋 Supported Differentiable Discrete Operators

### Built-in Operators (Out of the Box)

| Operator | Description | Usage Example |
|----------|-------------|---------------|
| `heaviside` | Heaviside step function | `sll.heaviside(x)` |
| `sign` | Sign function | `sll.sign(x)` |
| `round` | Round to nearest integer | `sll.round(x)` |
| `floor` | Floor function | `sll.floor(x)` |
| `ceil` | Ceiling function | `sll.ceil(x)` |
| `threshold` | General threshold function | `sll.threshold(x, threshold=0.5)` |

### Auto-Discovery Mechanism

Through runtime detection, SLL can automatically identify and soften:
- ✅ User-defined discrete functions
- ✅ Complex composite discrete logic

---

## 🔬 Real-World Applications

### Application 1: Quantization-Aware Training (QAT)

```python
import torch
import sll

def quantize(x, levels=256):
    scale = (levels - 1) / (x.max() - x.min() + 1e-10)
    return torch.round((x - x.min()) * scale) / scale + x.min()

x = torch.randn(10, requires_grad=True)

@ sll.linearize(eps=1e-3)
def forward(x):
    return quantize(x)

y = forward(x)
y.sum().backward()
print("Quantization gradient:", x.grad)  # ✅ Gradient flows
```

### Application 2: Combinatorial Optimization (Knapsack Problem)

```python
import torch
import sll

item_weights = torch.tensor([2, 3, 4, 5], dtype=torch.float32)
item_values = torch.tensor([3, 4, 5, 6], dtype=torch.float32)
capacity = torch.tensor(8.0)

@ sll.linearize(eps=1e-2)
def knapsack(probabilities):
    selected = (probabilities > 0.5).float()
    total_weight = (selected * item_weights).sum()
    total_value = (selected * item_values).sum()
    penalty = torch.max(torch.tensor(0.0), total_weight - capacity) * 100
    return total_value - penalty

probabilities = torch.sigmoid(torch.randn(4), requires_grad=True)
optimizer = torch.optim.Adam([probabilities], lr=1e-2)

for epoch in range(100):
    optimizer.zero_grad()
    total_value = knapsack(probabilities)
    (-total_value).backward()
    optimizer.step()

print("Optimal value:", total_value.item())  # ✅ Gradient flows
```

---

## ⚙️ Parameter Description

- `eps`: Half-width of linearization interval, default `1e-3`
  - Input within `eps` of hard boundary: Use linearization approximation
  - Input beyond `eps` from hard boundary: Use original hard logic
  - Smaller `eps`: Closer to hard logic, narrower gradient region
  - Larger `eps`: Smoother transition, wider approximation region

---

## 🏛️ Project Structure

```
sll-core/
├── sll/
│   ├── __init__.py          # Module exports
│   ├── core.py              # Core API (linearize)
│   ├── discovery.py         # Auto-discovery decorator
│   └── ops.py               # SLL operator implementations (with factory)
├── tests/
│   ├── test_discovery.py    # Discovery tests
│   ├── test_gradcheck.py    # Gradient check tests
│   ├── test_ops.py          # Operator tests
│   └── test_large_scale.py  # Large-scale scenario tests
├── README.md
├── README_EN.md
├── LICENSE
└── pyproject.toml
```

---

## 📄 License

MIT License

---

## 🤝 Contributing

Welcome to submit Issues and Pull Requests!

### Development Environment

```bash
git clone https://github.com/jacksong-sourse/sll-core.git
cd sll-core
pip install -e ".[dev]"
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
  author = {Jacksong},
  year = {2026},
  url = {https://github.com/jacksong-sourse/sll-core},
}
```

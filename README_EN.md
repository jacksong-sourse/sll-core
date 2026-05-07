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

## 🤯 Problem: Why Can't Discrete Programs Be Differentiated Automatically?

In deep learning, discrete decisions are everywhere:

- **Quantization**: `round(x)`, `floor(x)`
- **Thresholding**: `sign(x)`, `x > 0`
- **Classification**: `argmax(x)`

However, these operations have a fatal flaw: gradients are almost everywhere zero, making standard backpropagation impossible.

```python
x = torch.tensor([0.5], requires_grad=True)
y = torch.sign(x)   # ❌ Gradient is 0, parameters can never update
loss = (y - target).pow(2).sum()
loss.backward()
print(x.grad)       # tensor([0.]) ← Dead end
```

### Limitations of Traditional Approaches

| Method | Code Changes Needed | Deployment Overhead | Gradient Quality | Convergence Stability |
|--------|---------------------|---------------------|------------------|----------------------|
| Hard Function Direct Training | ✅ No changes | ✅ No overhead | ❌ Zero gradient, untrainable | ❌ No convergence |
| Sigmoid/Softmax Relaxation | ❌ Rewrite model | ❌ Approximation error | ⚠️ Vanishing/exploding gradients | ⚠️ Tuning difficult |
| Straight-Through Estimator (STE) | ❌ Custom gradient | ✅ No overhead | ⚠️ Incorrect gradient direction | ⚠️ Oscillations |
| Reparameterization/Gumbel-Softmax | ❌ Change model structure | ❌ Temperature parameter | ⚠️ High variance | ⚠️ Slow |
| ⭐ SLL (Static Local Linearization) | ✅ Zero invasion | ✅ Exact hard logic restoration | ✅ Constant gradient, no vanishing | ✅ Stable convergence |

**SLL Core Insight**: No need to approximate over the entire domain. Only linearize locally within an ε-neighborhood around decision boundaries, while keeping original hard logic elsewhere. As `ε → 0`, the optimal solution converges to the optimal solution of the original discrete problem.

---

## ⚡ One-Liner Solution

```python
import torch
import sll

x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

with sll.linearize(eps=1e-2):     # ← Just add this line
    y = torch.sign(x)              # Automatically differentiable!
    z = torch.round(y * 10)
    loss = z.sum()
    loss.backward()

print(x.grad)                      # Gradient flows normally ✅
```

After exiting the context, `torch.sign` automatically restores to original hard logic — differentiable during training, zero overhead during deployment.

---

## 🚀 Installation

```bash
pip install sll-core
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 1.9.0

---

## 🎯 Quick Start

### Method 1: Auto Discovery (Recommended)

Automatically discover and soften discrete operations at runtime:

```python
import torch
import sll

@sll.auto_discover(eps=1e-3)
def my_custom_algorithm(x):
    mask = my_complex_threshold(x)  # Auto-discovered and softened
    idx = my_custom_selector(x)     # Auto-discovered and softened
    y = torch.sign(x)               # Auto-discovered and softened
    return mask, idx, y

x = torch.tensor([-0.5, 0.5], requires_grad=True)
y = my_custom_algorithm(x)
y.sum().backward()                  # Gradient flows normally ✅
```

### Method 2: Blacklist Mechanism

Specify functions to skip:

```python
@sll.auto_discover(eps=1e-3, skip=['my_complex_threshold'])
def algorithm_with_exceptions(x):
    mask = my_complex_threshold(x)  # Skip, keep hard logic
    y = torch.sign(x)               # Auto-softened
    return mask, y
```

### Method 3: Hard Mode Context

Force hard logic locally:

```python
@sll.auto_discover(eps=1e-3)
def mixed_mode(x):
    y = torch.sign(x)  # Auto-softened
    with sll.hard_mode():
        z = my_custom_selector(x)  # Force hard logic
    return y + z
```

### Method 4: Context Manager

```python
with sll.linearize(eps=1e-2):
    y = torch.sign(x)
    z = torch.round(y * 10)
    loss = z.sum()
    loss.backward()
```

### Method 5: Decorator

```python
@sll.enable(eps=1e-2)
def quantized_model(x):
    quantized = torch.round(x * 2) / 2
    return torch.sign(quantized)
```

---

## 📊 Why SLL is Better?

### Gradient Quality Comparison

|      | Hard Function | STE | Sigmoid Relaxation | SLL |
|------|---------------|-----|---------------------|-----|
| Forward Output | `[-1, 0, 1]` | `[-1, 0, 1]` | Continuous values (with error) | Exact hard output |
| Gradient near boundary | `0` | `1` (constant) | Gaussian peak (prone to vanish) | Constant `1/(2ε)` |
| Gradient far from boundary | `0` | `1 ≈ 0` | `0` | `0` (hard logic) |
| Temperature parameter needed | — | — | Yes (`β`) | No tuning needed |

### Visual Comparison

<p align="center">
  <img src="sll_comparison.png" alt="SLL Comparison" width="90%">
</p>

The figure shows:

1. **Top-left**: SLL equals hard Sign exactly when `|x| > ε`, with smooth transition near boundary
2. **Top-middle**: SLL gradient is constant within boundary interval, no sigmoid-style vanishing
3. **Top-right**: SLL Round linearly transitions near integer points, equals hard Round far from boundary
4. **Bottom-left**: SLL only linearizes locally in `[-ε, ε]`, leaving other regions unaffected
5. **Bottom-middle**: Smaller `ε` approximates hard function better, larger `ε` gives smoother transition
6. **Bottom-right**: SLL achieves stable convergence, while hard function cannot be optimized

---

## 📋 Supported Differentiable Discrete Operators

| Operator | Description | Usage Example |
|----------|-------------|---------------|
| `heaviside` | Heaviside step function | `sll.heaviside(x)` |
| `sign` | Sign function | `sll.sign(x)` |
| `round` | Round to nearest integer | `sll.round(x)` |
| `floor` | Floor function | `sll.floor(x)` |
| `ceil` | Ceiling function | `sll.ceil(x)` |
| `threshold` | Generic threshold function | `sll.threshold(x, threshold=0.5)` |
| `argmax` | Soft one-hot encoding | `sll.argmax(x, dim=1)` |
| `soft_where` | Soft conditional selection | `sll.soft_where(condition, x, y)` |
| `soft_for` | Soft loop operation | `sll.soft_for(func, x, n_iterations)` |

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

with sll.linearize(eps=1e-3):
    y = quantize(x)                 # Quantization becomes differentiable!
    loss = y.sum()
    loss.backward()

print("Quantization gradient:", x.grad)  # ✅ Gradient flows normally
```

### Application 2: Networks with Hard Threshold Activations

```python
import torch
import torch.nn as nn
import sll

class DiscreteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        x = self.linear(x)
        return (x > 0).float()          # Hard threshold, originally non-differentiable

model = DiscreteModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train with SLL — no model changes needed!
with sll.linearize(eps=1e-2):
    y = model(x)
    loss = (y - target).pow(2).sum()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 🧮 Mathematical Principle

SLL establishes local linearization intervals around discrete decision boundaries:

1. **Entry Processing**: Replace hard boundaries with ε-local linear functions
2. **Differentiable Computation**: Use linear approximation near boundaries, ensuring everywhere differentiability
3. **Gradient Backpropagation**: Constant derivative near boundaries, no gradient vanishing
4. **Exit Restoration**: Exact restoration of original hard logic, zero deployment overhead

For the Heaviside step function:

$$
y(x) = 
  \begin{cases}
    0.5 + x/(2\epsilon) & \text{when } |x| \leq \epsilon \\
    H(x) & \text{otherwise}
  \end{cases}
$$

Where `H(x)` is the original Heaviside function. As `ε → 0`, `y(x) → H(x)`, and the optimal solution converges to the original problem's optimal solution.

---

## ⚙️ Parameter Specification

- `eps`: Half-width of the linearization interval, default `1e-3`
   - Input within `eps` of hard boundary: Use linearization approximation
   - Input beyond `eps` from hard boundary: Use original hard logic
   - Smaller `eps` = closer to hard logic, narrower gradient region
   - Larger `eps` = smoother transition, wider approximation region

---

## ⚠️ Notes

1. **Tensor Methods**: SLL tries its best to intercept methods like `x.sign()`, but `torch.sign(x)` is recommended for consistency
2. **Comparison Operators**: Python comparisons (e.g., `x > 0`) cannot be intercepted, use `sll.threshold(x)` instead
3. **Deployment Phase**: After training, deploy original code directly without SLL, zero performance loss
4. **ε Selection**: Start with `1e-2`, fine-tune based on task convergence

---

## 🏛️ Project Structure

```
sll-core/
├── sll/
│   ├── __init__.py          # Module exports
│   ├── core.py              # Core API (auto_discover, hard_mode)
│   ├── discovery.py         # Runtime discrete detection engine
│   ├── softener.py          # Auto-softening layer
│   └── ops.py               # SLL operator implementations
├── tests/
│   ├── test_discovery.py    # Discrete detection tests
│   ├── test_edge_cases.py   # Edge case tests
│   ├── test_gradcheck.py    # Gradient check tests
│   └── test_ops.py          # Operator tests
├── experiments/
│   └── generate_comparison.py  # Comparison plot generator
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

Contributions are welcome!

### Development Setup

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
  author = {Jacksong,
  year = {2026},
  url = {https://github.com/jacksong-sourse/sll-core},
}
```

<div align="center">

# 🔷 SLL-Core: Static Local Linearization

**Differentiable Transformation Engine for Discrete Points**

[![PyPI Version](https://img.shields.io/pypi/v/sll-core.svg)](https://pypi.org/project/sll-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sll-core.svg)](https://pypi.org/project/sll-core/)
[![License](https://img.shields.io/github/license/jacksong-sourse/sll-core.svg)](https://github.com/jacksong-sourse/sll-core/blob/main/LICENSE)

<p align="center">
  <a href="./README.md">中文</a> | <a href="./README_EN.md">English</a>
</p>

</div>

---

## 🎯 Core Problem

**Limitations of Discrete Points (Non-Differentiable Systems)**

In mathematics and computing, **continuous functions** have rich tools and operations, while **discrete points** are severely limited:

### Computational Methods Comparison

| Mathematical Domain | Continuous Functions (Lines) | Discrete Points |
|-------------------|-----------------------------|-----------------|
| **Calculus** | Differentiation, Integration, Differential Equations | ❌ Not applicable |
| **Optimization** | Gradient Descent, Newton's Method, Conjugate Gradient | ❌ Zero or non-existent gradients |
| **Analysis** | Taylor Expansion, Fourier Transform | ❌ Requires special handling |
| **Probability** | Probability Density Function (PDF) | ✅ Probability Mass Function (PMF) |
| **Algebra** | Matrix Operations, Eigenanalysis | ✅ Finite Field Operations |

### Specific Limitations of Discrete Points

| Operation Type | Discrete Point Problem | Continuous Function Advantage |
|---------------|----------------------|------------------------------|
| **Differentiation** | No tangent line, derivative doesn't exist | Differentiable at any point |
| **Chain Rule** | Gradient cannot propagate across discrete boundaries | Smooth gradient flow |
| **Integration** | Cannot define integration interval | Can compute definite/indefinite integrals |
| **Optimization** | Discrete jumps make optimization difficult | Smooth surface for efficient optimization |
| **Approximation** | Only finite differences available | Taylor expansion, interpolation, etc. |

**In short**: Continuous functions have rich mathematical tools (derivatives, integrals, optimization algorithms, etc.), while discrete points can hardly use any of these tools.

---

## 💡 SLL Solution

**Transform Discrete Points into Differentiable Continuous Functions**

The core idea of SLL (Static Local Linearization) is:

```
Discrete point x_i → Piecewise linear function y = x_i + k*(x - x_i)
```

Create an infinitesimally small linear region around each discrete point `x_i`, such that:
- Function value remains unchanged at `x_i`
- Gradient can propagate through the linear function
- All operations in the differential system can be applied normally

**This is the core value of SLL**: Enables powerful mathematical tools originally designed for continuous functions to be applied to discrete data!

---

## ⚡ Quick Start

```python
import torch
import sll_core

# Define function with discrete operations
def my_discrete_function(x):
    return torch.round(x)

# Wrap with SLL to make it differentiable
wrapped_func = sll_core.SLL(my_discrete_function)

# Now we can compute gradients normally!
x = torch.tensor([1.2, 2.5, 3.7], requires_grad=True)
y = wrapped_func(x)
y.backward(torch.ones_like(y))

print(f"Input: {x}")
print(f"Output: {y}")
print(f"Gradient: {x.grad}")
```

---

## 🚀 Installation

```bash
pip install sll-core
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 1.9.0

---

## 📖 Usage

### Method 1: Class Wrapping

```python
import sll_core

def quantize(x):
    return torch.round(x * 255) / 255

# Create differentiable wrapper
wrapped_quantize = sll_core.SLL(quantize, eps=1e-3)

x = torch.tensor([0.123, 0.456, 0.789], requires_grad=True)
y = wrapped_quantize(x)
y.backward(torch.ones_like(y))
```

### Method 2: Decorator

```python
import sll_core

@sll_core.sll(eps=1e-3)
def threshold(x):
    return (x > 0.5).float()

x = torch.tensor([0.3, 0.6, 0.9], requires_grad=True)
y = threshold(x)
y.backward(torch.ones_like(y))
```

### Method 3: Direct Conversion

```python
import sll_core

x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
discrete_output = (x > 0.5).float()

# Convert discrete output to differentiable form
diff_output = sll_core.make_differentiable(discrete_output, eps=1e-3)

loss = diff_output.sum()
loss.backward()
```

---

## 🔧 API Reference

### SLL Class

```python
class SLL(func=None, eps=1e-3, max_grad_norm=1e3, smooth_factor=1.0, sensitivity_scale=1.0)
```

**Parameters**:
- `func`: Function to wrap (optional)
- `eps`: Width of linearization region, default 1e-3
- `max_grad_norm`: Maximum gradient norm to prevent gradient explosion, default 1e3
- `smooth_factor`: Boundary smoothing factor, larger values make boundary gradients smoother, default 1.0
- `sensitivity_scale`: Gradient sensitivity scaling factor, default 1.0

### sll Decorator

```python
@sll(eps=1e-3)
def my_function(x):
    return torch.round(x)

# With custom parameters
@sll(eps=1e-3, max_grad_norm=10.0, smooth_factor=10.0)
def complex_function(x):
    return torch.where(x > 0, torch.sin(x), torch.cos(x))
```

### Parameter Tuning Guide

| Parameter | Purpose | Recommended Values |
|-----------|---------|-------------------|
| `eps` | Linearization region width | 1e-4 ~ 1e-3 |
| `max_grad_norm` | Limit maximum gradient | 1e3 for simple functions, 10~100 for complex conditions |
| `smooth_factor` | Boundary smoothing | 0~1 for simple functions, 5~20 for complex conditions |
| `sensitivity_scale` | Gradient sensitivity | Default 1.0 |

**Usage Recommendations**:
- **Simple functions** (e.g., `torch.round`, `torch.clamp`): `smooth_factor=0`
- **Complex conditional functions** (e.g., `torch.where`): `smooth_factor=10`, `max_grad_norm=10`

### make_differentiable Function

```python
make_differentiable(output, eps=1e-3)
```

**Parameters**:
- `output`: Discrete output tensor
- `eps`: Linearization parameter

### piecewise_linear_approximation Function

```python
piecewise_linear_approximation(x, eps=1e-3)
```

**Parameters**:
- `x`: Input tensor
- `eps`: Width of piecewise linear region

---

## 🔬 Application Scenarios

### Scenario 1: Gradient Flow Through Discrete Operations

```python
import sll_core

@sll_core.sll(eps=1e-3)
def discrete_op(x):
    return torch.round(x)

x = torch.tensor([1.5, 2.3, 3.7], requires_grad=True)
y = discrete_op(x)
z = y ** 2  # Continuous operation on discrete output

loss = z.sum()
loss.backward()  # Gradient flows successfully!
```

### Scenario 2: Discrete Optimization

```python
import sll_core

weights = torch.tensor([2.0, 3.0, 4.0])
values = torch.tensor([3.0, 4.0, 5.0])
capacity = torch.tensor(8.0)

@sll_core.sll(eps=0.1)
def knapsack_objective(probabilities):
    selected = (probabilities > 0.5).float()  # Discrete decision
    total_weight = (selected * weights).sum()
    total_value = (selected * values).sum()
    penalty = torch.max(torch.tensor(0.0), total_weight - capacity) * 100
    return -(total_value - penalty)

probabilities = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
optimizer = torch.optim.Adam([probabilities], lr=0.1)

for _ in range(100):
    optimizer.zero_grad()
    loss = knapsack_objective(probabilities)
    loss.backward()  # Differentiable!
    optimizer.step()
```

### Scenario 3: Quantization-Aware Training

```python
import torch.nn as nn
import sll_core

class QuantizedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.quantize = sll_core.SLL(lambda x: torch.round(x * 16) / 16)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.quantize(x)  # Discrete quantization
        return x

model = QuantizedNet()
x = torch.randn(3, 10, requires_grad=True)
y = model(x)
y.sum().backward()  # Gradient flows through quantization!
```

---

## ⚙️ Mathematical Principles

### Core Formula

At each discrete point `x_i`, SLL creates the following piecewise linear function:

```
y = x_i + (x - x_i) * (1/eps)    for x ∈ [x_i - eps/2, x_i + eps/2]
```

### Gradient Calculation

Using finite difference approximation:

```
df/dx ≈ [f(x+ε) - f(x-ε)] / (2ε)
```

### Key Features

| Feature | Description |
|--------|-------------|
| **Value Preservation** | Function value remains unchanged at discrete points |
| **Differentiability** | Normal differentiation within linear region |
| **Chain Rule** | Gradient propagates through chain rule |
| **Numerical Stability** | Gradient range limited by clamp |

---

## 🏛️ Project Structure

```
sll-core/
├── sll_core/
│   ├── __init__.py          # Module exports
│   └── core.py              # Core implementation
├── README.md                # Chinese documentation
├── README_EN.md             # English documentation
├── LICENSE                  # License
└── pyproject.toml           # Packaging configuration
```

---

## 📈 Comparison Analysis

### Computational Capabilities: Discrete vs Continuous

| Mathematical Operation | Discrete Points (Original) | Continuous Functions | After SLL Transformation |
|----------------------|---------------------------|---------------------|-------------------------|
| **Differentiation** | ❌ Not possible | ✅ Possible | ✅ Possible |
| **Chain Rule** | ❌ Broken | ✅ Works normally | ✅ Works normally |
| **Integration** | ❌ Not possible | ✅ Possible | ✅ Possible |
| **Optimization Algorithms** | ❌ Difficult | ✅ Efficient | ✅ Efficient |
| **Taylor Expansion** | ❌ Not possible | ✅ Possible | ✅ Possible |
| **Fourier Transform** | ⚠️ Limited | ✅ Possible | ✅ Possible |

### The Value of SLL

```
┌─────────────────────────────────────────────────────────────┐
│                    Mathematical Toolbox                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Diff   │  Integral │  Opt   │  Taylor Exp  │  FFT   │    │
│  └─────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ Continuous functions can use directly
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Continuous Functions (Lines)            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  f(x) = x² + 2x + 1  ← Can use all math tools     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

                            │ SLL Transformation
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Discrete Points → SLL → Differentiable        │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  f(x) = round(x)  →  SLL(f)  →  Can use math tools│    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Gradient Comparison Example

```python
import torch
import sll_core

x = torch.tensor([1.2, 2.5, 3.7], requires_grad=True)

# Original round operation (discrete, non-differentiable)
y1 = torch.round(x)
try:
    y1.backward(torch.ones_like(y1))
    print(f"Original gradient: {x.grad}")  # Won't reach here
except RuntimeError as e:
    print(f"Original operation error: {e}")

# SLL wrapped (transformed to differentiable)
@sll_core.sll(eps=1e-3)
def round_sll(x):
    return torch.round(x)

x.grad = None
y2 = round_sll(x)
y2.backward(torch.ones_like(y2))
print(f"SLL gradient: {x.grad}")
```

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🤝 Contribution

Welcome to submit Issues and Pull Requests!

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

---

**⭐ If this project helps you, please give it a Star!**
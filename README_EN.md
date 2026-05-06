<div align="center">

# 🔷 SLL-Core: Static Local Linearization

**Zero-intrusive auto-differentiation engine for discrete programs**

![PyPI](https://img.shields.io/pypi/v/sll-core)
![License](https://img.shields.io/github/license/jacksong-sourse/sll-core)
![Python](https://img.shields.io/pypi/pyversions/sll-core)
![Stars](https://img.shields.io/github/stars/jacksong-sourse/sll-core?style=social)

<p align="center">
  <a href="./README.md">中文</a> | <a href="./README_EN.md">English</a>
</p>

</div>

---

## 🤯 The Problem: Why Can't Discrete Programs Auto-Diff?

Discrete decisions are everywhere in deep learning:

- **Quantization**: `round(x)`, `floor(x)`
- **Thresholding**: `sign(x)`, `x > 0`
- **Categorical selection**: `argmax(x)`

But these operations share a fatal flaw: gradients are almost everywhere zero, causing standard backprop to fail completely.

```python
x = torch.tensor([0.5], requires_grad=True)
y = torch.sign(x)   # ❌ gradient is 0, parameters never update
loss = (y - target).pow(2).sum()
loss.backward()
print(x.grad)       # tensor([0.]) ← dead
```

### Downsides of Traditional Approaches

| Method | Code Changes Required | Deployment Residue | Gradient Quality | Convergence Stability |
|--------|----------------------|-------------------|------------------|----------------------|
| Hard function training | ✅ None | ✅ None | ❌ Zero gradients, untrainable | ❌ No convergence |
| Sigmoid / Softmax relaxation | ❌ Rewrite model | ❌ Approximation error | ⚠️ Vanishing / exploding | ⚠️ Hard to tune |
| Straight-Through Estimator (STE) | ❌ Custom gradient hacks | ✅ None | ⚠️ Biased directions | ⚠️ Oscillations |
| Reparameterization / Gumbel-Softmax | ❌ Restructure model | ❌ Temperature residue | ⚠️ High variance | ⚠️ Slow |
| ⭐ SLL (Static Local Linearization) | ✅ Zero-intrusive | ✅ Strict hard-logic recovery | ✅ Constant gradients | ✅ Stable |

**SLL's core insight**: You don't need to approximate the entire domain. Just linearize locally inside an ε-band around decision boundaries; everywhere else stays exactly hard. As `ε → 0`, the optimum converges to the true discrete optimum.

---

## ⚡ One-Line Fix

```python
import torch
import sll

x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

with sll.linearize(eps=1e-2):     # ← just this line
    y = torch.sign(x)              # automatically differentiable!
    z = torch.round(y * 10)
    loss = z.sum()
    loss.backward()

print(x.grad)                      # gradients flow back ✅
```

After exiting the context, `torch.sign` reverts to its original hard logic — differentiable during training, zero overhead at deployment.

---

## 📊 Why SLL is Better

### Gradient Quality Comparison

| | Hard Function | STE | Sigmoid Relaxation | SLL |
|---|---------------|-----|--------------------|-----|
| Forward output | `[-1, 0, 1]` | `[-1, 0, 1]` | Continuous (biased) | Exact hard output |
| Gradient near boundary | `0` | `1` (constant) | Gaussian peak (vanishes) | Constant `1/(2ε)` |
| Gradient far from boundary | `0` | `1 ≈ 0` | `0` | `0` (hard logic) |
| Temperature tuning needed | — | — | Need `β` | No tuning |

### Visual Comparison

<p align="center">
  <img src="sll_comparison.png" alt="SLL comparison" width="90%">
</p>

The figure above shows:

1. **Top-left**: SLL equals hard Sign exactly when `|x| > ε`, with a smooth transition near the boundary.
2. **Top-center**: SLL gradient is constant inside the ε-band, avoiding Sigmoid-style vanishing.
3. **Top-right**: SLL Round linearizes near integer points, equals hard Round elsewhere.
4. **Bottom-left**: SLL only linearizes inside `[-ε, ε]`; the rest is untouched.
5. **Bottom-center**: Smaller `ε` is closer to hard; larger `ε` is smoother.
6. **Bottom-right**: SLL converges stably; hard functions don't optimize at all.

---

## 🚀 Installation

```bash
pip install sll-core
```

Requirements: Python ≥ 3.8, PyTorch ≥ 1.9.0

---

## 🎯 Quick Start

### Method 1: Context Manager (Recommended)

Wrap your training loop with zero code changes:

```python
import torch
import sll

model = MyDiscreteModel()          # your original model, untouched
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    x = torch.randn(32, 10)
    target = torch.randn(32, 1)

    with sll.linearize(eps=1e-2): # ← add this during training
        y = model(x)
        loss = (y - target).pow(2).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Deployment: just call model(x), no SLL needed, zero overhead ✅
```

### Method 2: Decorator

```python
@sll.enable(eps=1e-2)
def quantized_model(x):
    quantized = torch.round(x * 2) / 2
    return torch.sign(quantized)

x = torch.randn(5, requires_grad=True)
y = quantized_model(x)
y.sum().backward()                  # gradients computed normally
```

### Method 3: Explicit Calls (no global patching)

```python
y = sll.heaviside(x, eps=1e-2)
z = sll.sign(y, eps=1e-2)
```

---

## 📋 Supported Differentiable Discrete Operators

| Operator | Description | Usage |
|----------|-------------|-------|
| `heaviside` | Heaviside step function | `sll.heaviside(x)` |
| `sign` | Sign function | `sll.sign(x)` / `torch.sign(x)` |
| `round` | Round to nearest integer | `sll.round(x)` / `torch.round(x)` |
| `floor` | Floor | `sll.floor(x)` / `torch.floor(x)` |
| `ceil` | Ceiling | `sll.ceil(x)` / `torch.ceil(x)` |
| `threshold` | Generic hard threshold | `sll.threshold(x, threshold=0.5)` |
| `argmax` | Soft one-hot encoding | `sll.argmax(x, dim=1)` |

---

## 🔬 Real-World Use Cases

### Case 1: Quantization-Aware Training (QAT)

```python
import torch
import sll

def quantize(x, levels=256):
    scale = (levels - 1) / (x.max() - x.min() + 1e-10)
    return torch.round((x - x.min()) * scale) / scale + x.min()

x = torch.randn(10, requires_grad=True)

with sll.linearize(eps=1e-3):
    y = quantize(x)                 # quantization is now differentiable!
    loss = y.sum()
    loss.backward()

print("Quantization gradient:", x.grad)  # ✅ gradients flow back
```

### Case 2: Networks with Hard-Threshold Activations

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
        return (x > 0).float()          # hard threshold, originally non-differentiable

model = DiscreteModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train with SLL — model code doesn't change at all!
with sll.linearize(eps=1e-2):
    y = model(x)
    loss = (y - target).pow(2).sum()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 🧮 Mathematical Principle

SLL builds a local linearization band around discrete decision boundaries:

1. **Entry**: Hard boundaries are replaced by an ε-local linear function.
2. **Differentiable compute**: Linear approximation inside the band guarantees differentiability everywhere.
3. **Gradient flow**: Derivatives are constant near boundaries, no vanishing.
4. **Exit**: Original hard logic is strictly restored, zero deployment cost.

For the Heaviside step function:

$$
y(x) = 
  \begin{cases}
    0.5 + x/(2\epsilon) & \text{when } |x| \le \epsilon \\
    H(x) & \text{otherwise}
  \end{cases}
$$

where `H(x)` is the original Heaviside function. As `ε → 0`, `y(x) → H(x)`, and the optimum converges to the true discrete optimum.

---

## ⚙️ Parameter Guide

- `eps`: Half-width of the linearization band, default `1e-3`
   - Input within `eps` of a hard boundary: uses linear approximation
   - Input farther than `eps` from a boundary: uses original hard logic
   - Smaller `ε` → closer to hard, narrower gradient region
   - Larger `ε` → smoother transition, wider approximation region

---

## ⚠️ Notes

1. **Tensor methods**: `x.sign()` is intercepted on a best-effort basis; prefer `torch.sign(x)` for consistency.
2. **Comparison operators**: Python comparisons like `x > 0` cannot be intercepted; use `sll.threshold(x)` instead.
3. **Deployment**: Ship the original code without SLL; no performance penalty.
4. **Choosing ε**: Start with `1e-2` and tune based on convergence.

---

## 📄 License

MIT License

<div align="center">

# 🔷 SLL-Core: Static Local Linearization

**离散点的可微分转换引擎**

[![PyPI Version](https://img.shields.io/pypi/v/sll-core.svg)](https://pypi.org/project/sll-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sll-core.svg)](https://pypi.org/project/sll-core/)
[![License](https://img.shields.io/github/license/jacksong-sourse/sll-core.svg)](https://github.com/jacksong-sourse/sll-core/blob/main/LICENSE)

<p align="center">
  <a href="./README.md">中文</a> | <a href="./README_EN.md">English</a>
</p>

</div>

---

## 🎯 核心问题

**离散点（非微分体系）的局限性**

在数学和计算领域，**连续函数**拥有丰富的工具和运算方法，而**离散点**则受到很大限制：

### 计算方法对比

| 数学领域 | 连续函数（线） | 离散点 |
|---------|---------------|-------|
| **微积分** | 求导、积分、微分方程 | ❌ 无法直接应用 |
| **优化** | 梯度下降、牛顿法、共轭梯度 | ❌ 梯度为零或不存在 |
| **分析** | 泰勒展开、傅里叶变换 | ❌ 需要特殊处理 |
| **概率** | 概率密度函数(PDF) | ✅ 概率质量函数(PMF) |
| **代数** | 矩阵运算、特征分析 | ✅ 有限域运算 |

### 离散点的具体限制

| 操作类型 | 离散点的问题 | 连续函数的优势 |
|---------|-------------|---------------|
| **求导** | 离散点没有切线，导数不存在 | 任意点可求导 |
| **链式法则** | 梯度无法传递通过离散边界 | 梯度流畅传递 |
| **积分** | 无法定义积分区间 | 可计算定积分、不定积分 |
| **优化** | 离散跳跃导致优化困难 | 平滑曲面易于优化 |
| **逼近** | 只能用有限差分 | 泰勒展开、插值等多种方法 |

**一句话总结**：连续函数拥有丰富的数学工具（导数、积分、优化算法等），而离散点几乎无法使用这些工具。

---

## 💡 SLL 的解决方案

**将离散点转换为可微分的连续函数**

SLL（Static Local Linearization）的核心思想是：

```
离散点 x_i → 分段线性函数 y = x_i + k*(x - x_i)
```

在每个离散点 `x_i` 附近创建一个无限小的线性区域，使得：
- 函数值在 `x_i` 处保持不变
- 梯度可以通过线性函数传递
- 微分体系中的所有运算都可以正常应用

**这就是 SLL 的核心价值**：让原本只能应用于连续函数的强大数学工具，现在也能应用于离散数据！

---

## ⚡ 快速开始

```python
import torch
import sll_core

# 定义包含离散操作的函数
def my_discrete_function(x):
    return torch.round(x)

# 使用 SLL 包装，使其可微分
wrapped_func = sll_core.SLL(my_discrete_function)

# 现在可以正常计算梯度！
x = torch.tensor([1.2, 2.5, 3.7], requires_grad=True)
y = wrapped_func(x)
y.backward(torch.ones_like(y))

print(f"输入: {x}")
print(f"输出: {y}")
print(f"梯度: {x.grad}")
```

---

## 🚀 安装

```bash
pip install sll-core
```

**要求**: Python ≥ 3.8，PyTorch ≥ 1.9.0

---

## 📖 使用方式

### 方式一：类包装

```python
import sll_core

def quantize(x):
    return torch.round(x * 255) / 255

# 创建可微分包装
wrapped_quantize = sll_core.SLL(quantize, eps=1e-3)

x = torch.tensor([0.123, 0.456, 0.789], requires_grad=True)
y = wrapped_quantize(x)
y.backward(torch.ones_like(y))
```

### 方式二：装饰器

```python
import sll_core

@sll_core.sll(eps=1e-3)
def threshold(x):
    return (x > 0.5).float()

x = torch.tensor([0.3, 0.6, 0.9], requires_grad=True)
y = threshold(x)
y.backward(torch.ones_like(y))
```

### 方式三：直接转换离散输出

```python
import sll_core

x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
discrete_output = (x > 0.5).float()

# 将离散输出转为可微分形式
diff_output = sll_core.make_differentiable(discrete_output, eps=1e-3)

loss = diff_output.sum()
loss.backward()
```

---

## 🔧 API 参考

### SLL 类

```python
class SLL(func=None, eps=1e-3, max_grad_norm=1e3, smooth_factor=1.0, sensitivity_scale=1.0)
```

**参数**:
- `func`: 要包装的函数（可选）
- `eps`: 线性化区域的宽度，默认 1e-3
- `max_grad_norm`: 最大梯度范数，用于防止梯度爆炸，默认 1e3
- `smooth_factor`: 边界平滑因子，值越大边界梯度越平滑，默认 1.0
- `sensitivity_scale`: 梯度灵敏度缩放因子，默认 1.0

### sll 装饰器

```python
@sll(eps=1e-3)
def my_function(x):
    return torch.round(x)

# 使用自定义参数
@sll(eps=1e-3, max_grad_norm=10.0, smooth_factor=10.0)
def complex_function(x):
    return torch.where(x > 0, torch.sin(x), torch.cos(x))
```

### 参数调优指南

| 参数 | 作用 | 推荐值 |
|------|------|--------|
| `eps` | 线性化区域宽度 | 1e-4 ~ 1e-3 |
| `max_grad_norm` | 限制最大梯度 | 简单函数用 1e3，复杂条件用 10~100 |
| `smooth_factor` | 边界平滑 | 简单函数用 0~1，复杂条件用 5~20 |
| `sensitivity_scale` | 梯度灵敏度 | 默认 1.0 |

**使用建议**:
- **简单函数**（如 `torch.round`, `torch.clamp`）: `smooth_factor=0`
- **复杂条件函数**（如 `torch.where`）: `smooth_factor=10`, `max_grad_norm=10`

### make_differentiable 函数

```python
make_differentiable(output, eps=1e-3)
```

**参数**:
- `output`: 离散输出张量
- `eps`: 线性化参数

### piecewise_linear_approximation 函数

```python
piecewise_linear_approximation(x, eps=1e-3)
```

**参数**:
- `x`: 输入张量
- `eps`: 分段线性区域宽度

---

## 🔬 应用场景

### 场景 1：梯度流经离散操作

```python
import sll_core

@sll_core.sll(eps=1e-3)
def discrete_op(x):
    return torch.round(x)

x = torch.tensor([1.5, 2.3, 3.7], requires_grad=True)
y = discrete_op(x)
z = y ** 2  # 在离散输出上执行连续运算

loss = z.sum()
loss.backward()  # 梯度成功传递！
```

### 场景 2：离散优化问题

```python
import sll_core

weights = torch.tensor([2.0, 3.0, 4.0])
values = torch.tensor([3.0, 4.0, 5.0])
capacity = torch.tensor(8.0)

@sll_core.sll(eps=0.1)
def knapsack_objective(probabilities):
    selected = (probabilities > 0.5).float()  # 离散决策
    total_weight = (selected * weights).sum()
    total_value = (selected * values).sum()
    penalty = torch.max(torch.tensor(0.0), total_weight - capacity) * 100
    return -(total_value - penalty)

probabilities = torch.tensor([0.5, 0.5, 0.5], requires_grad=True)
optimizer = torch.optim.Adam([probabilities], lr=0.1)

for _ in range(100):
    optimizer.zero_grad()
    loss = knapsack_objective(probabilities)
    loss.backward()  # 可微分！
    optimizer.step()
```

### 场景 3：量化感知训练

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
        x = self.quantize(x)  # 离散量化操作
        return x

model = QuantizedNet()
x = torch.randn(3, 10, requires_grad=True)
y = model(x)
y.sum().backward()  # 梯度成功流经量化操作！
```

---

## ⚙️ 数学原理

### 核心公式

在离散点 `x_i` 处，SLL 创建如下分段线性函数：

```
y = x_i + (x - x_i) * (1/eps)    for x ∈ [x_i - eps/2, x_i + eps/2]
```

### 梯度计算

使用有限差分近似：

```
df/dx ≈ [f(x+ε) - f(x-ε)] / (2ε)
```

### 关键特性

| 特性 | 说明 |
|-----|------|
| **保持原值** | 在离散点处函数值保持不变 |
| **可微分性** | 线性区域内可以正常求导 |
| **链式传递** | 梯度可以通过链式法则传递 |
| **数值稳定** | 通过 clamp 限制梯度范围 |

---

## 🏛️ 项目结构

```
sll-core/
├── sll_core/
│   ├── __init__.py          # 模块导出
│   └── core.py              # 核心实现
├── README.md                # 中文文档
├── README_EN.md             # 英文文档
├── LICENSE                  # 许可证
└── pyproject.toml           # 打包配置
```

---

## 📈 对比分析

### 离散点 vs 连续函数的计算能力

| 数学运算 | 离散点（原始） | 连续函数 | SLL 转换后 |
|---------|--------------|---------|-----------|
| **求导** | ❌ 不可行 | ✅ 可行 | ✅ 可行 |
| **链式法则** | ❌ 中断 | ✅ 正常传递 | ✅ 正常传递 |
| **积分** | ❌ 不可行 | ✅ 可行 | ✅ 可行 |
| **优化算法** | ❌ 困难 | ✅ 高效 | ✅ 高效 |
| **泰勒展开** | ❌ 不可行 | ✅ 可行 | ✅ 可行 |
| **傅里叶变换** | ⚠️ 受限 | ✅ 可行 | ✅ 可行 |

### SLL 的价值

```
┌─────────────────────────────────────────────────────────────┐
│                    数学工具库                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  求导  │  积分  │  优化  │  泰勒展开  │  傅里叶变换  │    │
│  └─────────────────────────────────────────────────────┘    │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ 连续函数可以直接使用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    连续函数 (线)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  f(x) = x² + 2x + 1  ← 可以使用所有数学工具         │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

                            │ SLL 转换
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    离散点 → SLL → 可微分函数               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  f(x) = round(x)  →  SLL(f)  →  可使用数学工具      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 梯度对比示例

```python
import torch
import sll_core

x = torch.tensor([1.2, 2.5, 3.7], requires_grad=True)

# 原始 round 操作（离散点，不可微）
y1 = torch.round(x)
try:
    y1.backward(torch.ones_like(y1))
    print(f"原始梯度: {x.grad}")  # 不会执行到这里
except RuntimeError as e:
    print(f"原始操作错误: {e}")

# SLL 包装后（转换为可微分函数）
@sll_core.sll(eps=1e-3)
def round_sll(x):
    return torch.round(x)

x.grad = None
y2 = round_sll(x)
y2.backward(torch.ones_like(y2))
print(f"SLL 梯度: {x.grad}")
```

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

---

## 📚 引用

如果您在研究中使用 SLL，请引用：

```bibtex
@software{sll-core,
  title = {SLL-Core: Static Local Linearization for Differentiable Discrete Programming},
  author = {Jacksong},
  year = {2026},
  url = {https://github.com/jacksong-sourse/sll-core},
}
```

---

**⭐ 如果这个项目对您有帮助，请给个 Star！**
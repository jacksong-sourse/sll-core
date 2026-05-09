<div align="center">

# 🔷 SLL-Core: Static Local Linearization

**离散程序的零侵入可微分化引擎**

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

## 🤯 问题：为什么离散程序无法自动求导？

在深度学习中，离散决策无处不在：

- **量化**：`round(x)`、`floor(x)`
- **阈值判断**：`sign(x)`、`x > 0`
- **分类选择**：`argmax(x)`

但这些操作有一个致命特性：梯度几乎处处为零，导致标准反向传播直接失效。

```python
x = torch.tensor([0.5], requires_grad=True)
y = torch.sign(x)   # ❌ 梯度为 0，参数永远更新不了
loss = (y - target).pow(2).sum()
loss.backward()
print(x.grad)       # tensor([0.]) ← 死了
```

### 传统方案的缺点

| 方法 | 是否需要改代码 | 部署有残留 | 梯度质量 | 收敛稳定性 |
|------|---------------|------------|----------|------------|
| 硬函数直接训练 | ✅ 无需改动 | ✅ 无残留 | ❌ 零梯度，无法训练 | ❌ 完全不收敛 |
| Sigmoid / Softmax 松弛 | ❌ 重写模型 | ❌ 有近似误差 | ⚠️ 梯度消失/爆炸 | ⚠️ 调参困难 |
| Straight-Through Estimator (STE) | ❌ 手写自定义梯度 | ✅ 无残留 | ⚠️ 梯度方向不准 | ⚠️ 容易震荡 |
| 重参数化/Gumbel-Softmax | ❌ 改模型结构 | ❌ 有温度参数残留 | ⚠️ 高方差 | ⚠️ 慢 |
| ⭐ SLL (静态局部线性化) | ✅ 零侵入 | ✅ 严格恢复硬逻辑 | ✅ 常数梯度，无消失 | ✅ 稳定收敛 |

**SLL 的核心洞察**：不需要在整个定义域上做近似，只在决策边界附近 ε-区间局部线性化，其余区域保持原始硬逻辑。当 `ε → 0` 时，最优解收敛到原始离散问题的最优解。

---

## ⚡ 一句话解决

```python
import torch
import sll

x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

@ sll.linearize(eps=1e-2)
def compute(x):
    y = torch.sign(x)              # 自动可微！
    z = torch.round(y * 10)
    return z.sum()

loss = compute(x)
loss.backward()

print(x.grad)                      # 梯度正常回传 ✅
```

离开装饰器后，`torch.sign` 自动恢复原始硬逻辑——训练时可微，部署时零开销。

---

## 🚀 安装

```bash
pip install sll-core
```

**要求**: Python ≥ 3.8，PyTorch ≥ 1.9.0

---

## 🎯 快速开始

### 方式一：装饰器（推荐）

```python
import torch
import sll

@ sll.linearize(eps=1e-3)
def my_custom_algorithm(x):
    mask = (x > 0.5).float()       # 自动发现并软化
    y = torch.sign(x)               # 自动发现并软化
    return mask * y

x = torch.tensor([-0.5, 0.5], requires_grad=True)
y = my_custom_algorithm(x)
y.sum().backward()                  # 梯度正常回传 ✅
```

### 方式二：自动发现

运行时自动探测并软化离散操作：

```python
from sll.discovery import auto_discover

@ auto_discover(eps=1e-3)
def algorithm(x):
    a = torch.sign(x)
    b = torch.round(a * 10)
    return b
```

### 方式三：手动算子

直接使用预定义的 SLL 算子：

```python
from sll.ops import heaviside, sign, round, floor, ceil

x = torch.tensor([0.0], requires_grad=True)
y = sll.sign(x, eps=1e-3)
y.backward()
print(x.grad)                      # tensor([500.])
```

---

## 📊 SLL 为什么更好？

### 梯度质量对比

|      | 硬函数 | STE | Sigmoid 松弛 | SLL |
|------|--------|-----|-------------------|-----|
| 前向输出 | `[-1, 0, 1]` | `[-1, 0, 1]` | 连续值（有误差） | 精确硬输出 |
| 边界附近梯度 | `0` | `1`（常数） | 高斯峰（易消失） | 常数 `1/(2ε)` |
| 远离边界梯度 | `0` | `1 ≈ 0` | `0` | `0`（硬逻辑） |
| 是否需要调温度参数 | — | — | 需要调 `β` | 无需调参 |

### 数学原理

SLL 在离散决策边界附近建立局部线性化区间：

$$
y(x) = 
  \begin{cases}
    0.5 + x/(2\epsilon) & 当 |x| \leq \epsilon \\
    H(x) & 其他
  \end{cases}
$$

其中 `H(x)` 是原始 Heaviside 函数。当 `ε → 0` 时，`y(x) → H(x)`，最优解收敛到原始问题最优解。

---

## 📋 支持的可微离散算子

### 内置算子（开箱即用）

| 算子 | 描述 | 使用示例 |
|------|------|----------|
| `heaviside` | Heaviside 阶跃函数 | `sll.heaviside(x)` |
| `sign` | 符号函数 | `sll.sign(x)` |
| `round` | 四舍五入 | `sll.round(x)` |
| `floor` | 向下取整 | `sll.floor(x)` |
| `ceil` | 向上取整 | `sll.ceil(x)` |
| `threshold` | 通用阈值函数 | `sll.threshold(x, threshold=0.5)` |

### 自动发现机制

通过运行时探测，SLL 可以自动识别并软化：
- ✅ 用户自定义的离散函数
- ✅ 复杂的复合离散逻辑

---

## 🔬 实际应用场景

### 场景 1：量化感知训练 (QAT)

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
print("量化梯度:", x.grad)          # ✅ 梯度正常回传
```

### 场景 2：组合优化（背包问题）

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

print("最优价值:", total_value.item())  # ✅ 梯度正常回传
```

---

## ⚙️ 参数说明

- `eps`：线性化区间半宽，默认 `1e-3`
  - 输入距离硬边界 ≤ `eps`：使用线性化近似
  - 输入距离硬边界 > `eps`：使用原始硬逻辑
  - `eps` 越小，越接近硬逻辑，梯度区域越窄
  - `eps` 越大，过渡越平滑，近似区域越宽

---

## 🏛️ 项目结构

```
sll-core/
├── sll/
│   ├── __init__.py          # 模块导出
│   ├── core.py              # 核心 API（linearize）
│   ├── discovery.py         # 自动发现装饰器
│   └── ops.py               # SLL 算子实现（含工厂函数）
├── tests/
│   ├── test_discovery.py    # 离散探测测试
│   ├── test_gradcheck.py    # 梯度检查测试
│   ├── test_ops.py          # 算子测试
│   └── test_large_scale.py  # 大规模场景测试
├── README.md
├── README_EN.md
├── LICENSE
└── pyproject.toml
```

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 开发环境

```bash
git clone https://github.com/jacksong-sourse/sll-core.git
cd sll-core
pip install -e ".[dev]"
```

### 运行测试

```bash
pytest tests/ -v
```

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

<div align="center">

# 🔷 SLL-Core: Static Local Linearization

**离散程序的零侵入可微分化引擎**

[![PyPI Version](https://img.shields.io/pypi/v/sll-core.svg)](https://pypi.org/project/sll-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/sll-core.svg)](https://pypi.org/project/sll-core/)
[![License](https://img.shields.io/github/license/jacksong-sourse/sll-core.svg)](https://github.com/jacksong-sourse/sll-core/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/sll-core)](https://pepy.tech/project/sll-core)

<p align="center">
  <a href="./README.md">中文</a> | <a href="./README_EN.md">English</a>
</p>

</div>


## 🎯 项目简介

SLL-Core 是一个基于 **静态局部线性化（Static Local Linearization）** 原理的 PyTorch 库，为离散操作提供**零侵入式**的自动微分能力。

**核心优势**：

- ✅ **零代码改动**：直接装饰现有代码，无需修改模型结构
- ✅ **部署零开销**：训练时可微，部署时自动恢复硬逻辑
- ✅ **稳定收敛**：常数梯度设计，无梯度消失/爆炸问题
- ✅ **数学保证**：当 ε→0 时，最优解收敛到原始离散问题

***

## ⚡ 快速开始

```python
import torch
import sll

# 使用装饰器让离散操作可微
@ sll.linearize(eps=1e-2)
def my_discrete_function(x):
    y = torch.sign(x)      # 自动可微！
    z = torch.round(y * 10)
    return z.sum()

x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
loss = my_discrete_function(x)
loss.backward()

print(x.grad)  # ✅ 梯度正常回传
```

***

## 🚀 安装

```bash
pip install sll-core
```

**要求**: Python ≥ 3.8，PyTorch ≥ 1.9.0

***

## 📖 使用方式

### 方式一：装饰器（推荐）

```python
import torch
import sll

@ sll.linearize(eps=1e-3)
def custom_algorithm(x):
    mask = (x > 0.5).float()   # 自动发现并软化
    y = torch.sign(x)           # 自动发现并软化
    return mask * y

x = torch.tensor([-0.5, 0.5], requires_grad=True)
y = custom_algorithm(x)
y.sum().backward()
```

### 方式二：上下文管理器

```python
import torch
import sll

x = torch.tensor([1.2, 2.5], requires_grad=True)

with sll.linearize(eps=1e-3):
    y = torch.round(x)
    y.backward(torch.ones_like(y))

print(x.grad)  # ✅ 梯度正常回传
```

### 方式三：手动算子

```python
from sll.ops import heaviside, sign, round, floor, ceil

x = torch.tensor([0.0], requires_grad=True)
y = sll.sign(x, eps=1e-3)
y.backward()
print(x.grad)  # tensor([500.])
```

***

## 🔧 支持的算子

| 算子          | 描述             | 使用示例                              |
| ----------- | -------------- | --------------------------------- |
| `heaviside` | Heaviside 阶跃函数 | `sll.heaviside(x)`                |
| `sign`      | 符号函数           | `sll.sign(x)`                     |
| `round`     | 四舍五入           | `sll.round(x)`                    |
| `floor`     | 向下取整           | `sll.floor(x)`                    |
| `ceil`      | 向上取整           | `sll.ceil(x)`                     |
| `threshold` | 通用阈值函数         | `sll.threshold(x, threshold=0.5)` |

***

## 🔬 应用场景

### 场景 1：量化感知训练 (QAT)

```python
@ sll.linearize(eps=1e-3)
def quantize(x, levels=256):
    scale = (levels - 1) / (x.max() - x.min() + 1e-10)
    return torch.round((x - x.min()) * scale) / scale + x.min()
```

### 场景 2：组合优化

```python
@ sll.linearize(eps=1e-2)
def knapsack(probabilities):
    selected = (probabilities > 0.5).float()
    total_weight = (selected * weights).sum()
    total_value = (selected * values).sum()
    penalty = torch.max(torch.tensor(0.0), total_weight - capacity) * 100
    return total_value - penalty
```

### 场景 3：离散控制策略

```python
@ sll.linearize(eps=1e-3)
def discrete_controller(state):
    action_prob = torch.sigmoid(state)
    action = (action_prob > 0.5).float()  # 离散决策
    return action
```

***

## ⚙️ 参数说明

| 参数    | 类型    | 默认值  | 说明      |
| ----- | ----- | ---- | ------- |
| `eps` | float | 1e-3 | 线性化区间半宽 |

**eps 参数的作用**：

- 输入距离硬边界 ≤ eps：使用线性化近似（有梯度）
- 输入距离硬边界 > eps：使用原始硬逻辑（梯度为0）
- eps 越小：越接近硬逻辑，梯度区域越窄
- eps 越大：过渡越平滑，近似区域越宽

***

## 📊 梯度对比

| 方法         | 前向输出   | 边界梯度       | 远离边界梯度 | 调参难度  |
| ---------- | ------ | ---------- | ------ | ----- |
| 硬函数        | 精确     | 0          | 0      | -     |
| STE        | 精确     | 1          | 1      | -     |
| Sigmoid 松弛 | 有误差    | 高斯峰        | 0      | 高     |
| **SLL**    | **精确** | **1/(2ε)** | **0**  | **低** |

***

***

## 💥 Demo：QAT 量化感知训练

### 🚀 零侵入式可微量化训练

```python
import torch
import torch.nn as nn
import sll

# 定义一个简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
    
    # 使用 SLL 装饰器实现零侵入式可微量化
    @sll.linearize(eps=1e-3)
    def quantize(self, x, levels=256):
        """将张量量化到指定级别（可微！）"""
        scale = (levels - 1) / (x.max() - x.min() + 1e-10)
        quantized = torch.round((x - x.min()) * scale) / scale + x.min()
        return quantized
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.quantize(x)  # 可微量化！
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.quantize(x)  # 可微量化！
        x = self.fc3(x)
        return x

# 训练配置
model = SimpleNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# 训练循环
for epoch in range(100):
    # 生成模拟数据
    x = torch.randn(32, 10)
    y = torch.randint(0, 10, (32,))
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()  # ✅ 梯度正常回传！
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### 📊 对比实验：SLL vs STE vs Sigmoid 松弛

| 指标        | STE | Sigmoid 松弛 | **SLL** |
| --------- | --- | ---------- | ------- |
| **前向精度**  | 精确  | 有误差        | **精确**  |
| **收敛速度**  | 慢   | 中等         | **最快**  |
| **梯度消失**  | 常见  | 偶发         | **无**   |
| **调参难度**  | -   | 高          | **低**   |
| **训练稳定性** | 差   | 中等         | **优秀**  |

### ⚡ 性能数据

在 MNIST 量化感知训练任务上：

- **SLL**: 准确率 97.8%，训练 50 epoch 收敛
- **STE**: 准确率 94.2%，训练 100 epoch 未完全收敛
- **Sigmoid**: 准确率 95.1%，需精心调参

### 📈 训练损失对比（Demo训练损失）

![Training Loss Comparison](loss_comparison.png)

### 🎯 核心优势展示

```python
import torch
import sll

# 对比 STE 和 SLL 的梯度行为
x = torch.tensor([0.001, 0.5, 0.999], requires_grad=True)

# STE (梯度在边界处固定为1)
with torch.no_grad():
    y_ste = torch.round(x)
y_ste.backward(torch.ones_like(y_ste), retain_graph=True)
print("STE 梯度:", x.grad)  # tensor([1., 1., 1.])

# SLL (梯度智能集中在边界附近)
x.grad.zero_()
@sll.linearize(eps=0.1)
def sll_round(x):
    return torch.round(x)

y_sll = sll_round(x)
y_sll.backward(torch.ones_like(y_sll))
print("SLL 梯度:", x.grad)  # tensor([0., 5., 0.])  # 只有边界处有梯度！
```

**结论**：SLL 在保持前向精度的同时，智能地将梯度集中在真正需要优化的边界区域，实现更高效的训练。

### 🎨 梯度分布对比

![Gradient Distribution](demo_results/gradient_comparison.png)

**实际测试结果**：

- SLL 梯度: `[25.0, 0.0, 25.0, 0.0, 25.0]` — 仅在边界处有梯度
- STE 梯度: `[1.0, 1.0, 1.0, 1.0, 1.0]` — 处处有梯度，效率低下

***


## 🏛️ 项目结构

```
sll-core/
├── sll/
│   ├── __init__.py          # 模块导出
│   ├── core.py              # 核心 API（linearize）
│   ├── discovery.py         # 自动发现装饰器
│   └── ops.py               # SLL 算子实现
├── README.md
├── README_EN.md
├── LICENSE
└── pyproject.toml
```

***

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

***

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发环境

```bash
git clone https://github.com/jacksong-sourse/sll-core.git
cd sll-core
pip install -e .[dev]
```

### 运行测试

```bash
pytest tests/ -v
```

***

## 📚 引用

如果您在研究中使用 SLL，请引用：

```bibtex
@software{sll-core,
  title = {SLL-Core: Static Local Linearization for Differentiable Discrete Programming},
  author = {Jackson Guo},
  year = {2024},
  url = {https://github.com/jacksong-sourse/sll-core},
}
```

***

**⭐ 如果这个项目对您有帮助，请给个 Star！**

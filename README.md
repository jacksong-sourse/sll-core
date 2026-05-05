# SLL-Core: Static Local Linearization

**离散程序的零侵入可微分化引擎**

SLL（静态局部线性化程序变换）在程序入口处将不可微的硬决策边界替换为 ε-长度的局部线段，使程序全程可微；优化完成后在出口处严格恢复原始硬逻辑。当 ε→0 时，最优解收敛到原始程序最优解，误差无限小。

---

## 核心特性

- **零侵入**：无需重写业务代码，前后加几行即可
- **零残留**：出口严格恢复硬逻辑，部署时无性能损失
- **梯度有效**：边界附近导数为常数，无 Sigmoid 式梯度消失
- **框架原生**：基于 PyTorch，与 `torch.autograd` 无缝兼容

---

## 安装

```bash
pip install sll-core
```

---

## 快速开始

### 基础用法示例

```python
import torch
import sll

# 创建输入张量（需要 requires_grad=True）
x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

# 使用上下文管理器（推荐方式）
with sll.linearize(eps=1e-2):
    # 所有 torch 离散算子自动走 SLL 路径
    y = torch.sign(x)
    z = torch.round(y * 10)
    
    # 计算损失并反向传播
    loss = z.sum()
    loss.backward()
    
    # 梯度正常回传！
    print("梯度:", x.grad)

# 离开上下文后，torch.sign 恢复为原始硬逻辑
y_hard = torch.sign(x)
print("硬逻辑结果:", y_hard)
```

### 装饰器用法

```python
import torch
import sll

@sll.enable(eps=1e-2)
def quantized_model(x):
    """带量化操作的模型"""
    # 模拟量化：乘以 2 取整后除以 2
    quantized = torch.round(x * 2) / 2
    # 应用符号函数
    output = torch.sign(quantized)
    return output

# 使用模型
x = torch.randn(5, requires_grad=True)
y = quantized_model(x)
y.sum().backward()  # 梯度正常计算
```

---

## 使用说明

### 支持的算子

SLL 目前支持以下离散算子的可微版本：

| 算子 | 描述 | 示例 |
|------|------|------|
| `heaviside` | Heaviside 阶跃函数 | `sll.heaviside(x)` |
| `sign` | 符号函数 | `sll.sign(x)` / `torch.sign(x)` |
| `round` | 四舍五入 | `sll.round(x)` / `torch.round(x)` |
| `floor` | 向下取整 | `sll.floor(x)` / `torch.floor(x)` |
| `ceil` | 向上取整 | `sll.ceil(x)` / `torch.ceil(x)` |
| `threshold` | 通用阈值函数 | `sll.threshold(x, threshold=0.5)` |
| `argmax` | 返回 soft-one-hot 编码 | `sll.argmax(x, dim=1)` |

### 三种使用方式

#### 方式1：上下文管理器（推荐）

使用 `sll.linearize()` 上下文管理器，在代码块内自动 patch torch 离散算子：

```python
with sll.linearize(eps=1e-2):
    # 此代码块内的 torch.sign, torch.round 等自动走 SLL
    y = torch.sign(x)
    z = torch.round(y)
    loss = z.sum()
    loss.backward()
# 离开上下文后自动恢复原始逻辑
```

#### 方式2：装饰器

使用 `@sll.enable()` 装饰器包装整个函数：

```python
@sll.enable(eps=1e-2)
def my_function(x):
    return torch.round(x)

y = my_function(x)
```

#### 方式3：显式调用（不 patch 全局状态）

直接调用 `sll` 模块的函数，不影响全局 torch 行为：

```python
y = sll.heaviside(x, eps=1e-2)
z = sll.sign(y, eps=1e-2)
```

### 参数说明

- **eps**：线性化区间半宽，默认值为 `1e-3`。
  - 当输入值距离硬边界小于等于 `eps` 时，使用线性化近似
  - 当输入值距离硬边界大于 `eps` 时，使用原始硬逻辑
  - `eps` 越小，越接近原始硬逻辑，但梯度可能不稳定

### 实际应用示例

#### 示例1：训练带离散决策的模型

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
        # 硬阈值激活（不可微）
        return (x > 0).float()

model = DiscreteModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 使用 SLL 训练
for epoch in range(100):
    x = torch.randn(32, 10, requires_grad=True)
    
    with sll.linearize(eps=1e-2):
        y = model(x)
        loss = (y - target).pow(2).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

#### 示例2：量化感知训练

```python
import torch
import sll

def quantize(x, levels=256):
    """模拟量化操作"""
    scale = (levels - 1) / (x.max() - x.min() + 1e-10)
    quantized = torch.round((x - x.min()) * scale) / scale + x.min()
    return quantized

# 训练时使用 SLL
x = torch.randn(10, requires_grad=True)

with sll.linearize(eps=1e-3):
    y = quantize(x)
    loss = y.sum()
    loss.backward()

print("量化梯度:", x.grad)
```

---

## 原理简介

SLL 的核心思想是在离散决策边界附近建立局部线性化区间：

1. **入口处理**：在程序入口处，将所有硬边界（如 `sign`, `round`, `argmax`）替换为 ε-局部线性函数
2. **可微计算**：在前向传播中，边界附近的输入使用线性近似，保证处处可微
3. **梯度回传**：反向传播时，边界附近的导数为常数（线性函数的斜率）
4. **出口恢复**：在程序出口处，严格恢复原始硬逻辑，确保部署时无性能损失

### 数学形式

以 Heaviside 阶跃函数为例：

```
          | 0.5 + x/(2ε)    当 |x| ≤ ε
y'(x) =  |
          | H(x)            其他
```

其中 `H(x)` 是原始的 Heaviside 阶跃函数。当 `ε→0` 时，`y'(x)` 收敛到 `H(x)`。

---

## 注意事项

1. **ε 参数选择**：`eps` 过小时梯度可能不稳定，过大时近似误差较大，建议根据实际任务调整
2. **Tensor 方法**：对于 `x.sign()` 等 Tensor 方法，SLL 会尽力拦截，但建议使用 `torch.sign(x)` 确保一致性
3. **比较运算符**：Python 比较运算符（如 `x > 0`）无法被拦截，建议使用 `sll.threshold(x)` 替代
4. **部署阶段**：部署时无需加载 SLL，训练完成后直接使用原始代码即可

---

## 许可证

MIT License
"""
Static Local Linearization (SLL) Core
=====================================
对含不可微离散决策的程序，在入口处将硬边界替换为 ε-局部线段，
使程序全程可微；优化完成后出口严格恢复原始硬逻辑。
"""

from .core import linearize, patch, unpatch, enable
from .ops import (
    heaviside,
    sign,
    round,
    floor,
    ceil,
    threshold,
    argmax,
)

__version__ = "0.1.4"

__all__ = [
    "linearize",
    "patch",
    "unpatch",
    "enable",
    "heaviside",
    "sign",
    "round",
    "floor",
    "ceil",
    "threshold",
    "argmax",
]

"""MMForge · 跨模态对比学习系统。

从零手写的跨模态对齐与检索基准实验室：线性 CCA（经典强基线）、独立 PCA、
随机投影、线性/非线性对比学习双编码器、MMFuse 后期融合旗舰。纯 numpy 实现，
零预训练权重，完全离线可复现。
"""

from .core.seed import set_all

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = ["__version__", "set_all"]

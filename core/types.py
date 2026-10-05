"""共享数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PairDataset:
    """配对跨模态数据集：(X, Y) 共享同一潜在概念 z。"""

    X: np.ndarray  # (n, x_dim)
    Y: np.ndarray  # (n, y_dim)
    labels: np.ndarray  # (n,) 类别 id（支撑分类度量）
    regime: str = "nonlinear"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X/Y 样本数不一致")
        if self.labels is None:
            self.labels = np.zeros(self.X.shape[0], dtype=int)
        self.labels = np.asarray(self.labels).astype(int)


@dataclass
class BenchmarkRow:
    """单条基准记录（一次 seed × 一个方法 × 一个 regime）。"""

    method: str
    regime: str
    seed: int
    recall1_x2y: float
    recall5_x2y: float
    recall1_y2x: float
    acc_1nn_x2x: float
    acc_1nn_x2y: float
    alignment: float
    uniformity: float
    elapsed_sec: float

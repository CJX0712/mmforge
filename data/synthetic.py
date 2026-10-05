"""配对跨模态合成数据生成器（确定性，可复现）。

生成逻辑：潜在概念 z ~ 高斯混合（支撑分类度量）；两个模态 x,y 均为 z 的
（确定性）映射 + 噪声。coupling='nonlinear' 时映射为两层 tanh 随机网络，
'linear' 时为单层线性映射。跨模态对齐需从配对 (x_i, y_i) 恢复共享 z：
线性 CCA 只能捕捉 x-y 的线性相关性，非线性对比模型可学得共享结构。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import DataError
from ..core.types import PairDataset


def generate_pairs(
    rng: np.random.Generator,
    n: int = 2000,
    z_dim: int = 4,
    x_dim: int = 32,
    y_dim: int = 32,
    h_dim: int = 24,
    coupling: str = "nonlinear",
    noise: float = 0.15,
    n_classes: int = 4,
    class_sep: float = 3.0,
) -> PairDataset:
    """生成配对跨模态数据。

    coupling: 'nonlinear'（tanh 随机网络）| 'linear'（线性映射）
    """
    if coupling not in ("nonlinear", "linear"):
        raise DataError(f"未知 coupling: {coupling}")
    if n <= 0 or z_dim <= 0 or x_dim <= 0 or y_dim <= 0:
        raise DataError("维度/样本数必须为正整数")

    centers = rng.standard_normal((n_classes, z_dim)) * class_sep
    cls = rng.integers(0, n_classes, size=n)
    z = centers[cls] + rng.standard_normal((n, z_dim))

    # 每个模态独立的随机映射（固定随机权重，模拟真实跨模态异构表征）
    W1x = rng.standard_normal((z_dim, h_dim)) / np.sqrt(z_dim)
    b1x = rng.standard_normal(h_dim) * 0.1
    W2x = rng.standard_normal((h_dim, x_dim)) / np.sqrt(h_dim)
    b2x = rng.standard_normal(x_dim) * 0.1
    W1y = rng.standard_normal((z_dim, h_dim)) / np.sqrt(z_dim)
    b1y = rng.standard_normal(h_dim) * 0.1
    W2y = rng.standard_normal((h_dim, y_dim)) / np.sqrt(h_dim)
    b2y = rng.standard_normal(y_dim) * 0.1

    if coupling == "nonlinear":
        hx = np.tanh(z @ W1x + b1x)
        hy = np.tanh(z @ W1y + b1y)
    else:
        hx = z @ W1x + b1x
        hy = z @ W1y + b1y

    X = hx @ W2x + b2x + rng.standard_normal((n, x_dim)) * noise
    Y = hy @ W2y + b2y + rng.standard_normal((n, y_dim)) * noise

    return PairDataset(
        X=X.astype(np.float64),
        Y=Y.astype(np.float64),
        labels=cls.astype(int),
        regime=coupling,
        meta={
            "z_dim": z_dim,
            "x_dim": x_dim,
            "y_dim": y_dim,
            "h_dim": h_dim,
            "noise": noise,
            "n_classes": n_classes,
            "class_sep": class_sep,
        },
    )


def train_test_split(
    ds: PairDataset, rng: np.random.Generator, n_test: int = 500
) -> tuple[PairDataset, PairDataset]:
    """确定性切分（不打乱类别分布，纯随机下标，无信息泄漏）。"""
    n = ds.X.shape[0]
    if n_test >= n:
        raise DataError("n_test 必须小于样本总数")
    idx = rng.permutation(n)
    ti = idx[:n_test]
    tr = idx[n_test:]
    tr_ds = PairDataset(
        X=ds.X[tr],
        Y=ds.Y[tr],
        labels=ds.labels[tr],
        regime=ds.regime,
        meta=dict(ds.meta),
    )
    te_ds = PairDataset(
        X=ds.X[ti],
        Y=ds.Y[ti],
        labels=ds.labels[ti],
        regime=ds.regime,
        meta=dict(ds.meta),
    )
    return tr_ds, te_ds

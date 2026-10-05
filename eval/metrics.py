"""跨模态评测指标（统一语义：分数越大越相似 / 越大越好）。"""

from __future__ import annotations

import numpy as np


def recall_at_k(S: np.ndarray, k: int) -> float:
    """S: (n_query, n_candidate) 相似度。命中=候选中第 i 个等于查询 i。

    返回 top-k 召回率。
    """
    N = S.shape[0]
    order = np.argsort(-S, axis=1)[:, :k]
    hits = (order == np.arange(N)[:, None]).any(axis=1)
    return float(hits.mean())


def recall1(S: np.ndarray) -> float:
    return recall_at_k(S, 1)


def recall5(S: np.ndarray) -> float:
    return recall_at_k(S, 5)


def one_nn_accuracy_q(
    Q: np.ndarray, Q_labels: np.ndarray, R: np.ndarray, R_labels: np.ndarray
) -> float:
    Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    S = Qn @ Rn.T
    pred = R_labels[np.argmax(S, axis=1)]
    return float(np.mean(pred == Q_labels))


def alignment(F: np.ndarray, G: np.ndarray) -> float:
    """Wang & Isola 对齐项：配对嵌入 L2 距离平方均值（越小越好）。"""
    return float(np.mean(np.sum((F - G) ** 2, axis=1)))


def uniformity(F: np.ndarray, sample: int = 500, seed: int = 0) -> float:
    """Wang & Isola 均匀性：exp(-2||f_i-f_j||^2) 均值（越小越好）。"""
    n = F.shape[0]
    if n > sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, sample, replace=False)
        F = F[idx]
    diff = F[:, None, :] - F[None, :, :]
    d2 = np.sum(diff**2, axis=2)
    return float(np.mean(np.exp(-2.0 * d2)))

"""线性典型相关分析（CCA）基线 —— 跨模态对齐的经典强基线。

纯 numpy 实现（基于白化 + SVD），带正则化保证数值稳定。
"""

from __future__ import annotations

import numpy as np


def _sqrt_inv(mat: np.ndarray, reg: float, eps: float = 1e-10) -> np.ndarray:
    eig, V = np.linalg.eigh(mat + reg * np.eye(mat.shape[0]))
    eig = np.clip(eig, eps, None)
    return V @ np.diag(1.0 / np.sqrt(eig)) @ V.T


class CCA:
    """线性 CCA：拟合 X,Y 的共享坐标，输出余弦相似度矩阵。"""

    def __init__(self, n_components: int = 16, reg: float = 1e-3):
        self.k = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, Y: np.ndarray) -> CCA:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        n = X.shape[0]
        self.mx_ = X.mean(0)
        self.my_ = Y.mean(0)
        Xc = X - self.mx_
        Yc = Y - self.my_
        Cxx = (Xc.T @ Xc) / (n - 1) + self.reg * np.eye(X.shape[1])
        Cyy = (Yc.T @ Yc) / (n - 1) + self.reg * np.eye(Y.shape[1])
        Cxy = (Xc.T @ Yc) / (n - 1)
        Xs = _sqrt_inv(Cxx, 0.0)
        Ys = _sqrt_inv(Cyy, 0.0)
        M = Xs @ Cxy @ Ys
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        self.wx_ = Xs @ U[:, : self.k]
        self.wy_ = Ys @ Vt[: self.k].T
        return self

    def _project(self, X: np.ndarray, Y: np.ndarray):
        Xc = X - self.mx_
        Yc = Y - self.my_
        Ux = Xc @ self.wx_
        Uy = Yc @ self.wy_
        return Ux, Uy

    def similarity(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        Ux, Uy = self._project(X, Y)
        Ux = Ux / (np.linalg.norm(Ux, axis=1, keepdims=True) + 1e-12)
        Uy = Uy / (np.linalg.norm(Uy, axis=1, keepdims=True) + 1e-12)
        return Ux @ Uy.T

    def transform(self, X: np.ndarray, Y: np.ndarray):
        return self._project(X, Y)

    def encode_x(self, X: np.ndarray) -> np.ndarray:
        Ux, _ = self._project(X, X)
        return Ux

    def encode_y(self, Y: np.ndarray) -> np.ndarray:
        _, Uy = self._project(Y, Y)
        return Uy

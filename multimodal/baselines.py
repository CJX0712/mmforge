"""基线方法：独立 PCA（无跨模态对齐）/ 随机投影（地板）/ 线性对比（消融）。

统一接口：fit(Xtr, Ytr) + similarity(Xte, Yte)。
"""

from __future__ import annotations

import numpy as np

from .contrastive import ContrastiveModel


class IndependentPCA:
    """两模态各自独立 PCA，无跨模态对齐 → 检索应接近随机。

    优先 scikit-learn；不可用时降级纯 numpy SVD-PCA（离线兜底）。
    """

    def __init__(self, n_components: int = 16):
        self.k = n_components
        self.backend = "sklearn"

    def _fit_one(self, X):
        k = min(self.k, X.shape[1])
        try:
            from sklearn.decomposition import PCA

            return ("sklearn", PCA(n_components=k).fit(X))
        except Exception:
            # 纯 numpy SVD-PCA 兜底
            self.backend = "numpy"
            Xc = X - X.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            comp = Vt[:k].T
            mean = X.mean(0)
            return ("numpy", (mean, comp))

    @staticmethod
    def _transform(obj, X):
        kind, payload = obj
        if kind == "sklearn":
            return payload.transform(X)
        mean, comp = payload
        return (X - mean) @ comp

    def fit(self, X, Y, **kw):
        self.px_ = self._fit_one(X)
        self.py_ = self._fit_one(Y)
        return self

    def similarity(self, X, Y):
        Ux = self._transform(self.px_, X)
        Uy = self._transform(self.py_, Y)
        Ux = Ux / (np.linalg.norm(Ux, axis=1, keepdims=True) + 1e-12)
        Uy = Uy / (np.linalg.norm(Uy, axis=1, keepdims=True) + 1e-12)
        return Ux @ Uy.T

    def encode_x(self, X):
        return self._transform(self.px_, X)

    def encode_y(self, Y):
        return self._transform(self.py_, Y)


class RandomProjection:
    """随机高斯投影（无训练）→ 检索地板。"""

    def __init__(self, dim: int = 32, rng: np.random.Generator | None = None):
        self.dim = dim
        self.rng = rng or np.random.default_rng(0)

    def fit(self, X, Y, **kw):
        self.Wx_ = self.rng.standard_normal((X.shape[1], self.dim)) / np.sqrt(X.shape[1])
        self.Wy_ = self.rng.standard_normal((Y.shape[1], self.dim)) / np.sqrt(Y.shape[1])
        return self

    def similarity(self, X, Y):
        Ux = X @ self.Wx_
        Uy = Y @ self.Wy_
        Ux = Ux / (np.linalg.norm(Ux, axis=1, keepdims=True) + 1e-12)
        Uy = Uy / (np.linalg.norm(Uy, axis=1, keepdims=True) + 1e-12)
        return Ux @ Uy.T

    def encode_x(self, X):
        return X @ self.Wx_

    def encode_y(self, Y):
        return Y @ self.Wy_


class LinearContrastive:
    """线性编码器对比（消融：去掉非线性，验证非线性贡献）。"""

    def __init__(
        self,
        x_dim=32,
        y_dim=32,
        temperature=0.1,
        out_dim=32,
        lr=1e-2,
        rng: np.random.Generator | None = None,
    ):
        self.model = ContrastiveModel(
            x_dim=x_dim,
            y_dim=y_dim,
            temperature=temperature,
            hidden=(),
            out_dim=out_dim,
            lr=lr,
            rng=rng,
            linear=True,
        )

    def fit(self, X, Y, epochs=150, batch_size=256, rng=None, **kw):
        self.model.train(X, Y, epochs=epochs, batch_size=batch_size, rng=rng)
        return self

    def similarity(self, X, Y):
        return self.model.similarity(X, Y)

    def encode_x(self, X):
        return self.model.encode_x(X)

    def encode_y(self, Y):
        return self.model.encode_y(Y)

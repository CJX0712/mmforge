"""MMFuse 旗舰：对比学习 + CCA 后期融合。

将非线性对比相似度与全局线性 CCA 相似度按 alpha 加权融合，兼顾
局部非线性结构与全局线性相关；alpha 在验证集上网格搜索。
"""

from __future__ import annotations

import numpy as np

from .cca import CCA
from .contrastive import ContrastiveModel


class MMFuse:
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        temperature: float = 0.1,
        hidden=(64, 64),
        out_dim: int = 32,
        lr: float = 1e-2,
        rng: np.random.Generator | None = None,
        cca_dim: int = 16,
        fusion_alpha: float = 0.5,
    ):
        self.contrastive = ContrastiveModel(
            x_dim, y_dim, temperature=temperature, hidden=hidden, out_dim=out_dim, lr=lr, rng=rng
        )
        self.cca = CCA(n_components=cca_dim, reg=1e-3)
        self.alpha = fusion_alpha
        self.best_alpha_ = fusion_alpha

    def fit(
        self, Xtr, Ytr, Xva=None, Yva=None, epochs=150, batch_size=256, rng=None, tune_alpha=True
    ):
        self.contrastive.train(Xtr, Ytr, epochs=epochs, batch_size=batch_size, rng=rng)
        self.cca.fit(Xtr, Ytr)
        if tune_alpha and Xva is not None and Yva is not None:
            self.best_alpha_ = self._tune_alpha(Xva, Yva)
            self.alpha = self.best_alpha_
        return self

    def _tune_alpha(self, Xva, Yva, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
        best_a, best_r = self.alpha, -1.0
        for a in grid:
            r = self._recall1(self._sim(Xva, Yva, a), None)
            if r > best_r:
                best_r, best_a = r, a
        return best_a

    def _sim(self, X, Y, alpha):
        sc = self.contrastive.similarity(X, Y)
        scca = self.cca.similarity(X, Y)
        return alpha * sc + (1 - alpha) * scca

    @staticmethod
    def _recall1(S, _):
        N = S.shape[0]
        order = np.argsort(-S, axis=1)  # 降序
        hits = (order[:, 0] == np.arange(N)).mean()
        return float(hits)

    def similarity(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return self._sim(X, Y, self.alpha)

    def encode_x(self, X):
        return self.contrastive.encode_x(X)

    def encode_y(self, Y):
        return self.contrastive.encode_y(Y)

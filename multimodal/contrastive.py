"""对比学习模型（InfoNCE 双编码器）—— 非线性跨模态对齐旗舰核心。

两个手写 MLP 编码器 + 温度缩放 InfoNCE 损失，pull 配对 (x_i,y_i) 靠近、
push 非配对远离。可切换线性编码器（消融：非线性是否贡献增益）。
"""

from __future__ import annotations

import numpy as np

from ..core.errors import TrainingError
from .net import MLPEncoder


class ContrastiveModel:
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        temperature: float = 0.1,
        hidden=(64, 64),
        out_dim: int = 32,
        lr: float = 1e-2,
        rng: np.random.Generator | None = None,
        linear: bool = False,
    ):
        self.temperature = temperature
        self.lr = lr
        self.out_dim = out_dim
        h = () if linear else hidden
        self.linear = linear
        self.fx = MLPEncoder(x_dim, h, out_dim, rng=rng, lr=lr)
        self.gy = MLPEncoder(y_dim, h, out_dim, rng=rng, lr=lr)
        self.last_loss = float("nan")

    def _infonce(self, F: np.ndarray, G: np.ndarray):
        N = F.shape[0]
        S = F @ G.T / self.temperature
        Smax = S.max(1, keepdims=True)
        E = np.exp(S - Smax)
        P = E / E.sum(1, keepdims=True)
        dLdS = (P - np.eye(N)) / N
        dF = (1.0 / self.temperature) * (dLdS @ G)
        dG = (1.0 / self.temperature) * (dLdS.T @ F)
        loss = float(-np.mean(np.log(P[np.arange(N), np.arange(N)] + 1e-12)))
        return dF, dG, loss

    def train(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 150,
        batch_size: int = 256,
        rng: np.random.Generator | None = None,
        verbose: bool = False,
    ):
        rng = rng or np.random.default_rng(0)
        N = X.shape[0]
        if N < 2:
            raise TrainingError("训练样本不足")
        for ep in range(epochs):
            perm = rng.permutation(N)
            for s in range(0, N, batch_size):
                idx = perm[s : s + batch_size]
                if idx.size < 2:
                    continue
                F = self.fx.forward(X[idx], cache=True)
                G = self.gy.forward(Y[idx], cache=True)
                dF, dG, loss = self._infonce(F, G)
                self.fx.backward_update(dF)
                self.gy.backward_update(dG)
            if verbose and (ep + 1) % 25 == 0:
                F = self.fx.forward(X, cache=False)
                G = self.gy.forward(Y, cache=False)
                _, _, loss = self._infonce(F, G)
                print(f"  epoch {ep + 1:03d} loss={loss:.4f}")
        self.last_loss = loss if "loss" in dir() else self.last_loss
        return self

    def fit(self, X, Y, epochs=150, batch_size=256, rng=None, **kw):
        return self.train(X, Y, epochs=epochs, batch_size=batch_size, rng=rng)

    def encode_x(self, X):
        return self.fx.forward(X, cache=False)

    def encode_y(self, Y):
        return self.gy.forward(Y, cache=False)

    def similarity(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        F = self.fx.forward(X, cache=False)
        G = self.gy.forward(Y, cache=False)
        return F @ G.T

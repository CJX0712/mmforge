"""纯 numpy 手写 MLP 编码器：前向 + 反向传播 + Adam + L2 归一化。

用于跨模态对比学习的可训练非线性编码器。含两类梯度自检工具
（归一化 Jacobian 对照 + 权重有限差分对照），是正确性不变量之一。
"""

from __future__ import annotations

import numpy as np


def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(mat, axis=axis, keepdims=True)
    return mat / np.maximum(n, eps)


class MLPEncoder:
    """二层隐藏 tanh + 线性输出的 MLP；输出 L2 归一化。"""

    def __init__(
        self,
        in_dim: int,
        hidden=(64, 64),
        out_dim: int = 32,
        rng: np.random.Generator | None = None,
        lr: float = 1e-2,
    ):
        self.in_dim = in_dim
        self.hidden = tuple(hidden)
        self.out_dim = out_dim
        self.lr = lr
        rng = rng or np.random.default_rng(0)
        dims = [in_dim] + list(hidden) + [out_dim]
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for i in range(len(dims) - 1):
            std = np.sqrt(2.0 / dims[i])
            self.W.append(rng.standard_normal((dims[i], dims[i + 1])) * std)
            self.b.append(np.zeros(dims[i + 1]))
        self.m = [np.zeros_like(w) for w in self.W]
        self.v = [np.zeros_like(b) for b in self.b]
        self.t = 0
        self._acts = None
        self._raw = None

    # ---- 前向 ----
    def forward(self, X: np.ndarray, cache: bool = True):
        a = np.asarray(X, dtype=np.float64)
        acts = [a]
        for i in range(len(self.W) - 1):
            z = a @ self.W[i] + self.b[i]
            a = np.tanh(z)
            acts.append(a)
        raw = a @ self.W[-1] + self.b[-1]
        acts.append(raw)
        emb = l2_normalize(raw, axis=1)
        if cache:
            self._acts = acts
            self._raw = raw
        return emb

    def forward_raw(self, X: np.ndarray) -> np.ndarray:
        a = np.asarray(X, dtype=np.float64)
        for i in range(len(self.W) - 1):
            a = np.tanh(a @ self.W[i] + self.b[i])
        return a @ self.W[-1] + self.b[-1]

    def _backward(self, grad_emb: np.ndarray) -> np.ndarray:
        """grad_emb: 对归一化输出 emb 的梯度 → 返回对 raw 的梯度。"""
        raw = self._raw
        n = np.linalg.norm(raw, axis=1, keepdims=True)
        n = np.maximum(n, 1e-12)
        e = raw / n
        eg = np.sum(e * grad_emb, axis=1, keepdims=True)
        return (grad_emb - e * eg) / n

    def _compute_grads(self, grad_raw: np.ndarray):
        acts = self._acts
        L = len(self.W)
        grads_W = [None] * L
        grads_b = [None] * L
        d = grad_raw
        a_prev = acts[L - 1]
        grads_W[L - 1] = a_prev.T @ d
        grads_b[L - 1] = d.sum(axis=0)
        da = d @ self.W[L - 1].T
        for i in range(L - 2, -1, -1):
            a_next = acts[i + 1]
            dz = da * (1.0 - a_next**2)
            grads_W[i] = acts[i].T @ dz
            grads_b[i] = dz.sum(axis=0)
            da = dz @ self.W[i].T
        return grads_W, grads_b

    def backward_update(self, grad_emb: np.ndarray, beta1=0.9, beta2=0.999, eps=1e-8):
        grad_raw = self._backward(grad_emb)
        grads_W, grads_b = self._compute_grads(grad_raw)
        self.t += 1
        for i in range(len(self.W)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grads_W[i]
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grads_W[i] ** 2)
            mhat = self.m[i] / (1 - beta1**self.t)
            vhat = self.v[i] / (1 - beta2**self.t)
            self.W[i] -= self.lr * mhat / (np.sqrt(vhat) + eps)
            self.b[i] -= self.lr * grads_b[i]
        return None

    # ---- 梯度自检 ----
    def gradient_check(self, X: np.ndarray, eps: float = 1e-5, tol: float = 1e-6):
        """双校验：(1) 归一化 Jacobian 对照；(2) 权重有限差分对照。返回最大 rel。"""
        rng = np.random.default_rng(0)
        X = np.asarray(X, dtype=np.float64)
        self.forward(X, cache=True)
        raw = self._raw

        # (1) 归一化 Jacobian：J = g·normalize(raw)，g 随机
        g = rng.standard_normal(raw.shape)
        d = rng.standard_normal(raw.shape)
        d /= np.linalg.norm(d)
        grad_raw = self._backward(g)
        ana = float(np.sum(grad_raw * d))
        fp = l2_normalize(raw + eps * d, axis=1)
        fm = l2_normalize(raw - eps * d, axis=1)
        num = float(np.sum((fp - fm) * g) / (2 * eps))
        denom = max(abs(num), abs(ana), 1e-12)
        rel_norm = abs(num - ana) / denom
        if rel_norm > tol:
            raise AssertionError(f"归一化 Jacobian 自检失败 rel={rel_norm:.2e}")

        # (2) 权重有限差分：J = c·raw，c 随机；逐个权重对照
        c = rng.standard_normal(raw.shape)
        grads_W, grads_b = self._compute_grads(c)
        max_rel = rel_norm
        for li in range(len(self.W)):
            gw = grads_W[li]
            for _ in range(3):  # 抽样若干权重项
                p, q = rng.integers(0, gw.shape[0]), rng.integers(0, gw.shape[1])
                w0 = self.W[li][p, q]
                self.W[li][p, q] = w0 + eps
                rpp = self.forward_raw(X)
                self.W[li][p, q] = w0 - eps
                rmm = self.forward_raw(X)
                self.W[li][p, q] = w0
                numw = float(np.sum(c * (rpp - rmm)) / (2 * eps))
                denomw = max(abs(numw), abs(gw[p, q]), 1e-12)
                relw = abs(numw - gw[p, q]) / denomw
                max_rel = max(max_rel, relw)
                if relw > tol:
                    raise AssertionError(f"权重梯度自检失败 layer={li} rel={relw:.2e}")
        return max_rel

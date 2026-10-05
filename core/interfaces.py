"""协议接口（统一语义：分数越大越相似）。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    def fit(self, X: np.ndarray) -> Encoder: ...

    def transform(self, X: np.ndarray) -> np.ndarray: ...

    def encode(self, X: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class Aligner(Protocol):
    def fit(self, X: np.ndarray, Y: np.ndarray) -> Aligner: ...

    def similarity(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray: ...

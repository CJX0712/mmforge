"""全局确定性种子。

所有随机入口统一走单个 ``numpy`` Generator，保证同 seed 两次运行逐位一致。
"""

from __future__ import annotations

import random

import numpy as np

_RNG = None
_SEED = 42


def set_all(seed: int = 42):
    """设置全局确定性：Python random + numpy legacy + 单个 Generator。"""
    global _RNG, _SEED
    _SEED = int(seed)
    random.seed(_SEED)
    np.random.seed(_SEED)
    _RNG = np.random.default_rng(_SEED)
    return _RNG


def get_rng() -> np.random.Generator:
    if _RNG is None:
        set_all(_SEED)
    return _RNG


def get_seed() -> int:
    return _SEED

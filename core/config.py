"""配置：ENV_MMFORGE_* 覆盖 + schema 校验。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError


@dataclass
class Config:
    seed: int = 42
    device: str = "cpu"
    epochs: int = 100
    batch_size: int = 256
    temperature: float = 0.1
    lr: float = 1e-2
    hidden: tuple = (64, 64)
    out_dim: int = 32
    fusion_alpha: float = 0.5
    n_seeds: int = 3
    n_train: int = 1300
    n_test: int = 320
    z_dim: int = 4
    x_dim: int = 32
    y_dim: int = 32
    noise: float = 0.15

    @classmethod
    def load(cls) -> Config:
        def gi(k, d):
            v = os.environ.get(k)
            return int(v) if (v is not None and v.strip()) else d

        def gf(k, d):
            v = os.environ.get(k)
            return float(v) if (v is not None and v.strip()) else d

        cfg = cls(
            seed=gi("MMFORGE_SEED", 42),
            epochs=gi("MMFORGE_EPOCHS", 150),
            batch_size=gi("MMFORGE_BATCH", 256),
            temperature=gf("MMFORGE_TEMP", 0.1),
            lr=gf("MMFORGE_LR", 1e-2),
            fusion_alpha=gf("MMFORGE_ALPHA", 0.5),
            n_seeds=gi("MMFORGE_SEEDS", 3),
            n_train=gi("MMFORGE_NTRAIN", 2000),
            n_test=gi("MMFORGE_NTEST", 500),
            z_dim=gi("MMFORGE_ZDIM", 4),
            x_dim=gi("MMFORGE_XDIM", 32),
            y_dim=gi("MMFORGE_YDIM", 32),
            noise=gf("MMFORGE_NOISE", 0.15),
        )
        if cfg.epochs <= 0:
            raise ConfigError("MMFORGE_EPOCHS 必须 > 0")
        if cfg.temperature <= 0:
            raise ConfigError("MMFORGE_TEMP 必须 > 0")
        if not (0.0 <= cfg.fusion_alpha <= 1.0):
            raise ConfigError("MMFORGE_ALPHA 必须 ∈ [0,1]")
        if cfg.n_seeds < 1:
            raise ConfigError("MMFORGE_SEEDS 必须 >= 1")
        if cfg.n_train <= 0 or cfg.n_test <= 0:
            raise ConfigError("样本数必须 > 0")
        return cfg

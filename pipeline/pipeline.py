"""MMForge 端到端管线：数据生成 → 多方法基准 → 落盘 benchmark.json。

确定性：每个 (regime, seed) 用 set_all(seed) 重置全局 RNG，随机入口顺序固定，
同 seed 两次运行核心指标逐位一致（elapsed_sec 除外）。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np

from ..core.config import Config
from ..core.seed import get_rng, set_all
from ..core.types import PairDataset
from ..data.synthetic import generate_pairs, train_test_split
from ..eval.metrics import alignment, one_nn_accuracy_q, recall1, recall5, uniformity
from ..multimodal.baselines import IndependentPCA, LinearContrastive, RandomProjection
from ..multimodal.cca import CCA
from ..multimodal.contrastive import ContrastiveModel
from ..multimodal.fuse import MMFuse

METHODS = [
    "random_proj",
    "independent_pca",
    "cca",
    "linear_contrastive",
    "contrastive",
    "mmfuse",
]


def build_method(name: str, cfg: Config, rng):
    if name == "random_proj":
        return RandomProjection(dim=cfg.out_dim, rng=rng)
    if name == "independent_pca":
        return IndependentPCA(n_components=cfg.out_dim)
    if name == "cca":
        return CCA(n_components=cfg.out_dim, reg=1e-3)
    if name == "linear_contrastive":
        return LinearContrastive(
            x_dim=cfg.x_dim,
            y_dim=cfg.y_dim,
            temperature=cfg.temperature,
            out_dim=cfg.out_dim,
            lr=cfg.lr,
            rng=rng,
        )
    if name == "contrastive":
        return ContrastiveModel(
            x_dim=cfg.x_dim,
            y_dim=cfg.y_dim,
            temperature=cfg.temperature,
            hidden=cfg.hidden,
            out_dim=cfg.out_dim,
            lr=cfg.lr,
            rng=rng,
        )
    if name == "mmfuse":
        return MMFuse(
            x_dim=cfg.x_dim,
            y_dim=cfg.y_dim,
            temperature=cfg.temperature,
            hidden=cfg.hidden,
            out_dim=cfg.out_dim,
            lr=cfg.lr,
            rng=rng,
            cca_dim=cfg.out_dim,
            fusion_alpha=cfg.fusion_alpha,
        )
    raise ValueError(f"未知方法 {name}")


def _evaluate(name, m, tr2, va, te, cfg, regime, seed):
    t0 = time.time()
    if name == "mmfuse":
        m.fit(
            tr2.X,
            tr2.Y,
            va.X,
            va.Y,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            rng=get_rng(),
            tune_alpha=True,
        )
    elif name in ("contrastive", "linear_contrastive"):
        m.fit(tr2.X, tr2.Y, epochs=cfg.epochs, batch_size=cfg.batch_size, rng=get_rng())
    else:
        m.fit(tr2.X, tr2.Y)
    elapsed = time.time() - t0

    S = m.similarity(te.X, te.Y)
    F = m.encode_x(te.X)
    G = m.encode_y(te.Y)
    refX = m.encode_x(tr2.X)
    refY = m.encode_y(tr2.Y)

    return {
        "method": name,
        "regime": regime,
        "seed": seed,
        "recall1_x2y": recall1(S),
        "recall5_x2y": recall5(S),
        "recall1_y2x": recall1(S.T),
        "recall5_y2x": recall5(S.T),
        "acc_1nn_x2x": one_nn_accuracy_q(F, te.labels, refX, tr2.labels),
        "acc_1nn_x2y": one_nn_accuracy_q(F, te.labels, refY, tr2.labels),
        "alignment": alignment(F, G),
        "uniformity": uniformity(F),
        "elapsed_sec": round(elapsed, 4),
    }


def run(cfg: Config, regimes=("nonlinear", "linear"), out_path="benchmark.json"):
    rows = []
    for regime in regimes:
        for seed in range(cfg.seed, cfg.seed + cfg.n_seeds):
            set_all(seed)
            rng = get_rng()
            n_total = cfg.n_train + cfg.n_test
            full = generate_pairs(
                rng,
                n=n_total,
                z_dim=cfg.z_dim,
                x_dim=cfg.x_dim,
                y_dim=cfg.y_dim,
                coupling=regime,
                noise=cfg.noise,
            )
            tr_all, te = train_test_split(full, rng, n_test=cfg.n_test)
            ntr = tr_all.X.shape[0]
            nva = max(2, ntr // 5)
            va_idx = set(rng.permutation(ntr)[:nva].tolist())
            tr_idx = np.array([i for i in range(ntr) if i not in va_idx])
            tr2 = PairDataset(tr_all.X[tr_idx], tr_all.Y[tr_idx], tr_all.labels[tr_idx], regime)
            va = PairDataset(
                tr_all.X[list(va_idx)], tr_all.Y[list(va_idx)], tr_all.labels[list(va_idx)], regime
            )
            for name in METHODS:
                m = build_method(name, cfg, get_rng())
                row = _evaluate(name, m, tr2, va, te, cfg, regime, seed)
                rows.append(row)
                print(
                    f"  [{regime}/{seed}] {name:18s} "
                    f"R1_x2y={row['recall1_x2y']:.3f} "
                    f"R5_x2y={row['recall5_x2y']:.3f} "
                    f"acc_x2y={row['acc_1nn_x2y']:.3f} "
                    f"({row['elapsed_sec']:.1f}s)"
                )

    summary = _summarize(rows, regimes, cfg)
    payload = {
        "config": asdict(cfg),
        "regimes": list(regimes),
        "summary": summary,
        "rows": rows,
    }
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def _summarize(rows, regimes, cfg):
    summary = {}
    for regime in regimes:
        rrows = [r for r in rows if r["regime"] == regime]
        by_method = {}
        for name in METHODS:
            vals = [r["recall1_x2y"] for r in rrows if r["method"] == name]
            by_method[name] = {
                "recall1_x2y_mean": float(np.mean(vals)),
                "recall1_x2y_std": float(np.std(vals)),
                "acc_x2y_mean": float(
                    np.mean([r["acc_1nn_x2y"] for r in rrows if r["method"] == name])
                ),
            }
        summary[regime] = by_method
    return summary

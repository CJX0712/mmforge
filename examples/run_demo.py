"""端到端演示：生成 → 基准 → 落盘 benchmark.json → 确定性自检。

用法（仓库根目录）：
    python examples/run_demo.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mmforge.core.config import Config
from mmforge.pipeline.pipeline import run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "benchmark.json")


def determinism_check(cfg: Config) -> float:
    """轻量确定性校验：同配置两次运行核心指标应逐位一致。"""
    small = Config()
    small.n_seeds = 1
    small.epochs = 30
    small.n_train = 600
    small.n_test = 150
    p1 = run(small, regimes=("nonlinear",), out_path=None)
    p2 = run(small, regimes=("nonlinear",), out_path=None)
    keys = ["recall1_x2y", "recall5_x2y", "recall1_y2x", "acc_1nn_x2y", "alignment", "uniformity"]
    maxd = 0.0
    for a, b in zip(p1["rows"], p2["rows"], strict=False):
        for k in keys:
            maxd = max(maxd, abs(a[k] - b[k]))
    return maxd


def main() -> int:
    cfg = Config()  # 默认参数
    print("=== MMForge 端到端演示 ===")
    payload = run(cfg, regimes=("nonlinear", "linear"), out_path=OUT)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nbenchmark 已落盘: {OUT}")

    maxd = determinism_check(cfg)
    print(f"\n确定性自检: 同配置两次运行核心指标 max|Δ| = {maxd:.2e}")
    if maxd != 0.0:
        print("[FAIL] 确定性被破坏")
        return 1
    print("[PASS] 同 seed 两次运行核心指标逐位一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())

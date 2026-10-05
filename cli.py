"""MMForge 命令行入口（argparse）。"""

from __future__ import annotations

import argparse
import sys

from .core.config import Config
from .pipeline.pipeline import METHODS, run


def _doctor() -> int:
    print("=== MMForge 后端体检 ===")
    ok = True
    try:
        import numpy as np

        print(f"[PASS] numpy {np.__version__}")
    except Exception as e:
        print(f"[FAIL] numpy 不可用: {e}")
        ok = False
    try:
        import sklearn

        print(f"[PASS] scikit-learn {sklearn.__version__}")
    except Exception as e:
        print(f"[WARN] scikit-learn 不可用（部分基线降级）: {e}")
    print(f"[INFO] 内置方法: {', '.join(METHODS)}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mmforge", description="MMForge · 跨模态对比学习系统")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    ap.add_argument("--seeds", type=int, default=None, help="覆盖随机种子数")
    ap.add_argument("--regimes", nargs="*", default=["nonlinear", "linear"])
    ap.add_argument("--out", default="benchmark.json")
    ap.add_argument("--doctor", action="store_true", help="后端可用性体检")
    args = ap.parse_args(argv)

    if args.doctor:
        return _doctor()

    cfg = Config.load()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.seeds is not None:
        cfg.n_seeds = args.seeds

    payload = run(cfg, regimes=tuple(args.regimes), out_path=args.out)
    print("\n=== 汇总（recall1_x2y mean / acc_x2y mean）===")
    for regime, bym in payload["summary"].items():
        print(f"\nregime={regime}")
        for name, m in bym.items():
            print(
                f"  {name:18s} R1={m['recall1_x2y_mean']:.3f}±{m['recall1_x2y_std']:.3f} "
                f"acc_x2y={m['acc_x2y_mean']:.3f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

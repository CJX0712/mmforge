"""离线兜底：验证纯 numpy 路径在 scikit-learn 不可用时仍可用。"""

import builtins

import mmforge.multimodal.baselines as B
import numpy as np


def test_independent_pca_numpy_fallback(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn (offline)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pca = B.IndependentPCA(3)
    X = np.random.default_rng(0).standard_normal((50, 6))
    Y = np.random.default_rng(1).standard_normal((50, 6))
    pca.fit(X, Y)
    S = pca.similarity(X, Y)
    assert S.shape == (50, 50)
    assert pca.backend == "numpy"


def test_all_methods_run_without_optional_deps():
    """除 sklearn（已本地可用）外，其余均为纯 numpy，离线即可运行。"""
    import numpy as np
    from mmforge.core.config import Config
    from mmforge.pipeline.pipeline import run

    cfg = Config()
    cfg.n_seeds = 1
    cfg.epochs = 5
    cfg.n_train = 200
    cfg.n_test = 60
    payload = run(cfg, regimes=("linear",), out_path=None)
    assert len(payload["rows"]) > 0
    for r in payload["rows"]:
        assert np.isfinite(r["recall1_x2y"])

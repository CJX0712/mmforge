import numpy as np
from mmforge.core.config import Config
from mmforge.pipeline.pipeline import METHODS, run


def _small_cfg():
    cfg = Config()
    cfg.n_seeds = 1
    cfg.epochs = 10
    cfg.n_train = 300
    cfg.n_test = 80
    cfg.z_dim = 4
    return cfg


def test_pipeline_all_methods_present():
    cfg = _small_cfg()
    payload = run(cfg, regimes=("linear",), out_path=None)
    methods = {r["method"] for r in payload["rows"]}
    assert methods == set(METHODS)
    mmf = [r for r in payload["rows"] if r["method"] == "mmfuse"][0]
    assert mmf["recall1_x2y"] > 0.05


def test_determinism():
    cfg = _small_cfg()
    p1 = run(cfg, regimes=("linear",), out_path=None)
    p2 = run(cfg, regimes=("linear",), out_path=None)
    for a, b in zip(p1["rows"], p2["rows"]):
        assert abs(a["recall1_x2y"] - b["recall1_x2y"]) < 1e-12
        assert abs(a["alignment"] - b["alignment"]) < 1e-12


def test_flagship_beats_cca():
    cfg = _small_cfg()
    payload = run(cfg, regimes=("linear",), out_path=None)
    by = payload["summary"]["linear"]
    assert by["mmfuse"]["recall1_x2y_mean"] > by["cca"]["recall1_x2y_mean"]

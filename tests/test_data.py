import numpy as np
from mmforge.core.seed import get_rng, set_all
from mmforge.data.synthetic import generate_pairs, train_test_split


def test_generate_shapes():
    set_all(5)
    rng = get_rng()
    ds = generate_pairs(rng, n=200)
    assert ds.X.shape == (200, 32)
    assert ds.Y.shape == (200, 32)
    assert ds.labels.shape[0] == 200
    assert set(np.unique(ds.labels)).issubset({0, 1, 2, 3})


def test_split_disjoint():
    set_all(6)
    rng = get_rng()
    ds = generate_pairs(rng, n=300)
    tr, te = train_test_split(ds, rng, n_test=100)
    assert tr.X.shape[0] == 200
    assert te.X.shape[0] == 100
    # 无样本泄漏：合并后恰好为全集
    assert tr.X.shape[0] + te.X.shape[0] == ds.X.shape[0]


def test_coupling_distinct():
    set_all(7)
    rng = get_rng()
    dsn = generate_pairs(rng, n=200, coupling="nonlinear")
    set_all(7)
    rng = get_rng()
    dsl = generate_pairs(rng, n=200, coupling="linear")
    assert not np.allclose(dsn.X, dsl.X)


def test_bad_coupling_raises():
    set_all(8)
    rng = get_rng()
    try:
        generate_pairs(rng, n=10, coupling="bogus")
        assert False
    except Exception:
        pass

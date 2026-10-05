import numpy as np
from mmforge.core.seed import get_rng, set_all
from mmforge.multimodal.net import MLPEncoder, l2_normalize


def test_gradient_check_passes():
    set_all(0)
    rng = get_rng()
    net = MLPEncoder(6, (10, 10), 5, rng=rng)
    rel = net.gradient_check(np.random.default_rng(1).standard_normal((15, 6)))
    assert rel < 1e-6


def test_forward_shape_and_unit_norm():
    set_all(1)
    rng = get_rng()
    net = MLPEncoder(4, (8,), 6, rng=rng)
    X = np.random.default_rng(2).standard_normal((10, 4))
    e = net.forward(X, cache=False)
    assert e.shape == (10, 6)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0)


def test_backward_updates_weights():
    set_all(3)
    rng = get_rng()
    net = MLPEncoder(4, (8,), 6, rng=rng)
    X = np.random.default_rng(2).standard_normal((20, 4))
    w0 = net.W[0].copy()
    net.forward(X, cache=True)
    g = np.random.default_rng(4).standard_normal((20, 6))
    net.backward_update(g)
    assert not np.array_equal(net.W[0], w0)


def test_l2_normalize_idempotent():
    X = np.random.default_rng(0).standard_normal((5, 3))
    n1 = l2_normalize(X)
    n2 = l2_normalize(n1)
    assert np.allclose(n1, n2)

import numpy as np
from mmforge.multimodal.cca import CCA


def test_cca_matched_pairs_above_mean():
    rng = np.random.default_rng(0)
    n = 500
    z = rng.standard_normal((n, 3))
    X = z @ rng.standard_normal((3, 5))
    Y = z @ rng.standard_normal((3, 5)) + 0.01 * rng.standard_normal((n, 5))
    cca = CCA(n_components=3).fit(X, Y)
    S = cca.similarity(X, Y)
    assert np.mean(np.diag(S)) > np.mean(S)


def test_cca_encode_shape():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 4))
    Y = rng.standard_normal((100, 4))
    cca = CCA(2).fit(X, Y)
    assert cca.encode_x(X).shape == (100, 2)
    assert cca.encode_y(Y).shape == (100, 2)


def test_cca_symmetry_of_self():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((50, 6))
    cca = CCA(3).fit(X, X)
    S = cca.similarity(X, X)
    assert np.allclose(S, S.T, atol=1e-8)

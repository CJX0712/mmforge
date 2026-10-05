import numpy as np
from mmforge.core.seed import get_rng, set_all
from mmforge.data.synthetic import generate_pairs, train_test_split
from mmforge.multimodal.contrastive import ContrastiveModel


def _data(seed=2, coupling="linear"):
    set_all(seed)
    rng = get_rng()
    ds = generate_pairs(rng, n=400, coupling=coupling, noise=0.1)
    return train_test_split(ds, rng, n_test=100)


def test_train_reduces_loss():
    tr, _ = _data()
    m = ContrastiveModel(
        32, 32, temperature=0.1, hidden=(32, 32), out_dim=16, lr=1e-2, rng=get_rng()
    )
    m.train(tr.X, tr.Y, epochs=20, batch_size=128, rng=get_rng())
    F = m.encode_x(tr.X)
    G = m.encode_y(tr.Y)
    S = F @ G.T / m.temperature
    N = tr.X.shape[0]
    P = np.exp(S - S.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    loss = float(-np.mean(np.log(P[np.arange(N), np.arange(N)] + 1e-12)))
    assert np.isfinite(loss) and loss < np.log(N)


def test_similarity_beats_random():
    tr, te = _data()
    m = ContrastiveModel(
        32, 32, temperature=0.1, hidden=(32, 32), out_dim=16, lr=1e-2, rng=get_rng()
    )
    m.train(tr.X, tr.Y, epochs=30, batch_size=128, rng=get_rng())
    S = m.similarity(te.X, te.Y)
    N = te.X.shape[0]
    hits = float((np.argmax(S, axis=1) == np.arange(N)).mean())
    assert hits > (1.0 / N) * 3  # 至少 3× 随机基线


def test_linear_contrastive_beats_cca_on_linear():
    from mmforge.multimodal.baselines import LinearContrastive
    from mmforge.multimodal.cca import CCA

    tr, te = _data(coupling="linear")
    lc = LinearContrastive(32, 32, temperature=0.1, out_dim=16, lr=1e-2, rng=get_rng())
    lc.fit(tr.X, tr.Y, epochs=40, batch_size=128, rng=get_rng())
    cca = CCA(16).fit(tr.X, tr.Y)
    N = te.X.shape[0]
    r_lc = float((np.argmax(lc.similarity(te.X, te.Y), axis=1) == np.arange(N)).mean())
    r_cca = float((np.argmax(cca.similarity(te.X, te.Y), axis=1) == np.arange(N)).mean())
    assert r_lc > r_cca

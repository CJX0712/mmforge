import numpy as np
from mmforge.core.seed import get_rng, set_all


def test_set_all_deterministic():
    set_all(7)
    a = get_rng().standard_normal(5)
    set_all(7)
    b = get_rng().standard_normal(5)
    assert np.array_equal(a, b)


def test_set_all_changes_stream():
    set_all(1)
    a = get_rng().standard_normal(3)
    set_all(2)
    b = get_rng().standard_normal(3)
    assert not np.array_equal(a, b)

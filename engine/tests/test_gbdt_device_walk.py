"""S6-5: the GBDT's device walk takes a few steps per depth level, not per tree, and keeps predict's bits.

At 87baeec the A100 pipeline profile spent 100.7 ms of a query in the rank stage: the walk went one tree at a time
(T x depth small kernels). Now every tree walks at once; trees are still added in predict's order, in float64.
"""
import numpy as np
import pytest

from engine.backend import available_devices, get_backend
from engine.rank.gbdt import GBDT


def _model(n_rounds, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(1500, 8)).astype(np.float32)
    X[rng.random(X.shape) < 0.1] = np.nan
    X[:40, 2] = 0.25
    y = np.nan_to_num(X[:, 0]) * 2 + np.nan_to_num(X[:, 3]) - (np.nan_to_num(X[:, 5]) > 0)
    return GBDT(n_rounds=n_rounds, max_depth=5, min_leaf=10).fit(X[:1000], y[:1000]), X


@pytest.mark.parametrize("device", available_devices())
def test_bits_equal_predict_on_every_device(device):
    m, X = _model(60)
    b = get_backend(device)
    assert np.array_equal(b.to_numpy(m.predict_backend(b, b.asarray(X, "float32"))), m.predict(X))


def test_branching_steps_do_not_grow_with_the_number_of_trees():
    b = get_backend("cpu")
    counts = []
    for n_rounds in (5, 80):
        m, X = _model(n_rounds)
        calls = []
        real = b.where
        b.where = lambda *a, _r=real: (calls.append(1), _r(*a))[1]
        try:
            m.predict_backend(b, b.asarray(X, "float32"))
        finally:
            del b.where
        counts.append(len(calls))
    assert counts[0] == counts[1], f"b.where calls: {counts} for 5 and 80 trees"

import numpy as np
import pytest

from engine.backend import DEVICES, DeviceUnavailable, available_devices, get_backend, resolve_device


def test_cpu_first_and_resolve(monkeypatch):
    assert available_devices()[0] == "cpu"
    monkeypatch.delenv("S3_SEARCH_DEVICE", raising=False)
    assert resolve_device() == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")
    for d in DEVICES:
        if d not in available_devices():
            with pytest.raises(DeviceUnavailable):
                get_backend(d)


def test_topk_ties_by_index(device):
    b = get_backend(device)
    s = np.array([[1, 3, 3, 2, 3, 0], [5, 5, 5, 5, 5, 5]], np.float32)
    valid = np.array([[1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 1, 0]], bool)
    idx, val = b.topk(b.asarray(s), 4, b.asarray(valid))
    assert b.to_numpy(idx).tolist() == [[1, 2, 4, 3], [1, 3, 4, -1]]
    assert b.to_numpy(val)[1, 3] == -np.inf


def test_topk_matches_reference_sort(device):
    b = get_backend(device)
    rng = np.random.default_rng(1)
    s = rng.integers(0, 5, size=(7, 300)).astype(np.float32)          # many ties
    for k in (1, 10, 300):
        idx, _ = b.topk(b.asarray(s), k)
        want = [np.lexsort((np.arange(300), -row))[:k] for row in s]
        assert (b.to_numpy(idx) == np.stack(want)).all()


def test_bitmaps_roundtrip(device):
    b = get_backend(device)
    m = np.random.default_rng(2).random(77) > 0.5
    words = b.pack_bits(m)
    assert words.dtype == np.uint32 and words.shape == (3,)
    assert int(words[0]) & 1 == int(m[0])
    assert (b.to_numpy(b.unpack_bits(words, 77)) == m).all()


def test_csr_expand_and_segment_sum(device):
    b = get_backend(device)
    ptr = b.asarray(np.array([0, 2, 2, 5], np.int64))
    owner, entry = b.csr_expand(ptr, [2, 0, 1], [7, 8, 9])
    assert b.to_numpy(owner).tolist() == [7, 7, 7, 8, 8]
    assert b.to_numpy(entry).tolist() == [2, 3, 4, 0, 1]
    s = b.segment_sum(b.asarray(np.array([0, 2, 2, 1], np.int64)), b.asarray(np.array([1, 2, 3, 4], np.float32)), 3)
    assert b.to_numpy(s).tolist() == [1, 4, 5]


def test_bincount_rows(device):
    b = get_backend(device)
    codes = b.asarray(np.array([0, 1, 1, 2, 2, 2], np.int64))
    mask = b.asarray(np.array([[1, 1, 1, 1, 1, 1], [0, 0, 1, 0, 1, 1]], bool))
    assert b.to_numpy(b.bincount_rows(codes, mask, 3)).tolist() == [[1, 2, 3], [0, 1, 2]]


def test_numpy_count_nonzero_is_a_numpy_integer(monkeypatch):
    """numpy < 2 returns a Python int for axis=None; the backend's callers use .astype on it (aggs.range_agg)."""
    import numpy as np
    from engine.backend.numpy_backend import NumpyBackend
    real = np.count_nonzero
    monkeypatch.setattr(np, "count_nonzero", lambda a, axis=None: int(real(a)) if axis is None else real(a, axis=axis))
    b = NumpyBackend()
    n = b.count_nonzero(np.array([True, False, True]))
    assert n == 2 and b.astype(n, "float64").reshape(1)[0] == 2.0

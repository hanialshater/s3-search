"""S6-4: encoding a query reads only the components rows its terms use, and gives the full product's bits.

The A100 profile at 87baeec spent 150.8 ms of a 272.6 ms pipeline query in encode: the projection copied all
V x 128 components to float64 (twice) for every call. On CPU with V = 100,000 that was 177 ms; now 0.6 ms.
"""
import numpy as np
import pytest

from engine.encode import TfidfSvdEncoder, _sparse_matmul
from referee import loader


@pytest.fixture(scope="module")
def enc_and_texts():
    ds = loader.load_dataset("toy")
    texts = ds.catalogue["product_title"].tolist()
    return TfidfSvdEncoder(dim=32).fit(texts), texts + ["zzzz unknown", "", "the and of"]


def _full_product(enc, texts):
    r, c, v = enc._coo(texts)
    out = _sparse_matmul(r, c, v, enc.components.astype(np.float64), len(texts), "rows")
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.where(n > 0, n, 1.0)).astype(np.float32)


def test_bits_equal_the_full_product(enc_and_texts):
    enc, texts = enc_and_texts
    full = _full_product(enc, texts)
    assert np.array_equal(enc.encode(texts), full)
    for i, t in enumerate(texts):                 # a query alone (as the executor encodes it) gives the same bits
        assert np.array_equal(enc.encode_queries([t])[0], full[i])


class _NoWholeCopy(np.ndarray):
    limit = 0

    def astype(self, *a, **k):
        assert self.shape[0] <= self.limit, f"copied {self.shape[0]} components rows for one query"
        return super().astype(*a, **k)


def test_a_query_does_not_copy_every_component_row(enc_and_texts):
    enc, texts = enc_and_texts
    want = enc.encode_queries([texts[0]])
    enc2 = TfidfSvdEncoder.from_arrays(enc.to_arrays())
    enc2._word_id = enc._word_id
    enc2.components = np.asarray(enc.components).view(_NoWholeCopy)
    _NoWholeCopy.limit = len(texts[0].split())    # at most one row per word of the query
    assert len(enc2.components) > 2 * _NoWholeCopy.limit
    assert np.array_equal(np.asarray(enc2.encode_queries([texts[0]])), want)

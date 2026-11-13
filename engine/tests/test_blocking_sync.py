"""S6-15: on cuda the host waits for the device on a blocking event (no spinning core); results are the same."""
import pytest

api = pytest.importorskip("engine.api")
from referee import checks, loader  # noqa: E402

CUDA = [d for d in api.available_devices() if d == "cuda"]


@pytest.mark.parametrize("device", CUDA)
def test_blocking_and_spinning_waits_give_the_same_responses(device):
    ds = loader.load_dataset("toy")
    ix = api.build_index(ds.catalogue, ds.mapping, device=device)
    b = ix._st.b
    assert b.blocking_sync                                   # the default on cuda
    reqs = checks.lexical_requests(ds)[:30]
    got = [api.expand_hits(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
    b.blocking_sync = False
    try:
        want = [api.expand_hits(r) for r in ix.search_batch(reqs, hits="columns", ids="bytes")]
    finally:
        b.blocking_sync = True
    strip = lambda r: {k: v for k, v in r.items() if k != "took_ms"}  # noqa: E731
    assert [strip(r) for r in got] == [strip(r) for r in want]


def test_the_cpu_backends_never_block():
    from engine.backend import get_backend
    for d in api.available_devices():
        if d != "cuda":
            assert not getattr(get_backend(d), "blocking_sync", False)

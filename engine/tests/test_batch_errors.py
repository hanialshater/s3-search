"""S6-10: ``search_batch(..., errors="return")`` (INTERFACES §5, "The engine in one GPU process")."""
import pytest

api = pytest.importorskip("engine.api")
from engine.tests.test_gpu_paths import ranked_index  # noqa: E402


def _strip(r):
    return {k: v for k, v in api.expand_hits(r).items() if k != "took_ms"}


def test_refused_requests_take_their_slot_and_the_rest_answer_as_alone(segment, catalogue, device):
    ix = ranked_index(segment, catalogue, device)
    good = [{"query": {"match": {"product_title": "black shoes"}}},
            {"query": {"hybrid": {"queries": [{"match": {"product_title": "shoes"}},
                                              {"neural": {"embedding": {"query_text": "shoes", "k": 10}}}]}},
             "rescore": {"window_size": 10, "s3_ranker": {"model": "m"}}},
            {"query": {"match_all": {}}, "sort": [{"price": "desc"}], "size": 3}]
    bad = [{"query": {"no_such_query": {}}},                                              # parse
           {"query": {"match_all": {}}, "from": 9999, "size": 10},                         # invalid
           {"query": {"match_all": {}}, "rescore": {"window_size": 5, "s3_ranker": {"model": "x"}}}]  # ranker
    reqs = [good[0], bad[0], good[1], bad[1], bad[2], good[2]]
    for hits in ("dicts", "columns"):
        out = ix.search_batch(reqs, hits=hits, errors="return")
        assert len(out) == len(reqs)
        for r, o in zip(reqs, out):
            if r in bad:
                with pytest.raises(api.UnsupportedQuery) as e:
                    ix.search(r)
                assert o == {"error": {"type": type(e.value).__name__, "path": e.value.path,
                                       "reason": e.value.reason}}
            else:
                assert _strip(o) == _strip(ix.search(r, hits=hits))


def test_raise_is_still_the_default(segment, catalogue):
    ix = ranked_index(segment, catalogue, "cpu")
    with pytest.raises(api.UnsupportedQuery):
        ix.search_batch([{"query": {"match_all": {}}}, {"query": {"no_such_query": {}}}])
    with pytest.raises(ValueError):
        ix.search_batch([{"query": {"match_all": {}}}], errors="ignore")


def test_an_execution_time_refusal_is_attributed_to_its_request(segment, catalogue, monkeypatch):
    """A refusal raised while the batch runs (not while planning) still lands in its own slot."""
    from engine.plan import execute as E
    ix = ranked_index(segment, catalogue, "cpu")
    real = E.Executor.run

    def run(self, plans, columns=False):
        if any(p.size == 7 for p in plans):
            raise api.InvalidRequest("size", "refused while running (test)")
        return real(self, plans, columns=columns)
    monkeypatch.setattr(E.Executor, "run", run)
    reqs = [{"query": {"match_all": {}}, "size": 3}, {"query": {"match_all": {}}, "size": 7}]
    out = ix.search_batch(reqs, errors="return")
    assert out[1] == {"error": {"type": "InvalidRequest", "path": "size", "reason": "refused while running (test)"}}
    assert _strip(out[0]) == _strip(ix.search(reqs[0]))

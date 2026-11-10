"""The contract checks pass on the referee's reference index, so a failure on the engine is the engine's."""
import json

import pytest

from referee import checks, loader, reference


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


@pytest.fixture(scope="module")
def ref(ds):
    return checks.reference_for(ds)


def test_api_surface():
    checks.check_api_surface(reference)


def test_lexical_scores(ref, ds):
    checks.check_lexical_scores(checks.reference_for(ds), ds, ref)


def test_errors(ref):
    """The reference implements none of S7, so it refuses every construct in referee/scope.py."""
    checks.check_errors(ref, reference, honor_landed=False)


def test_scope_table_catches_drift(ds):
    """A construct answered before it is marked landed fails the check, and so does one marked landed but refused."""
    import copy
    from referee import scope

    class AnswersPhraseAndFuzzy:
        def __init__(self, inner):
            self.inner = inner

        def search(self, r):
            q = json.dumps(r)
            if "match_phrase" in q or "fuzzy" in q:
                return {"hits": {"total": 0, "hits": []}}
            return self.inner.search(r)

    with pytest.raises(AssertionError, match="fuzzy"):        # fuzzy answered, S7-2 not marked landed
        checks.check_errors(AnswersPhraseAndFuzzy(checks.reference_for(ds)), reference)
    saved = copy.deepcopy(scope.SCOPE)
    try:                                                      # marked landed, but the implementation refuses it
        scope.SCOPE[:] = [(n, i, True if i == "S7-3" else l, f, r) for n, i, l, f, r in scope.SCOPE]
        with pytest.raises(AssertionError, match="marked landed"):
            checks.check_landed_are_answered(checks.reference_for(ds), reference)
    finally:
        scope.SCOPE[:] = saved
    names = [e[0] for e in scope.SCOPE]
    assert len(names) == len(set(names))
    assert {"query_string", "wildcard", "regexp", "nested", "dis_max", "constant_score", "terms lookup", "suggest",
            "pit", "profile", "highlight keyword field"} <= {e[0] for e in scope.SCOPE if e[1] is None}


@pytest.mark.parametrize("check", checks.ALL_INDEX_CHECKS, ids=lambda f: f.__name__)
def test_index(check, ref, ds):
    check(ref, ds)


def test_checks_catch_a_wrong_engine(ds, ref):
    """A planted error (k1 off by 10%) must fail the lexical check: the check can see what it is meant to see."""
    wrong = checks.reference_for(ds)
    wrong.k1 = 1.32
    with pytest.raises(AssertionError):
        checks.check_lexical_scores(wrong, ds, ref)


def test_checks_catch_a_wrong_facet(ds):
    class Off:
        def __init__(self, inner):
            self.inner = inner

        def search(self, r):
            out = self.inner.search(r)
            for a in out.get("aggregations", {}).values():
                if a["buckets"]:
                    a["buckets"][0]["doc_count"] += 1
            return out
    with pytest.raises(AssertionError):
        checks.check_facets(Off(checks.reference_for(ds)), ds)


def test_checks_catch_a_wrong_page(ds):
    class Off:
        def __init__(self, inner):
            self.inner = inner

        def search(self, r):
            out = self.inner.search(r)
            if r.get("search_after") is not None and len(out["hits"]["hits"]) > 1:
                out["hits"]["hits"] = out["hits"]["hits"][1:]  # drops one hit per page
            return out
    with pytest.raises(AssertionError):
        checks.check_pagination(Off(checks.reference_for(ds)), ds)

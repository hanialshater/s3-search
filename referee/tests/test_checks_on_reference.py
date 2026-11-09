"""The contract checks pass on the referee's reference index, so a failure on the engine is the engine's."""
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
    checks.check_errors(ref, reference)


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

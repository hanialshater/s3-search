"""The engine's contract on `toy`, on every device `engine.api.available_devices()` reports (cpu in CI; cuda in Colab).

`python -m referee fast` runs this folder once `engine/api.py` exists. The checks are in `referee/checks.py`; the
same checks pass on the referee's reference index (`referee/tests/test_checks_on_reference.py`).

Owner: evaluation-steward. Makers do not edit this folder (P6); a maker who thinks a check is wrong raises it.
"""
import pytest

api = pytest.importorskip("engine.api")

from referee import checks, loader  # noqa: E402

DEVICES = api.available_devices()


@pytest.fixture(scope="module")
def ds():
    return loader.load_dataset("toy")


@pytest.fixture(scope="module")
def ref(ds):
    return checks.reference_for(ds)


@pytest.fixture(scope="module", params=DEVICES)
def index(request, ds):
    return api.build_index(ds.catalogue, ds.mapping, device=request.param)


def test_api_surface():
    checks.check_api_surface(api)


def test_lexical_scores_match_the_reference(index, ds, ref):
    checks.check_lexical_scores(index, ds, ref)


def test_errors_are_named(index):
    """Everything referee/scope.py says must be refused is refused, naming its path."""
    checks.check_errors(index, api)


def test_landed_constructs_are_answered(index):
    """Everything referee/scope.py marks landed (S7) is answered."""
    checks.check_landed_are_answered(index, api)


def test_landed_constructs_are_correct(index, ds):
    """Every landed S7 item's match sets equal brute force (referee/s7_checks.py)."""
    from referee import s7_checks
    s7_checks.check_landed_correct(index, ds)


class _Rewrites:
    """A planted error: rewrites one construct into a looser one before the engine sees it."""

    def __init__(self, inner, fn):
        self.inner, self.fn = inner, fn

    def search(self, r):
        return self.inner.search(self.fn(r))


def _phrase_as_match(r):
    q = r.get("query", {})
    if "match_phrase" in q:
        (f, v), = q["match_phrase"].items()
        return {**r, "query": {"match": {f: {"query": v["query"] if isinstance(v, dict) else v, "operator": "and"}}}}
    return r


def _fuzzy_as_term(r):
    q = r.get("query", {})
    if "fuzzy" in q:
        (f, v), = q["fuzzy"].items()
        return {**r, "query": {"match": {f: v}}}
    return r


def _post_filter_into_query(r):
    if "post_filter" in r:
        r = dict(r)
        pf = r.pop("post_filter")
        return {**r, "query": {"bool": {"must": [r.get("query", {"match_all": {}})], "filter": [pf]}}}
    return r


def _collapse_ignored(r):
    return {k: v for k, v in r.items() if k != "collapse"}


class _DropsEmptyBuckets:
    def __init__(self, inner):
        self.inner = inner

    def search(self, r):
        out = self.inner.search(r)
        for a in (out.get("aggregations") or {}).values():
            if isinstance(a, dict) and "buckets" in a and a["buckets"] and "from" not in a["buckets"][0] \
                    and "to" not in a["buckets"][0]:
                a["buckets"] = [b for b in a["buckets"] if b["doc_count"]]
        return out


def _avg_of_price(r):
    aggs = r.get("aggs")
    if aggs and any("avg" in v for v in aggs.values()):
        r = {**r, "aggs": {k: ({"avg": {"field": "price"}} if "avg" in v else v) for k, v in aggs.items()}}
    return r


class _BadHighlight:
    """Planted: tags one more word that is not a query term, or drops the last fragment of each field."""

    def __init__(self, inner, mode):
        self.inner, self.mode = inner, mode

    def search(self, r):
        out = self.inner.search(r)
        for h in out["hits"]["hits"]:
            for f, frags in (h.get("highlight") or {}).items():
                if self.mode == "extra" and frags:
                    import re
                    words = [w for w in re.findall(r"(?<![<>/\w])([A-Za-z]{4,})(?![^<]*</em>)", frags[0])]
                    if words:
                        frags[0] = frags[0].replace(words[0], f"<em>{words[0]}</em>", 1)
                elif self.mode == "drop" and frags:
                    h["highlight"][f] = []
        return out


def _boost_mode_ignored(r):
    q = r.get("query", {})
    if "function_score" in q:
        return {**r, "query": {"function_score": {k: v for k, v in q["function_score"].items()
                                                  if k not in ("boost_mode", "min_score")}}}
    return r


def _build_without_synonyms(cat, mapping, device="cpu"):
    import copy
    m = copy.deepcopy(mapping)
    m["_meta"]["engine"].pop("synonyms", None)
    return api.build_index(cat, m, device=device)


def _build_two_way(cat, mapping, device="cpu"):
    import copy
    m = copy.deepcopy(mapping)
    m["_meta"]["engine"]["synonyms"] = [x.replace("=>", ",") for x in m["_meta"]["engine"].get("synonyms", [])]
    return api.build_index(cat, m, device=device)


def test_correctness_checks_catch_planted_errors(index, ds):
    from referee import s7_checks
    assert s7_checks.check_s7_6(index, ds, build=_build_without_synonyms), "ignored synonyms must be caught"
    assert s7_checks.check_s7_6(index, ds, build=_build_two_way), "a one-way rule read both ways must be caught"
    assert s7_checks.check_s7_5(_BadHighlight(index, "extra"), ds), "a highlighted non-query word must be caught"
    assert s7_checks.check_s7_5(_BadHighlight(index, "drop"), ds), "a missing highlight must be caught"
    assert s7_checks.check_s7_5(_Rewrites(index, _boost_mode_ignored), ds), "an ignored boost_mode must be caught"
    assert s7_checks.check_s7_4(_DropsEmptyBuckets(index), ds), "a histogram without its empty buckets must be caught"
    assert s7_checks.check_s7_4(_Rewrites(index, _avg_of_price), ds), "an avg over the wrong field must be caught"
    assert s7_checks.check_s7_3(_Rewrites(index, _post_filter_into_query), ds), "facets after post_filter must be caught"
    assert s7_checks.check_s7_3(_Rewrites(index, _collapse_ignored), ds), "an ignored collapse must be caught"
    assert s7_checks.check_s7_1(_Rewrites(index, _phrase_as_match), ds), "a bag-of-words phrase must be caught"
    assert s7_checks.check_s7_2(_Rewrites(index, _fuzzy_as_term), ds), "fuzzy answered as an exact term must be caught"


@pytest.mark.parametrize("check", checks.ALL_INDEX_CHECKS, ids=lambda f: f.__name__)
def test_index(check, index, ds):
    check(index, ds)


def test_devices_agree(ds):
    """Every device gives the cpu path's results: scores to rtol 1e-5, ids exactly except inside tie groups."""
    if len(DEVICES) < 2:
        pytest.skip("only one device here")
    base = api.build_index(ds.catalogue, ds.mapping, device="cpu")
    for dev in DEVICES[1:]:
        other = api.build_index(ds.catalogue, ds.mapping, device=dev)
        for r in checks.lexical_requests(ds)[:60]:
            checks.assert_same_ranking(other.search(r), base.search(r), f"{dev} vs cpu: {r['query']}",
                                       tol=checks.DEVICE_RTOL)


# ---- synonyms are a mapping setting (S7-6): what a mapping may hold ----------------------------------------------------

_BAD_SYNONYMS = [(42, "not a list"), ([42], "an entry that is not a string"), (["red lamp, light"], "a multi-word entry"),
                 (["the, lamp"], "an entry that analyses to no token (a stop word)")]
_BAD_SYNONYMS_DEFECTS = [(["lamp =>"], "a rule with an empty right side (accepted, and makes lamp match nothing)"),
                         (["=> light"], "a rule with an empty left side"),
                         ([" , "], "an entry of empty alternatives (OpenSearch rejects the same mapping)")]


def _with_synonyms(ds, syn):
    import copy
    m = copy.deepcopy(ds.mapping)
    m["_meta"]["engine"]["synonyms"] = syn
    return m


@pytest.mark.parametrize("syn,what", _BAD_SYNONYMS, ids=[w for _, w in _BAD_SYNONYMS])
def test_synonym_mappings_refused(syn, what, ds):
    with pytest.raises(api.InvalidMapping):
        api.build_index(ds.catalogue, _with_synonyms(ds, syn))


@pytest.mark.xfail(strict=True, reason="defect reported 2026-11-10 (steward): accepted instead of InvalidMapping; "
                                       "remove this mark when the engine refuses it")
@pytest.mark.parametrize("syn,what", _BAD_SYNONYMS_DEFECTS, ids=[w[:40] for _, w in _BAD_SYNONYMS_DEFECTS])
def test_synonym_mappings_refused_defects(syn, what, ds):
    with pytest.raises(api.InvalidMapping):
        api.build_index(ds.catalogue, _with_synonyms(ds, syn))


def test_synonym_mappings_accepted(ds):
    for syn in ([], ["# only a comment"], ["lamp, light"], "lamp, light", ["kettle => mug", "boots, shoe"]):
        api.build_index(ds.catalogue, _with_synonyms(ds, syn))

"""OpenSearch 2.19's recorded answers to the conformance suite: they load, match the suite, and Lucene's BM25 model
(referee/lucene_model.py) reproduces their `match` scores, so score differences with the engine are by design."""
from referee import checks, conformance, loader, lucene_model


def test_recording_is_of_this_suite_and_complete():
    suite = conformance.build_suite()
    rec = conformance.Recorded(suite)
    assert rec.meta["suite_version"] == conformance.SUITE_VERSION
    assert all(rec.search(s["request"]) is not None for s in suite if s["family"] != "unsupported")


def test_lucene_model_reproduces_opensearch_match_scores():
    ds = loader.load_dataset("toy")
    suite = conformance.build_suite(ds)
    rec = conformance.Recorded(suite)
    ids = list(ds.catalogue["product_id"])
    n = 0
    for s in suite:
        if s["family"] != "match":
            continue
        (field, text), = s["request"]["query"]["match"].items()
        model = lucene_model.match_scores(ds.catalogue[field].tolist(), text)
        for h in rec.search(s["request"])["hits"]["hits"]:
            assert abs(model[ids.index(h["_id"])] - h["_score"]) <= 1e-5 * max(1.0, h["_score"])
            n += 1
    assert n > 50


def test_reference_against_opensearch_by_family():
    ds = loader.load_dataset("toy")
    suite = conformance.build_suite(ds)
    rep = conformance.run_suite_os(checks.reference_for(ds), conformance.Recorded(suite), suite, os_scores=False)
    assert rep["pass_rate"] == 1.0, rep["failures"][:3]


def test_lucene_length_encoding():
    assert [lucene_model.lucene_length(n) for n in (0, 5, 23, 24)] == [0, 5, 23, 24]
    assert lucene_model.lucene_length(1000) != 1000  # lossy above 24

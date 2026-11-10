"""OpenSearch 2.19's recorded answers to suite v2: they load and match the suite; Lucene's BM25 model
(referee/lucene_model.py) reproduces their `match` scores, so lexical differences with the engine are by design; the
reference agrees with every exact-operator answer."""
from referee import checks, conformance, loader, lucene_model


def test_recording_is_of_this_suite_and_complete():
    suite = conformance.build_suite()
    rec = conformance.Recorded(suite)
    assert rec.meta["suite_version"] == conformance.SUITE_VERSION == 2
    for s in suite:
        o = conformance.outcome(rec, s)
        assert "ok" in o, (s["family"], o)   # OpenSearch answers everything, the unsupported family included


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
        for h in conformance.outcome(rec, s)["ok"]["hits"]["hits"]:
            assert abs(model[ids.index(h["_id"])] - h["_score"]) <= 1e-5 * max(1.0, h["_score"])
            n += 1
    assert n > 20


def test_reference_against_opensearch_on_exact_operators():
    ds = loader.load_dataset("toy")
    suite = conformance.build_suite(ds)
    rep = conformance.judge(checks.reference_for(ds), conformance.Recorded(suite), suite)
    assert rep["p1"]["rate"] == 1.0, rep["failures"][:3]
    assert rep["p2"]["rate"] == 1.0


def test_lucene_length_encoding():
    assert [lucene_model.lucene_length(n) for n in (0, 5, 23, 24)] == [0, 5, 23, 24]
    assert lucene_model.lucene_length(1000) != 1000  # lossy above 24

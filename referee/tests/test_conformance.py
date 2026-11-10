"""Conformance suite v2: 200 fixed requests with roles; the reference passes against itself; planted errors are seen."""
from referee import checks, conformance, loader


def test_suite_is_fixed_and_has_the_roles():
    a, b = conformance.build_suite(), conformance.build_suite()
    assert len(a) == conformance.N_REQUESTS == 200
    assert conformance.suite_sha256(a) == conformance.suite_sha256(b)
    roles = {r: sum(s["role"] == r for s in a) for r in ("p1", "p2", "info")}
    assert roles == {"p1": 160, "p2": 16, "info": 24}
    for s in a:  # every p1 request has a total order no score decides, or asks for counts only
        if s["role"] == "p1":
            assert s["request"].get("size") == 0 or s["request"]["sort"][-1] == conformance.TIE


def test_reference_passes_against_itself():
    ds = loader.load_dataset("toy")
    rep = conformance.judge(checks.reference_for(ds), checks.reference_for(ds), conformance.build_suite(ds))
    assert rep["p1"]["rate"] == 1.0 and rep["p2"]["rate"] == 1.0, rep["failures"][:3]
    assert all(f["rate"] == 1.0 for f in rep["by_family"].values())


class _DropsAHit:
    def __init__(self, inner):
        self.inner = inner

    def search(self, r):
        out = self.inner.search(r)
        if len(out["hits"]["hits"]) > 2:
            out["hits"]["hits"] = out["hits"]["hits"][:1] + out["hits"]["hits"][2:]
        return out


class _AnswersEverything:
    def __init__(self, inner):
        self.inner = inner

    def search(self, r):
        try:
            return self.inner.search(r)
        except ValueError:
            return {"hits": {"total": 0, "hits": []}}


def test_planted_errors_fail_the_right_prediction():
    ds = loader.load_dataset("toy")
    suite = conformance.build_suite(ds)
    ref = checks.reference_for(ds)
    rep = conformance.judge(_DropsAHit(checks.reference_for(ds)), ref, suite)
    assert rep["p1"]["rate"] < 0.98 and rep["by_family"]["search_after"]["rate"] < 1.0
    rep = conformance.judge(_AnswersEverything(checks.reference_for(ds)), ref, suite)
    assert rep["p2"]["passed"] == 0 and rep["p1"]["rate"] == 1.0

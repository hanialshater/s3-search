"""The conformance suite: 200 fixed requests; the reference passes against itself; a planted error is seen."""
from referee import checks, conformance, loader


def test_suite_is_fixed_and_deterministic():
    a, b = conformance.build_suite(), conformance.build_suite()
    assert len(a) == conformance.N_REQUESTS == 200
    assert conformance.suite_sha256(a) == conformance.suite_sha256(b)
    assert len({s["family"] for s in a}) == 12


def test_reference_passes_against_itself_and_a_planted_error_fails():
    ds = loader.load_dataset("toy")
    ref = checks.reference_for(ds)
    rep = conformance.run_suite(checks.reference_for(ds), ref)
    assert rep["pass_rate"] == 1.0, rep["failures"][:3]
    wrong = checks.reference_for(ds)
    wrong.b = 0.6
    rep = conformance.run_suite(wrong, ref)
    assert rep["pass_rate"] < 0.98 and rep["failures"]

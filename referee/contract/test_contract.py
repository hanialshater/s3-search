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
    """Everything referee/scope.py marks landed (S7) is answered; its correctness is S7-9's conformance."""
    checks.check_landed_are_answered(index, api)


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

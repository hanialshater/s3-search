"""The operator on watches (no leader election here): change-to-reconcile latency against a real kube-apiserver."""
import statistics
import time

import pytest

from deploy.tests.test_envtest import Sim, SEG_A  # noqa: F401  (the simulator of the controllers that do not run)

pytestmark = pytest.mark.envtest


def wait_for(fn, timeout=10.0, tick=0.01):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        v = fn()
        if v:
            return time.perf_counter() - t0, v
        time.sleep(tick)
    raise AssertionError("timed out")


@pytest.fixture
def sim(kube, ns):
    return Sim(kube, ns)


def test_watches_reconcile_changes_within_a_second_without_resync(sim, operators):
    operators("solo", "--no-leader-elect", "--resync", "3600", "--namespace", sim.ns)   # resync cannot be what we see
    time.sleep(1.0)                                                                       # watches established
    sim.mapping()
    lat = []
    sim.create()
    t, st = wait_for(lambda: (sim.get().get("status") or {}).get("phase"))
    lat.append(t)
    assert st == "Indexing"
    (job,) = sim.jobs()
    # a child event (the Job finishing) reconciles its index with no change to the SearchIndex itself
    t0 = time.perf_counter()
    sim.finish_job(job["metadata"]["name"], SEG_A, 60)
    t, _ = wait_for(lambda: (sim.get()["status"].get("indexJob") or {}).get("segmentId") == SEG_A)
    child_latency = time.perf_counter() - t0
    # every spec change: observedGeneration catches up
    for i in range(10):
        gen = sim.update(reindexToken=f"t{i}")["metadata"]["generation"]
        t, _ = wait_for(lambda: sim.get()["status"].get("observedGeneration") == gen)
        lat.append(t)
    print(f"\nWATCH LATENCY spec change to status: n={len(lat)} p50={statistics.median(lat) * 1000:.0f} ms "
          f"max={max(lat) * 1000:.0f} ms; Job finish to status: {child_latency * 1000:.0f} ms")
    assert max(lat) < 1.0 and child_latency < 1.0


def test_resync_is_the_backstop_for_unwatched_objects(sim, operators):
    """A ConfigMap edit is not watched; the resync picks it up."""
    sim.mapping()
    sim.create()
    operators("solo", "--no-leader-elect", "--resync", "2", "--namespace", sim.ns)
    wait_for(lambda: (sim.get().get("status") or {}).get("indexJob"))
    first = sim.get()["status"]["indexJob"]["name"]
    sim.mapping(data={"properties": {"product_title": {"type": "text"}, "price": {"type": "float"}}})
    wait_for(lambda: sim.get()["status"]["indexJob"]["name"] != first, timeout=8)

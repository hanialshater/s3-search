"""Two operator processes against a real kube-apiserver: one leads, the other takes over. Default timings
(lease 15 s, renew deadline 10 s, retry 2 s), so the takeover test takes about 20 s."""
import time

import pytest

from deploy.s3operator.kube import LEASE
from deploy.s3operator.leader import LeaderElector
from deploy.tests.test_envtest import Sim
from deploy.tests.test_watches import wait_for

pytestmark = pytest.mark.envtest


@pytest.fixture
def sim(kube, ns):
    return Sim(kube, ns)


def holder(kube, ns):
    lease = kube.get(LEASE, ns, "s3-operator")
    return ((lease or {}).get("spec") or {}).get("holderIdentity")


def start_pair(operators, sim):
    args = ("--lease-namespace", sim.ns, "--namespace", sim.ns)
    a = operators("op-a", *args)
    wait_for(lambda: holder(sim.kube, sim.ns) == "op-a", 20)
    b = operators("op-b", *args)
    time.sleep(1.0)
    return a, b


def test_exactly_one_replica_reconciles_each_change_within_a_second(sim, operators):
    a, b = start_pair(operators, sim)
    assert holder(sim.kube, sim.ns) == "op-a"
    sim.mapping()
    sim.create()
    lat = []
    t, _ = wait_for(lambda: (sim.get().get("status") or {}).get("phase"))
    lat.append(t)
    for i in range(8):
        gen = sim.update(reindexToken=f"t{i}")["metadata"]["generation"]
        t, _ = wait_for(lambda: sim.get()["status"].get("observedGeneration") == gen)
        lat.append(t)
    time.sleep(0.5)
    print(f"\nLEADER LATENCY change to reconciled: n={len(lat)} max={max(lat) * 1000:.0f} ms; "
          f"reconciles leader={a.reconciles()} standby={b.reconciles()}")
    assert a.reconciles() >= 9 and b.reconciles() == 0         # the standby never reconciles
    assert "became leader" in a.text() and "became leader" not in b.text()
    assert max(lat) < 1.0


def test_standby_takes_over_within_20_seconds_when_the_leader_is_killed(sim, operators):
    a, b = start_pair(operators, sim)
    sim.mapping()
    sim.create()
    wait_for(lambda: (sim.get().get("status") or {}).get("phase"))
    t0 = time.perf_counter()
    a.kill()                                                    # SIGKILL: no release, the lease must expire
    wait_for(lambda: holder(sim.kube, sim.ns) == "op-b", 25, tick=0.1)
    takeover = time.perf_counter() - t0
    gen = sim.update(reindexToken="after-failover")["metadata"]["generation"]
    t, _ = wait_for(lambda: sim.get()["status"].get("observedGeneration") == gen, 5)
    print(f"\nTAKEOVER after kill -9: lease held by the standby after {takeover:.1f} s; "
          f"first change reconciled {t * 1000:.0f} ms later; Lease transitions="
          f"{sim.kube.get(LEASE, sim.ns, 's3-operator')['spec']['leaseTransitions']}")
    assert takeover < 20 and b.reconciles() >= 1
    assert sim.kube.get(LEASE, sim.ns, "s3-operator")["spec"]["leaseTransitions"] == 1


def test_clean_stop_releases_the_lease_for_a_fast_handover(sim, operators):
    a, b = start_pair(operators, sim)
    t0 = time.perf_counter()
    a.proc.terminate()
    wait_for(lambda: holder(sim.kube, sim.ns) == "op-b", 15, tick=0.1)
    handover = time.perf_counter() - t0
    print(f"\nHANDOVER after SIGTERM: {handover:.1f} s")
    assert handover < 6


def test_elector_rejects_inconsistent_timings():
    with pytest.raises(ValueError):
        LeaderElector(None, "ns", "n", "id", lease_duration=5, renew_deadline=10, retry_period=2)

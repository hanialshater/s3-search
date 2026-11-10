"""Leader election on a ``coordination.k8s.io/v1`` Lease, the way client-go does it.

Every operator replica runs ``LeaderElector.run``. One holds the Lease and renews it every ``retry_period``; the others
watch it. A candidate takes over when it has seen the Lease unchanged for ``lease_duration`` by its own clock (it never
compares its clock with the holder's, so clock skew cannot split the brain). A holder that cannot renew for
``renew_deadline`` stops leading at once, which is before any candidate can take over (renew_deadline < lease_duration).
Defaults: lease duration 15 s, renew deadline 10 s, retry period 2 s. A holder that is stopped cleanly releases the Lease,
so a rolling update of the operator hands over in about one retry period.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Callable

from .kube import LEASE, ApiError, Kube

log = logging.getLogger("s3operator.leader")


def micro_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class LeaderElector:
    def __init__(self, kube: Kube, namespace: str, name: str, identity: str, lease_duration: float = 15.0,
                 renew_deadline: float = 10.0, retry_period: float = 2.0, clock: Callable[[], float] = time.monotonic):
        if not (lease_duration > renew_deadline > retry_period > 0):
            raise ValueError("need lease_duration > renew_deadline > retry_period > 0")
        self.kube, self.ns, self.name, self.id = kube, namespace, name, identity
        self.lease_duration, self.renew_deadline, self.retry_period = lease_duration, renew_deadline, retry_period
        self.clock = clock
        self._observed: tuple | None = None      # (holder, renewTime) last seen
        self._observed_at = 0.0                  # local time we first saw it
        self.is_leader = False

    # -- one attempt ------------------------------------------------------------------------------------------------
    def _spec(self, transitions: int, acquire: str) -> dict:
        return {"holderIdentity": self.id, "leaseDurationSeconds": int(round(self.lease_duration)),
                "acquireTime": acquire, "renewTime": micro_now(), "leaseTransitions": transitions}

    def try_acquire_or_renew(self) -> bool:
        """True if we hold the Lease after this call. Any API failure or lost race is just False."""
        try:
            lease = self.kube.get(LEASE, self.ns, self.name)
            if lease is None:
                body = {"apiVersion": "coordination.k8s.io/v1", "kind": "Lease",
                        "metadata": {"name": self.name, "namespace": self.ns}, "spec": self._spec(0, micro_now())}
                self.kube.create(LEASE, body)
                self._observed, self._observed_at = (self.id, body["spec"]["renewTime"]), self.clock()
                return True
            spec = lease.get("spec") or {}
            seen = (spec.get("holderIdentity"), spec.get("renewTime"))
            if seen != self._observed:
                self._observed, self._observed_at = seen, self.clock()
            mine = spec.get("holderIdentity") == self.id
            vacant = not spec.get("holderIdentity")
            expired = self.clock() - self._observed_at > float(spec.get("leaseDurationSeconds") or self.lease_duration)
            if not (mine or vacant or expired):
                return False
            transitions = int(spec.get("leaseTransitions") or 0) + (0 if mine else 1)
            lease["spec"] = self._spec(transitions, spec.get("acquireTime") if mine and spec.get("acquireTime")
                                       else micro_now())
            out = self.kube.request("PUT", f"{LEASE.collection(self.ns)}/{self.name}", lease)  # resourceVersion: a race loses
            new = out.get("spec") or {}
            self._observed, self._observed_at = (new.get("holderIdentity"), new.get("renewTime")), self.clock()
            return True
        except ApiError as exc:
            if exc.status not in (409, 404):
                log.warning("lease %s/%s: %s", self.ns, self.name, exc)
            return False
        except Exception as exc:  # noqa: BLE001 - the API server may be briefly unreachable
            log.warning("lease %s/%s: %s", self.ns, self.name, exc)
            return False

    def release(self) -> None:
        """Hand the Lease over now (clean shutdown) so a standby does not wait out the lease duration."""
        try:
            lease = self.kube.get(LEASE, self.ns, self.name)
            if lease and (lease.get("spec") or {}).get("holderIdentity") == self.id:
                lease["spec"].update(holderIdentity=None, leaseDurationSeconds=1, renewTime=micro_now())
                self.kube.request("PUT", f"{LEASE.collection(self.ns)}/{self.name}", lease)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not release the lease: %s", exc)

    # -- the loop ---------------------------------------------------------------------------------------------------
    def run(self, on_started: Callable[[], None], on_stopped: Callable[[], None], stop: threading.Event) -> None:
        """Campaign, lead, lose or stop, repeat, until ``stop`` is set."""
        while not stop.is_set():
            if not self.try_acquire_or_renew():
                stop.wait(self.retry_period)
                continue
            self.is_leader = True
            log.info("became leader: %s/%s as %s", self.ns, self.name, self.id)
            on_started()
            last_ok = self.clock()
            while not stop.is_set():
                stop.wait(self.retry_period)
                if stop.is_set():
                    break
                if self.try_acquire_or_renew():
                    last_ok = self.clock()
                elif self.clock() - last_ok > self.renew_deadline:
                    log.error("lost the lease: no renewal for %.1fs", self.clock() - last_ok)
                    break
            self.is_leader = False
            on_stopped()
            if stop.is_set():
                self.release()

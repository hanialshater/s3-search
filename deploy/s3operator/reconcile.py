"""Reconcile one ``SearchIndex``: level-triggered, idempotent, safe to run every few seconds.

    spec ──► PVC (segments) ──► mapping ConfigMap ──► indexer Job (named by build hash) ──► segment id (termination msg)
         └─► Service                                                         │
                                         replica Deployment (segment id in the pod template, maxUnavailable 0) ◄─┘

The operator never reads the segment volume. The indexer's termination message (``{"segment_id", "version"}``) is
copied to ``status.indexJob`` and ``status.segments`` once, so deleting the finished Job loses nothing.

Phases: Pending (waiting for a mapping or the first replicas), Indexing (a build is running; replicas keep serving the
last segment), Swapping (replicas rolling to another segment, either a new build or a pin), Ready, Failed (invalid spec
or a failed build; replicas keep serving the last segment).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from . import render
from .kube import CONFIGMAP, DEPLOYMENT, JOB, POD, SEARCHINDEX, SERVICE, PVC, ApiError, Kube

log = logging.getLogger("s3operator")
MAX_HISTORY = 10


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def merge_patch(prev: dict, new: dict) -> dict:
    """A JSON merge patch that turns ``prev`` into exactly ``new``: keys that vanished are sent as null (a bare merge
    patch would keep them, e.g. a stale ``indexJob.segmentId`` after a new build starts)."""
    out: dict = {}
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(prev.get(k), dict):
            out[k] = merge_patch(prev[k], v)
        else:
            out[k] = v
    for k in prev:
        if k not in new:
            out[k] = None
    return out


class Conditions:
    """Standard status conditions; ``lastTransitionTime`` only moves when a condition's status flips."""

    def __init__(self, previous: list[dict] | None, generation: int, now: str):
        self._prev = {c["type"]: c for c in (previous or [])}
        self._new: dict[str, dict] = {}
        self.generation, self.now = generation, now

    def set(self, type_: str, ok: bool, reason: str, message: str = "") -> None:
        status = "True" if ok else "False"
        old = self._prev.get(type_)
        when = old["lastTransitionTime"] if old and old.get("status") == status else self.now
        self._new[type_] = {"type": type_, "status": status, "reason": reason, "message": message,
                            "lastTransitionTime": when, "observedGeneration": self.generation}

    def list(self) -> list[dict]:
        order = ["Valid", "MappingAvailable", "IndexBuilt", "Available", "Ready"]
        return [self._new[t] for t in order if t in self._new]


def segment_version(segment_id: str | None) -> int | None:
    try:
        return int(segment_id.split("-")[1])  # type: ignore[union-attr]
    except (AttributeError, IndexError, ValueError):
        return None


def job_state(job: dict) -> str:
    st = job.get("status") or {}
    for c in st.get("conditions") or []:
        if c.get("status") == "True" and c.get("type") == "Complete":
            return "Succeeded"
        if c.get("status") == "True" and c.get("type") == "Failed":
            return "Failed"
    return "Succeeded" if (st.get("succeeded") or 0) >= 1 else "Active"


def _terminated(pod: dict) -> dict | None:
    for cs in (pod.get("status") or {}).get("containerStatuses") or []:
        if cs.get("name") == "indexer":
            return (cs.get("state") or {}).get("terminated")
    return None


def read_result(pods: list[dict]) -> dict | None:
    """The indexer's ``{"segment_id", "version"}``, from the termination message of a pod that exited 0."""
    for pod in sorted(pods, key=lambda p: p["metadata"].get("creationTimestamp", ""), reverse=True):
        t = _terminated(pod)
        if t and t.get("exitCode") == 0:
            try:
                out = json.loads(t.get("message") or "")
            except ValueError:
                continue
            if render.SEGMENT_ID.match(str(out.get("segment_id", ""))):
                return out
    return None


def failure_message(pods: list[dict]) -> str:
    for pod in sorted(pods, key=lambda p: p["metadata"].get("creationTimestamp", ""), reverse=True):
        t = _terminated(pod)
        if t and t.get("exitCode") not in (0, None):
            tail = (t.get("message") or t.get("reason") or "").strip().splitlines()
            return f"indexer exited {t['exitCode']}: {tail[-1] if tail else ''}"
    return "the indexer job failed"


class Reconciler:
    def __init__(self, kube: Kube, now=utcnow):
        self.kube, self.now = kube, now

    # -- one resource -----------------------------------------------------------------------------------------------
    def reconcile(self, si: dict) -> dict:
        md = si["metadata"]
        ns, name = md["namespace"], md["name"]
        prev = si.get("status") or {}
        now = self.now()
        conds = Conditions(prev.get("conditions"), md.get("generation", 1), now)
        status: dict = {"observedGeneration": md.get("generation", 1), "segments": prev.get("segments", []),
                        "segmentId": prev.get("segmentId"), "version": prev.get("version"),
                        "latestSegmentId": prev.get("latestSegmentId"), "indexJob": prev.get("indexJob")}
        spec = render.spec_of(si)
        status["replicas"] = spec["replicas"]

        def done(phase: str, message: str = "") -> dict:
            status["phase"], status["message"] = phase, message
            conds.set("Ready", phase == "Ready", phase if phase != "Ready" else "AllReplicasReady", message)
            status["conditions"] = conds.list()
            clean = {k: v for k, v in status.items() if v is not None}
            old = {k: v for k, v in prev.items() if v is not None}
            if clean != old:
                self.kube.patch_status(SEARCHINDEX, ns, name, merge_patch(old, clean))
            return clean

        errs = render.validate(si)
        if errs:
            conds.set("Valid", False, "InvalidSpec", "; ".join(errs))
            return done("Failed", "; ".join(errs))
        conds.set("Valid", True, "SpecValid")

        # storage and service do not depend on a build
        pvc = render.render_pvc(si)
        if pvc:
            self.kube.apply(PVC, pvc)
        self.kube.apply(SERVICE, render.render_service(si))

        # the mapping
        ref = spec["mapping"]["configMapRef"]
        key = ref.get("key", "mapping.json")
        cm = self.kube.get(CONFIGMAP, ns, ref["name"])
        if cm is None or key not in (cm.get("data") or {}):
            why = f"ConfigMap {ref['name']!r} not found" if cm is None else f"ConfigMap {ref['name']!r} has no key {key!r}"
            conds.set("MappingAvailable", False, "MappingMissing", why)
            return done("Pending", why)
        conds.set("MappingAvailable", True, "MappingFound")

        # the indexer job for this build
        digest = render.build_hash(si, cm["data"])
        job_name = f"{render.names(si)['job_prefix']}{digest}"
        job = self.kube.get(JOB, ns, job_name)
        if job is None:
            try:
                job = self.kube.create(JOB, render.render_job(si, digest))
            except ApiError as e:
                if e.status != 409:
                    raise
                job = self.kube.get(JOB, ns, job_name) or {}
        state = job_state(job)
        pods = self.kube.list(POD, ns, f"job-name={job_name}") if state != "Active" else []
        ij = {"name": job_name, "hash": digest, "state": state}
        if state == "Succeeded":
            known = prev.get("indexJob") or {}
            result = {"segment_id": known.get("segmentId"), "version": known.get("version")} \
                if known.get("hash") == digest and known.get("segmentId") else read_result(pods)
            if result is None:
                conds.set("IndexBuilt", False, "ResultUnavailable", "the job succeeded but its segment id is unreadable")
                status["indexJob"] = {**ij, "state": "Failed"}
                return done("Failed", "the indexer job succeeded but its result (segment id) cannot be read; "
                                      "change spec.reindexToken to build again")
            ij.update(segmentId=result["segment_id"], version=result.get("version"))
            status["latestSegmentId"] = result["segment_id"]
            hist = [h for h in status["segments"] if h["segmentId"] != result["segment_id"]]
            hist.append({"segmentId": result["segment_id"], "version": result.get("version"), "build": digest})
            status["segments"] = hist[-MAX_HISTORY:]
            conds.set("IndexBuilt", True, "IndexBuilt", f"segment {result['segment_id']}")
        elif state == "Failed":
            ij["message"] = failure_message(pods or self.kube.list(POD, ns, f"job-name={job_name}"))
            conds.set("IndexBuilt", False, "IndexFailed", ij["message"])
        else:
            conds.set("IndexBuilt", False, "Indexing", f"job {job_name} is running")
        status["indexJob"] = ij

        # which segment the replicas should serve
        live = self.kube.get(DEPLOYMENT, ns, render.names(si)["deployment"])
        live_seg = (((live or {}).get("spec") or {}).get("template", {}).get("metadata", {}).get("annotations") or {}) \
            .get(render.ANNOTATION_SEGMENT)
        pin = spec.get("segmentVersion")
        target = status.get("latestSegmentId")
        notes = ""
        if pin:
            known_ids = {h["segmentId"] for h in status["segments"]}
            if spec["storage"].get("existingClaim") or pin in known_ids or (not known_ids and state == "Succeeded"):
                target = pin
            else:
                notes = (f"segmentVersion {pin} is not a segment this resource built "
                         f"({', '.join(sorted(known_ids)) or 'none yet'}); keeping {live_seg or target}")
                target = live_seg or target
        if target is None:
            return done("Failed" if state == "Failed" else ("Indexing" if state == "Active" else "Pending"),
                        ij.get("message") or notes or "waiting for the first segment")
        dep = self.kube.apply(DEPLOYMENT, render.render_deployment(si, target))

        # rollout state
        ds = dep.get("status") or {}
        gen_ok = (ds.get("observedGeneration") or 0) >= dep["metadata"].get("generation", 0)
        desired, ready = spec["replicas"], ds.get("readyReplicas") or 0
        rolled = gen_ok and (ds.get("updatedReplicas") or 0) == desired and ready >= desired \
            and (ds.get("replicas") or 0) == desired
        status["readyReplicas"] = ready
        conds.set("Available", ready >= 1 or desired == 0, "MinimumReplicasAvailable" if ready >= 1 else "NoReadyReplicas",
                  f"{ready}/{desired} replicas ready")
        if rolled:
            status["segmentId"], status["version"] = target, segment_version(target)
        if state == "Succeeded":  # old build Jobs (and their pods) are no longer needed
            for j in self.kube.list(JOB, ns, f"{render.LABEL_INDEX}={name}"):
                jn = j["metadata"]["name"]
                if jn.startswith(render.names(si)["job_prefix"]) and jn != job_name:
                    self.kube.delete(JOB, ns, jn)

        if state == "Failed":
            return done("Failed", ij["message"])
        if rolled and state == "Active":
            return done("Indexing", f"serving {target}; building the next segment")
        if rolled:
            return done("Ready", notes or f"{ready}/{desired} replicas serving {target}")
        if prev.get("segmentId") and prev["segmentId"] != target:
            return done("Swapping", f"rolling {prev['segmentId']} -> {target}: {ready}/{desired} ready")
        return done("Pending", f"waiting for replicas: {ready}/{desired} ready")

    # -- all resources ----------------------------------------------------------------------------------------------
    def reconcile_all(self, namespace: str | None = None) -> int:
        n = 0
        for si in self.kube.list(SEARCHINDEX, namespace):
            if si["metadata"].get("deletionTimestamp"):
                continue
            try:
                self.reconcile(si)
            except Exception:  # noqa: BLE001 - one broken resource must not stop the others
                log.exception("reconcile %s/%s failed", si["metadata"]["namespace"], si["metadata"]["name"])
            n += 1
        return n

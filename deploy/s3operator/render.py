"""Pure functions: a ``SearchIndex`` (as a dict) becomes the objects the operator keeps in the cluster.

Nothing here talks to Kubernetes, so the rendering is unit-tested in milliseconds (deploy/tests/test_render.py).

The rules (docs/INTERFACES.md §7, decision docs/decisions/2026-11-09-operator-language.md):
- the indexer Job is named by a hash of everything that changes what it builds, so a change creates a new Job and
  never edits a finished one;
- the replica Deployment rolls with ``maxUnavailable: 0`` and carries the segment id in its pod template, so a new
  segment is a rolling update and a pinned ``segmentVersion`` (a roll back) is the same update in the other direction;
- a GPU is never assumed: only ``device: cuda`` adds ``nvidia.com/gpu`` limits, a nodeSelector and a toleration, and
  only to the replicas (the indexer builds on the host with the numpy reference).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re

GROUP, VERSION, KIND = "search.s3.dev", "v1alpha1", "SearchIndex"
API_VERSION = f"{GROUP}/{VERSION}"
GPU = "nvidia.com/gpu"
DEVICES = ("cpu", "torch-cpu", "cuda")
SEGMENT_ID = re.compile(r"^seg-[0-9]+-[0-9a-f]{8}$")
HTTP_PORT = 8080
RUN_AS = 10001
MANAGER = "s3-operator"

LABEL_INDEX = "search.s3.dev/index"
ANNOTATION_SEGMENT = "search.s3.dev/segment-id"
DEFAULT_RESOURCES = {"requests": {"cpu": "500m", "memory": "1Gi"}}
DRAIN_SECONDS = 5


def default_image(device: str, env: dict | None = None) -> str:
    e = os.environ if env is None else env
    if device == "cuda":
        return e.get("S3_OPERATOR_IMAGE_CUDA", "s3-search:cuda")
    return e.get("S3_OPERATOR_IMAGE_CPU", "s3-search:cpu")


def spec_of(si: dict) -> dict:
    """The spec with the CRD's defaults applied (the API server also applies them; this keeps unit tests honest)."""
    s = copy.deepcopy(si.get("spec") or {})
    s.setdefault("device", "cpu")
    s.setdefault("replicas", 1)
    s.setdefault("storage", {})
    s["storage"].setdefault("size", "5Gi")
    s["storage"].setdefault("accessModes", ["ReadWriteOnce"])
    s["image"] = s.get("image") or default_image(s["device"])
    return s


def validate(si: dict) -> list[str]:
    """Why this spec cannot be reconciled (empty list: fine). Checked in the operator as well as by the CRD schema."""
    s = spec_of(si)
    errs: list[str] = []
    if s["device"] not in DEVICES:
        errs.append(f"spec.device must be one of {', '.join(DEVICES)}, not {s['device']!r}")
    if not isinstance(s["replicas"], int) or s["replicas"] < 0:
        errs.append("spec.replicas must be an integer >= 0")
    cm = ((s.get("mapping") or {}).get("configMapRef") or {})
    if not cm.get("name"):
        errs.append("spec.mapping.configMapRef.name is required")
    if not (s.get("source") or {}).get("path"):
        errs.append("spec.source.path is required")
    pin = s.get("segmentVersion")
    if pin is not None and not SEGMENT_ID.match(str(pin)):
        errs.append(f"spec.segmentVersion must be a segment id like seg-1044-3fa2c9d1, not {pin!r}")
    res = s.get("resources") or {}
    if s["device"] != "cuda" and (GPU in (res.get("limits") or {}) or GPU in (res.get("requests") or {})):
        errs.append(f"spec.resources asks for {GPU} but spec.device is {s['device']!r}; "
                    "set device: cuda for a GPU (the operator never assumes one)")
    return errs


# ---- names and labels -------------------------------------------------------------------------------------------------
def names(si: dict) -> dict:
    n = si["metadata"]["name"]
    claim = (spec_of(si)["storage"].get("existingClaim")) or f"{n}-data"
    return {"pvc": claim, "deployment": f"{n}-replicas", "service": n, "job_prefix": f"{n}-index-"}


def labels(si: dict, component: str) -> dict:
    return {"app.kubernetes.io/name": "searchindex", "app.kubernetes.io/instance": si["metadata"]["name"],
            "app.kubernetes.io/component": component, "app.kubernetes.io/managed-by": MANAGER,
            LABEL_INDEX: si["metadata"]["name"]}


def selector(si: dict) -> dict:
    return {"app.kubernetes.io/instance": si["metadata"]["name"], "app.kubernetes.io/component": "replica"}


def owner_ref(si: dict) -> list[dict]:
    return [{"apiVersion": API_VERSION, "kind": KIND, "name": si["metadata"]["name"], "uid": si["metadata"]["uid"],
             "controller": True, "blockOwnerDeletion": True}]


def _meta(si: dict, name: str, component: str, extra_annotations: dict | None = None) -> dict:
    m = {"name": name, "namespace": si["metadata"]["namespace"], "labels": labels(si, component),
         "ownerReferences": owner_ref(si)}
    if extra_annotations:
        m["annotations"] = extra_annotations
    return m


def build_hash(si: dict, mapping_data: dict) -> str:
    """Everything that changes what the indexer builds. ``segmentVersion`` and ``replicas`` are not in it."""
    s = spec_of(si)
    blob = json.dumps({"mapping": mapping_data, "source": s.get("source"), "image": s["image"],
                       "token": s.get("reindexToken", "")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


# ---- children ---------------------------------------------------------------------------------------------------------
def render_pvc(si: dict) -> dict | None:
    st = spec_of(si)["storage"]
    if st.get("existingClaim"):
        return None
    spec = {"accessModes": st["accessModes"], "resources": {"requests": {"storage": st["size"]}}}
    if st.get("storageClassName"):
        spec["storageClassName"] = st["storageClassName"]
    return {"apiVersion": "v1", "kind": "PersistentVolumeClaim", "metadata": _meta(si, names(si)["pvc"], "storage"),
            "spec": spec}


def _resources(s: dict, gpu: bool) -> dict:
    res = copy.deepcopy(s.get("resources") or DEFAULT_RESOURCES)
    for part in ("requests", "limits"):
        if part in res:
            res[part].pop(GPU, None)
    if gpu:
        want = ((s.get("resources") or {}).get("limits") or {}).get(GPU) or \
               ((s.get("resources") or {}).get("requests") or {}).get(GPU) or "1"
        res.setdefault("limits", {})[GPU] = str(want)
    return {k: v for k, v in res.items() if v}


def _volumes(si: dict) -> tuple[list[dict], list[dict]]:
    s = spec_of(si)
    vols = [{"name": "data", "persistentVolumeClaim": {"claimName": names(si)["pvc"]}}]
    mounts = [{"name": "data", "mountPath": "/data"}]
    return vols, mounts


def _security(pod: bool) -> dict:
    return {"fsGroup": RUN_AS, "runAsUser": RUN_AS, "runAsGroup": RUN_AS, "runAsNonRoot": True} if pod else \
           {"allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}}


def source_path(si: dict) -> str:
    src = spec_of(si)["source"]
    return f"/source/{src['path'].lstrip('/')}" if src.get("pvc") else src["path"]


def render_job(si: dict, digest: str) -> dict:
    """The indexer: ``python -m serving build``. Output is the segment on the shared volume; the result (segment id and
    version) is the container's termination message, which the operator reads from the pod's status."""
    s = spec_of(si)
    name = f"{names(si)['job_prefix']}{digest}"
    cm = s["mapping"]["configMapRef"]
    vols, mounts = _volumes(si)
    vols.append({"name": "mapping", "configMap": {"name": cm["name"]}})
    mounts.append({"name": "mapping", "mountPath": "/etc/s3/mapping", "readOnly": True})
    if (s["source"] or {}).get("pvc"):
        vols.append({"name": "source", "persistentVolumeClaim": {"claimName": s["source"]["pvc"], "readOnly": True}})
        mounts.append({"name": "source", "mountPath": "/source", "readOnly": True})
    container = {
        "name": "indexer", "image": s["image"], "imagePullPolicy": s.get("imagePullPolicy", "IfNotPresent"),
        "command": ["python", "-m", "serving", "build", "--name", si["metadata"]["name"],
                    "--mapping", f"/etc/s3/mapping/{cm.get('key', 'mapping.json')}",
                    "--source", source_path(si), "--data-dir", "/data"],
        "env": [{"name": "S3_SEARCH_DEVICE", "value": "cpu"}],
        "resources": _resources(s, gpu=False), "volumeMounts": mounts,
        "terminationMessagePolicy": "FallbackToLogsOnError", "securityContext": _security(False)}
    return {"apiVersion": "batch/v1", "kind": "Job",
            "metadata": _meta(si, name, "indexer", {"search.s3.dev/build-hash": digest}),
            "spec": {"backoffLimit": 2, "template": {
                "metadata": {"labels": labels(si, "indexer")},
                "spec": {"restartPolicy": "Never", "securityContext": _security(True), "containers": [container],
                         "volumes": vols}}}}


def render_deployment(si: dict, segment_id: str) -> dict:
    s = spec_of(si)
    n = si["metadata"]["name"]
    gpu = s["device"] == "cuda"
    vols, mounts = _volumes(si)
    serving = s.get("serving") or {}
    env = [{"name": "S3_SEARCH_DEVICE", "value": s["device"]},
           {"name": "S3_SERVING_DATA_DIR", "value": "/data"},
           {"name": "S3_SERVING_PRELOAD", "value": f"{n}:{segment_id}"},
           {"name": "S3_SERVING_PORT", "value": str(HTTP_PORT)},
           {"name": "S3_SERVING_DRAIN_SECONDS", "value": str(DRAIN_SECONDS)}]
    for key, var in (("maxBatch", "S3_SERVING_MAX_BATCH"), ("maxWaitMs", "S3_SERVING_MAX_WAIT_MS"),
                     ("cacheSize", "S3_SERVING_CACHE")):
        if key in serving:
            env.append({"name": var, "value": str(serving[key])})
    http = {"port": "http"}
    container = {
        "name": "replica", "image": s["image"], "imagePullPolicy": s.get("imagePullPolicy", "IfNotPresent"),
        "command": ["python", "-m", "serving"],
        "ports": [{"name": "http", "containerPort": HTTP_PORT}], "env": env,
        "resources": _resources(s, gpu), "volumeMounts": mounts, "securityContext": _security(False),
        "startupProbe": {"httpGet": {**http, "path": "/healthz"}, "periodSeconds": 2, "failureThreshold": 150},
        "readinessProbe": {"httpGet": {**http, "path": "/readyz"}, "periodSeconds": 2, "failureThreshold": 2,
                           "successThreshold": 1, "timeoutSeconds": 2},
        "livenessProbe": {"httpGet": {**http, "path": "/healthz"}, "periodSeconds": 10, "failureThreshold": 6}}
    pod = {"securityContext": _security(True), "containers": [container], "volumes": vols,
           "terminationGracePeriodSeconds": DRAIN_SECONDS + 25}
    if gpu:
        pod["nodeSelector"] = {"nvidia.com/gpu.present": "true"}
        pod["tolerations"] = [{"key": GPU, "operator": "Exists", "effect": "NoSchedule"}]
    return {"apiVersion": "apps/v1", "kind": "Deployment",
            "metadata": _meta(si, names(si)["deployment"], "replica"),
            "spec": {"replicas": s["replicas"], "minReadySeconds": 2, "revisionHistoryLimit": 2,
                     "selector": {"matchLabels": selector(si)},
                     "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxUnavailable": 0, "maxSurge": 1}},
                     "template": {"metadata": {"labels": labels(si, "replica"),
                                               "annotations": {ANNOTATION_SEGMENT: segment_id}},
                                  "spec": pod}}}


def render_service(si: dict) -> dict:
    return {"apiVersion": "v1", "kind": "Service", "metadata": _meta(si, names(si)["service"], "replica"),
            "spec": {"type": "ClusterIP", "selector": selector(si),
                     "ports": [{"name": "http", "port": 80, "targetPort": "http"}]}}

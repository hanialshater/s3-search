"""The operator against a real kube-apiserver (envtest). Run with deploy/test-envtest.sh.

No controller-manager and no kubelet run, so a ``Sim`` plays them: it completes Jobs (with the termination message a
real indexer writes), and moves Deployment status through a rolling update the way the Deployment controller would
with ``maxUnavailable: 0, maxSurge: 1``. What these tests check is the operator's side of every exchange, against the
real API server's validation, defaulting, server-side apply, ownership and RBAC.
"""
import json
import time
from datetime import datetime, timezone

import pytest

from deploy.s3operator import render
from deploy.s3operator.kube import (CONFIGMAP, DEPLOYMENT, JOB, POD, PVC, SEARCHINDEX, SERVICE, ApiError)

pytestmark = pytest.mark.envtest

MAPPING = {"properties": {"product_title": {"type": "text"}}}
SEG_A, SEG_B, SEG_C = "seg-60-aaaaaaaa", "seg-75-bbbbbbbb", "seg-90-cccccccc"


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Sim:
    """The parts of Kubernetes that do not run here."""

    def __init__(self, kube, ns):
        self.kube, self.ns = kube, ns

    # -- the object under test --------------------------------------------------------------------------------------
    def mapping(self, name="m", data=None):
        self.kube.apply(CONFIGMAP, {"apiVersion": "v1", "kind": "ConfigMap",
                                    "metadata": {"name": name, "namespace": self.ns},
                                    "data": {"mapping.json": json.dumps(data or MAPPING)}})

    def create(self, name="toy", **spec):
        body = {"apiVersion": render.API_VERSION, "kind": "SearchIndex",
                "metadata": {"name": name, "namespace": self.ns},
                "spec": {"mapping": {"configMapRef": {"name": "m"}},
                         "source": {"path": "/app/deploy/samples/toy-products.jsonl"}, "replicas": 2, **spec}}
        return self.kube.create(SEARCHINDEX, body)

    def get(self, name="toy"):
        return self.kube.get(SEARCHINDEX, self.ns, name)

    def update(self, name="toy", **spec):
        """Like ``kubectl patch --type merge`` on the spec."""
        return self.kube.request("PATCH", f"{SEARCHINDEX.collection(self.ns)}/{name}", {"spec": spec},
                                 "application/merge-patch+json")

    # -- Job controller and kubelet ---------------------------------------------------------------------------------
    def jobs(self, name="toy"):
        return self.kube.list(JOB, self.ns, f"{render.LABEL_INDEX}={name}")

    def finish_job(self, job_name, segment_id=None, version=None, fail=None):
        job = self.kube.get(JOB, self.ns, job_name)
        pod_name = job_name + "-x1"
        self.kube.create(POD, {"apiVersion": "v1", "kind": "Pod", "metadata": {
            "name": pod_name, "namespace": self.ns, "labels": {"job-name": job_name}},
            "spec": {"restartPolicy": "Never", "containers": [{"name": "indexer", "image": "x"}]}})
        message = fail or json.dumps({"segment_id": segment_id, "version": version, "n_docs": 60})
        self.kube.request("PATCH", f"{POD.collection(self.ns)}/{pod_name}/status", {"status": {
            "phase": "Failed" if fail else "Succeeded",
            "containerStatuses": [{"name": "indexer", "image": "x", "imageID": "", "ready": False,
                                   "state": {"terminated": {"exitCode": 1 if fail else 0, "message": message,
                                                            "startedAt": now(), "finishedAt": now()}}}]}},
            "application/merge-patch+json")
        t = now()
        if fail:
            status = {"failed": 3, "startTime": t, "conditions": [
                {"type": "FailureTarget", "status": "True", "reason": "BackoffLimitExceeded", "lastTransitionTime": t},
                {"type": "Failed", "status": "True", "reason": "BackoffLimitExceeded", "lastTransitionTime": t}]}
        else:
            status = {"succeeded": 1, "startTime": t, "completionTime": t, "conditions": [
                {"type": "SuccessCriteriaMet", "status": "True", "lastTransitionTime": t},
                {"type": "Complete", "status": "True", "lastTransitionTime": t}]}
        self.kube.request("PATCH", f"{JOB.collection(self.ns)}/{job['metadata']['name']}/status", {"status": status},
                          "application/merge-patch+json")

    # -- Deployment controller and kubelet --------------------------------------------------------------------------
    def deployment(self, name="toy"):
        return self.kube.get(DEPLOYMENT, self.ns, f"{name}-replicas")

    def set_rollout(self, name="toy", *, replicas, updated, ready):
        d = self.deployment(name)
        self.kube.request("PATCH", f"{DEPLOYMENT.collection(self.ns)}/{name}-replicas/status", {"status": {
            "observedGeneration": d["metadata"]["generation"], "replicas": replicas, "updatedReplicas": updated,
            "readyReplicas": ready, "availableReplicas": ready}}, "application/merge-patch+json")

    def settle(self, name="toy", n=None):
        n = n if n is not None else self.get(name)["spec"]["replicas"]
        self.set_rollout(name, replicas=n, updated=n, ready=n)


@pytest.fixture
def sim(kube, ns):
    return Sim(kube, ns)


def phase(rec, sim, name="toy"):
    rec.reconcile(sim.get(name))
    return sim.get(name)["status"]


def cond(status, type_):
    return next(c for c in status["conditions"] if c["type"] == type_)


def containers(obj):
    return obj["spec"]["template"]["spec"]["containers"]


# ---- the CRD ----------------------------------------------------------------------------------------------------
def test_crd_defaults_and_validation(sim, kube):
    si = sim.create(replicas=3)
    assert si["spec"]["device"] == "cpu" and si["spec"]["storage"] == {"size": "5Gi", "accessModes": ["ReadWriteOnce"]}
    assert si["spec"]["mapping"]["configMapRef"]["name"] == "m"
    for bad, field in (({"device": "tpu"}, "device"), ({"segmentVersion": "latest"}, "segmentVersion"),
                       ({"replicas": -1}, "replicas")):
        with pytest.raises(ApiError) as e:
            sim.create(name="bad", **bad)
        assert e.value.status == 422 and field in e.value.message
    out = kube.request("GET", f"{SEARCHINDEX.collection(sim.ns)}/toy/scale")   # the scale subresource is served
    assert out["spec"]["replicas"] == 3


# ---- bring-up ---------------------------------------------------------------------------------------------------
def test_bring_up_from_one_resource(sim, rec, kube):
    sim.create()
    st = phase(rec, sim)                                                    # no mapping yet
    assert st["phase"] == "Pending" and cond(st, "MappingAvailable")["status"] == "False"
    assert kube.get(SERVICE, sim.ns, "toy") is not None and kube.get(PVC, sim.ns, "toy-data") is not None
    assert not sim.jobs()

    sim.mapping()
    st = phase(rec, sim)
    assert st["phase"] == "Indexing" and cond(st, "Valid")["status"] == "True"
    (job,) = sim.jobs()
    assert sim.deployment() is None                                         # replicas wait for a segment
    c = job["spec"]["template"]["spec"]["containers"][0]
    assert c["command"][:4] == ["python", "-m", "serving", "build"]
    assert job["metadata"]["ownerReferences"][0]["kind"] == "SearchIndex"

    sim.finish_job(job["metadata"]["name"], SEG_A, 60)
    st = phase(rec, sim)
    assert st["indexJob"]["segmentId"] == SEG_A and st["latestSegmentId"] == SEG_A
    assert st["phase"] == "Pending" and "0/2" in st["message"]             # Deployment exists, no replica ready yet
    dep = sim.deployment()
    assert dep["spec"]["replicas"] == 2 and dep["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    env = {e["name"]: e["value"] for e in containers(dep)[0]["env"]}
    assert env["S3_SERVING_PRELOAD"] == f"toy:{SEG_A}" and env["S3_SEARCH_DEVICE"] == "cpu"
    assert dep["metadata"]["ownerReferences"][0]["uid"] == sim.get()["metadata"]["uid"]

    sim.set_rollout(replicas=2, updated=2, ready=1)
    assert phase(rec, sim)["phase"] == "Pending"
    sim.settle()
    st = phase(rec, sim)
    assert st["phase"] == "Ready" and st["segmentId"] == SEG_A and st["version"] == 60 and st["readyReplicas"] == 2
    assert cond(st, "Ready")["status"] == "True" and cond(st, "IndexBuilt")["status"] == "True"
    assert cond(st, "Available")["status"] == "True"
    assert [s["segmentId"] for s in st["segments"]] == [SEG_A]
    before = sim.get()["metadata"]["resourceVersion"]
    rec.reconcile(sim.get())                                                # steady state writes nothing
    assert sim.get()["metadata"]["resourceVersion"] == before


# ---- the GPU is never assumed -----------------------------------------------------------------------------------
def build_ready(sim, rec, name="toy", seg=SEG_A, ver=60):
    sim.mapping()
    phase(rec, sim, name)
    (job,) = sim.jobs(name)
    sim.finish_job(job["metadata"]["name"], seg, ver)
    phase(rec, sim, name)
    sim.settle(name)
    return phase(rec, sim, name)


def test_cpu_index_has_no_gpu_fields(sim, rec, kube):
    sim.create(resources={"requests": {"cpu": "100m", "memory": "128Mi"}, "limits": {"memory": "512Mi"}})
    build_ready(sim, rec)
    for obj in (sim.deployment(), sim.jobs()[0]):
        text = json.dumps(obj["spec"])
        assert "nvidia" not in text and "nodeSelector" not in text and "tolerations" not in text
    assert containers(sim.deployment())[0]["resources"]["limits"] == {"memory": "512Mi"}


def test_cuda_index_gets_gpu_limits_selector_and_toleration(sim, rec):
    sim.create(device="cuda", image="s3-search:cuda",
               resources={"requests": {"cpu": "2"}, "limits": {"memory": "8Gi", "nvidia.com/gpu": 1}})
    build_ready(sim, rec)
    dep = sim.deployment()
    pod = dep["spec"]["template"]["spec"]
    assert containers(dep)[0]["resources"]["limits"] == {"memory": "8Gi", "nvidia.com/gpu": "1"}
    assert pod["nodeSelector"] == {"nvidia.com/gpu.present": "true"}
    assert {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"} in pod["tolerations"]
    assert containers(dep)[0]["image"] == "s3-search:cuda"
    assert "nvidia" not in json.dumps(sim.jobs()[0]["spec"])                 # the indexer never takes a GPU


def test_gpu_resources_without_cuda_are_refused_and_nothing_is_created(sim, rec, kube):
    sim.create(resources={"limits": {"nvidia.com/gpu": 1}})
    sim.mapping()
    st = phase(rec, sim)
    assert st["phase"] == "Failed" and cond(st, "Valid")["status"] == "False"
    assert "never assumes" in st["message"]
    assert not sim.jobs() and sim.deployment() is None and kube.get(SERVICE, sim.ns, "toy") is None


# ---- rolling segment swaps and roll back ------------------------------------------------------------------------
def test_new_segment_rolls_without_dropping_the_old_one(sim, rec):
    sim.create()
    assert build_ready(sim, rec)["segmentId"] == SEG_A
    template_a = sim.deployment()["spec"]["template"]

    sim.update(reindexToken="2")                                            # reindex: a new build
    st = phase(rec, sim)
    assert st["phase"] == "Indexing" and st["segmentId"] == SEG_A            # still serving A
    assert sim.deployment()["spec"]["template"] == template_a                # replicas untouched while building
    assert len(sim.jobs()) == 2                                             # the old Job stays until the new one is done
    cur = st["indexJob"]["name"]

    sim.finish_job(cur, SEG_B, 75)
    st = phase(rec, sim)
    dep = sim.deployment()
    assert dep["spec"]["template"]["metadata"]["annotations"][render.ANNOTATION_SEGMENT] == SEG_B
    assert dep["spec"]["strategy"]["rollingUpdate"] == {"maxUnavailable": 0, "maxSurge": 1}
    assert st["phase"] == "Swapping" and st["segmentId"] == SEG_A and "seg-60" in st["message"]

    sim.set_rollout(replicas=3, updated=1, ready=2)                          # surge pod up, not ready yet
    assert phase(rec, sim)["phase"] == "Swapping"
    sim.set_rollout(replicas=3, updated=2, ready=3)                          # one old pod still being removed
    assert phase(rec, sim)["phase"] == "Swapping"
    sim.settle()
    st = phase(rec, sim)
    assert st["phase"] == "Ready" and st["segmentId"] == SEG_B and st["version"] == 75
    assert [s["segmentId"] for s in st["segments"]] == [SEG_A, SEG_B]
    assert [j["metadata"]["name"] for j in sim.jobs()] == [cur]              # the superseded Job is gone


def test_mapping_change_triggers_a_rebuild(sim, rec):
    sim.create()
    build_ready(sim, rec)
    first = sim.jobs()[0]["metadata"]["name"]
    sim.mapping(data={"properties": {"product_title": {"type": "text"}, "price": {"type": "float"}}})
    st = phase(rec, sim)
    assert st["phase"] == "Indexing" and st["indexJob"]["name"] != first


def test_pinning_a_segment_rolls_back_and_unpinning_returns(sim, rec):
    sim.create()
    build_ready(sim, rec)
    sim.update(reindexToken="2")
    phase(rec, sim)
    cur = phase(rec, sim)["indexJob"]["name"]
    sim.finish_job(cur, SEG_B, 75)
    phase(rec, sim)
    sim.settle()
    assert phase(rec, sim)["segmentId"] == SEG_B

    sim.update(segmentVersion=SEG_A)                                        # roll back
    st = phase(rec, sim)
    ann = sim.deployment()["spec"]["template"]["metadata"]["annotations"]
    assert ann[render.ANNOTATION_SEGMENT] == SEG_A and st["phase"] == "Swapping"
    sim.settle()
    st = phase(rec, sim)
    assert st["phase"] == "Ready" and st["segmentId"] == SEG_A and st["latestSegmentId"] == SEG_B

    sim.update(segmentVersion="seg-1-deadbeef")                             # not a segment this resource built
    st = phase(rec, sim)
    assert sim.deployment()["spec"]["template"]["metadata"]["annotations"][render.ANNOTATION_SEGMENT] == SEG_A
    assert "not a segment this resource built" in st["message"]

    sim.update(segmentVersion=None)                                         # back to the latest
    st = phase(rec, sim)
    assert sim.deployment()["spec"]["template"]["metadata"]["annotations"][render.ANNOTATION_SEGMENT] == SEG_B
    assert st["phase"] == "Swapping"


def test_scaling_does_not_rebuild(sim, rec):
    sim.create()
    build_ready(sim, rec)
    job = sim.jobs()[0]["metadata"]["name"]
    sim.update(replicas=4)
    st = phase(rec, sim)
    assert sim.deployment()["spec"]["replicas"] == 4 and sim.jobs()[0]["metadata"]["name"] == job
    assert st["phase"] == "Pending" and st["segmentId"] == SEG_A             # not a swap: same segment, more replicas
    sim.settle(n=4)
    assert phase(rec, sim)["phase"] == "Ready"


def test_failed_build_keeps_serving_and_is_reported(sim, rec):
    sim.create()
    build_ready(sim, rec)
    sim.update(reindexToken="bad")
    cur = phase(rec, sim)["indexJob"]["name"]
    sim.finish_job(cur, fail="Traceback...\nengine.api.InvalidMapping: properties.price: unknown type 'flot'")
    st = phase(rec, sim)
    assert st["phase"] == "Failed" and "InvalidMapping" in st["message"] and cond(st, "IndexBuilt")["status"] == "False"
    assert st["segmentId"] == SEG_A and sim.deployment()["spec"]["template"]["metadata"]["annotations"][
        render.ANNOTATION_SEGMENT] == SEG_A
    assert cond(st, "Ready")["status"] == "False"


def test_missing_result_is_a_failure_not_a_guess(sim, rec):
    sim.create()
    sim.mapping()
    cur = phase(rec, sim)["indexJob"]["name"]
    sim.finish_job(cur, segment_id="not-a-segment", version=1)
    st = phase(rec, sim)
    assert st["phase"] == "Failed" and "cannot be read" in st["message"] and sim.deployment() is None


# ---- the running operator and its RBAC --------------------------------------------------------------------------
def test_operator_loop_once_reconciles_every_resource(sim, apiserver, kube):
    import subprocess
    import sys
    sim.mapping()
    sim.create("a")
    sim.create("b")
    out = subprocess.run([sys.executable, "-m", "deploy.s3operator", "--once", "--server", apiserver.url,
                          "--token", apiserver.token, "--insecure", "--namespace", sim.ns],
                         capture_output=True, text=True, timeout=60, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    assert out.returncode == 0, out.stderr
    assert sim.get("a")["status"]["phase"] == "Indexing" and sim.get("b")["status"]["phase"] == "Indexing"


def test_rbac_in_the_manifest_is_enough(sim, apiserver, ns):
    """Apply the operator's ClusterRole and binding, take its service account's token, run the scenario with it."""
    yaml = pytest.importorskip("yaml")
    from deploy.s3operator.kube import Kube, Res
    docs = list(yaml.safe_load_all((render_path() / "manifests" / "operator.yaml").read_text()))
    k = apiserver.kube
    for d in docs:
        if d["kind"] == "Namespace":
            try:
                k.create(Res("v1", "namespaces", "Namespace"), d)
            except ApiError:
                pass
        elif d["kind"] == "ServiceAccount":
            try:
                k.create(Res("v1", "serviceaccounts", "ServiceAccount"), d)
            except ApiError:
                pass
        elif d["kind"] in ("ClusterRole", "ClusterRoleBinding"):
            res = Res("rbac.authorization.k8s.io/v1", d["kind"].lower() + "s", d["kind"])
            k.request("PATCH", f"{res.collection(None)}/{d['metadata']['name']}", d, "application/apply-patch+yaml",
                      {"fieldManager": "test", "force": "true"})
    tok = k.request("POST", "/api/v1/namespaces/s3-search-system/serviceaccounts/s3-operator/token",
                    {"apiVersion": "authentication.k8s.io/v1", "kind": "TokenRequest", "spec": {"expirationSeconds": 3600}})
    limited = Kube(apiserver.url, tok["status"]["token"], insecure=True)
    with pytest.raises(ApiError) as e:                                       # the role is narrow: no secrets, no nodes
        limited.request("GET", "/api/v1/secrets")
    assert e.value.status == 403
    from deploy.s3operator.reconcile import Reconciler
    from deploy.s3operator.leader import LeaderElector
    el = LeaderElector(limited, "s3-search-system", "rbac-check", "me")
    assert el.try_acquire_or_renew() and el.try_acquire_or_renew()          # the manifest's leases rule: create, get, update
    with pytest.raises(ApiError):                                            # and it can watch what it reconciles
        limited.request("GET", "/api/v1/nodes")
    assert list(limited.watch(SEARCHINDEX, sim.ns, "0", timeout_seconds=1)) is not None
    sim.mapping()
    sim.create()
    r = Reconciler(limited)
    r.reconcile(sim.get())
    (job,) = sim.jobs()
    sim.finish_job(job["metadata"]["name"], SEG_A, 60)
    r.reconcile(sim.get())
    sim.settle()
    r.reconcile(sim.get())
    sim.update(reindexToken="2")
    r.reconcile(sim.get())
    cur = sim.get()["status"]["indexJob"]["name"]
    sim.finish_job(cur, SEG_B, 75)
    r.reconcile(sim.get())                                                   # deletes the old Job: needs `delete jobs`
    st = sim.get()["status"]
    assert st["phase"] == "Swapping" and len(sim.jobs()) == 1


def render_path():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]

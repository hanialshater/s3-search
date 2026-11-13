"""Rendering and validation, no cluster. Under a second."""
import copy
from pathlib import Path

import pytest

from deploy.s3operator import render
from deploy.s3operator.reconcile import Conditions, job_state, merge_patch, read_result, segment_version

HERE = Path(__file__).resolve().parents[1]


def si(**spec):
    base = {"mapping": {"configMapRef": {"name": "m", "key": "mapping.json"}},
            "source": {"path": "/app/deploy/samples/toy-products.jsonl"}, "replicas": 2}
    base.update(spec)
    return {"metadata": {"name": "toy", "namespace": "default", "uid": "u-1", "generation": 1}, "spec": base}


def pod_spec(dep):
    return dep["spec"]["template"]["spec"]


def test_cpu_has_no_gpu_anywhere():
    for device in ("cpu", "torch-cpu"):
        dep = render.render_deployment(si(device=device), "seg-1-aaaaaaaa")
        text = repr(dep) + repr(render.render_job(si(device=device), "h"))
        assert "nvidia" not in text
        assert "nodeSelector" not in pod_spec(dep) and "tolerations" not in pod_spec(dep)


def test_cuda_adds_gpu_fields_to_replicas_only():
    s = si(device="cuda", resources={"requests": {"cpu": "2"}, "limits": {"memory": "8Gi", "nvidia.com/gpu": 2}})
    dep = render.render_deployment(s, "seg-1-aaaaaaaa")
    c = pod_spec(dep)["containers"][0]
    assert c["resources"]["limits"] == {"memory": "8Gi", "nvidia.com/gpu": "2"}
    assert pod_spec(dep)["nodeSelector"] == {"nvidia.com/gpu.present": "true"}
    assert pod_spec(dep)["tolerations"] == [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}]
    assert {"name": "S3_SEARCH_DEVICE", "value": "cuda"} in c["env"]
    job = render.render_job(s, "h")["spec"]["template"]["spec"]            # the indexer builds on the host
    assert "nvidia" not in repr(job) and "nodeSelector" not in job
    # no GPU quantity given: one
    d1 = render.render_deployment(si(device="cuda"), "seg-1-aaaaaaaa")
    assert pod_spec(d1)["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"


def test_gpu_resources_on_a_cpu_index_are_refused():
    errs = render.validate(si(device="cpu", resources={"limits": {"nvidia.com/gpu": 1}}))
    assert len(errs) == 1 and "never assumes" in errs[0]
    assert render.validate(si(device="cuda", resources={"limits": {"nvidia.com/gpu": 1}})) == []


@pytest.mark.parametrize("spec,needle", [
    ({"device": "tpu"}, "spec.device"), ({"replicas": -1}, "replicas"), ({"segmentVersion": "latest"}, "segmentVersion"),
    ({"source": {}}, "source.path"), ({"mapping": {}}, "mapping")])
def test_validation(spec, needle):
    assert any(needle in e for e in render.validate(si(**spec)))


def test_rolling_update_is_zero_unavailable_and_carries_the_segment():
    dep = render.render_deployment(si(), "seg-1044-3fa2c9d1")
    assert dep["spec"]["strategy"]["rollingUpdate"] == {"maxUnavailable": 0, "maxSurge": 1}
    ann = dep["spec"]["template"]["metadata"]["annotations"]
    assert ann[render.ANNOTATION_SEGMENT] == "seg-1044-3fa2c9d1"
    env = {e["name"]: e["value"] for e in pod_spec(dep)["containers"][0]["env"]}
    assert env["S3_SERVING_PRELOAD"] == "toy:seg-1044-3fa2c9d1"
    other = render.render_deployment(si(), "seg-1045-00000000")
    assert dep["spec"]["template"] != other["spec"]["template"]            # a new segment changes the pod template
    probes = pod_spec(dep)["containers"][0]
    assert probes["readinessProbe"]["httpGet"]["path"] == "/readyz" and probes["startupProbe"]


def test_children_are_owned_and_labelled():
    s = si()
    for obj in (render.render_deployment(s, "seg-1-aaaaaaaa"), render.render_service(s), render.render_pvc(s),
                render.render_job(s, "h")):
        ref = obj["metadata"]["ownerReferences"][0]
        assert ref["uid"] == "u-1" and ref["kind"] == "SearchIndex" and ref["controller"]
        assert obj["metadata"]["labels"][render.LABEL_INDEX] == "toy"
    assert render.render_service(s)["spec"]["selector"] == render.selector(s)


def test_existing_claim_means_no_pvc_and_source_pvc_mounts():
    assert render.render_pvc(si(storage={"existingClaim": "mine"})) is None
    assert render.names(si(storage={"existingClaim": "mine"}))["pvc"] == "mine"
    job = render.render_job(si(source={"pvc": "esci", "path": "a/b.parquet"}), "h")
    c = job["spec"]["template"]["spec"]["containers"][0]
    assert c["command"][c["command"].index("--source") + 1] == "/source/a/b.parquet"
    assert {"name": "source", "mountPath": "/source", "readOnly": True} in c["volumeMounts"]


def test_build_hash_ignores_replicas_and_pin_but_not_inputs():
    base = si()
    h = render.build_hash(base, {"mapping.json": "{}"})
    assert render.build_hash(si(replicas=9, segmentVersion="seg-1-aaaaaaaa"), {"mapping.json": "{}"}) == h
    assert render.build_hash(si(reindexToken="2"), {"mapping.json": "{}"}) != h
    assert render.build_hash(base, {"mapping.json": '{"a": 1}'}) != h
    assert render.build_hash(si(image="x:2"), {"mapping.json": "{}"}) != h
    assert render.build_hash(si(source={"path": "/other"}), {"mapping.json": "{}"}) != h


def test_job_runs_the_serving_build_command():
    job = render.render_job(si(), "abcd1234")
    assert job["metadata"]["name"] == "toy-index-abcd1234"
    c = job["spec"]["template"]["spec"]["containers"][0]
    assert c["command"][:4] == ["python", "-m", "serving", "build"] and "/etc/s3/mapping/mapping.json" in c["command"]
    assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"


def test_default_image_by_device():
    assert render.default_image("cpu", {}) == "s3-search:cpu" and render.default_image("cuda", {}) == "s3-search:cuda"
    assert render.spec_of({"metadata": {}, "spec": {"device": "cuda"}})["image"] == "s3-search:cuda"


def test_reconcile_helpers():
    assert segment_version("seg-1044-3fa2c9d1") == 1044 and segment_version(None) is None
    assert job_state({"status": {"conditions": [{"type": "Complete", "status": "True"}]}}) == "Succeeded"
    assert job_state({"status": {"conditions": [{"type": "Failed", "status": "True"}]}}) == "Failed"
    assert job_state({"status": {"active": 1}}) == "Active"
    pod = {"metadata": {}, "status": {"containerStatuses": [{"name": "indexer", "state": {"terminated": {
        "exitCode": 0, "message": '{"segment_id": "seg-5-0badc0de", "version": 5}'}}}]}}
    assert read_result([pod])["version"] == 5
    bad = copy.deepcopy(pod)
    bad["status"]["containerStatuses"][0]["state"]["terminated"]["message"] = "not json"
    assert read_result([bad]) is None
    c = Conditions([{"type": "Ready", "status": "False", "lastTransitionTime": "t0"}], 1, "t1")
    c.set("Ready", False, "x")
    assert c.list()[0]["lastTransitionTime"] == "t0"                      # unchanged status keeps its time
    c2 = Conditions([{"type": "Ready", "status": "False", "lastTransitionTime": "t0"}], 1, "t1")
    c2.set("Ready", True, "y")
    assert c2.list()[0]["lastTransitionTime"] == "t1"


def test_crd_and_manifests_parse_and_agree_with_the_code():
    yaml = pytest.importorskip("yaml")   # PyYAML is not in requirements.txt; the test skips without it
    crd = yaml.safe_load((HERE / "crd" / "searchindex.yaml").read_text())
    assert crd["spec"]["group"] == render.GROUP and crd["spec"]["versions"][0]["name"] == render.VERSION
    spec = crd["spec"]["versions"][0]["schema"]["openAPIV3Schema"]["properties"]["spec"]["properties"]
    assert spec["device"]["enum"] == list(render.DEVICES)
    docs = list(yaml.safe_load_all((HERE / "manifests" / "operator.yaml").read_text()))
    assert {d["kind"] for d in docs} == {"Namespace", "ServiceAccount", "ClusterRole", "ClusterRoleBinding", "Deployment"}
    for f in ("toy-searchindex.yaml", "gpu-searchindex.yaml"):
        s = yaml.safe_load((HERE / "samples" / f).read_text())
        assert render.validate(s | {"metadata": {"name": "x", "namespace": "d", "uid": "u"}}) == []


def test_merge_patch_nulls_what_vanished():
    prev = {"phase": "Ready", "indexJob": {"hash": "a", "segmentId": "s", "state": "Succeeded"}, "gone": 1}
    new = {"phase": "Indexing", "indexJob": {"hash": "b", "state": "Active"}}
    assert merge_patch(prev, new) == {"phase": "Indexing", "gone": None,
                                      "indexJob": {"hash": "b", "state": "Active", "segmentId": None}}

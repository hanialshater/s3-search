"""The envtest harness: a real kube-apiserver and etcd, started from binaries, and helpers that play the parts of the
controllers that are not running (the Job controller, the Deployment controller and the kubelet), because no pod runs
in this environment.

Tests marked ``envtest`` run only when ``S3_ENVTEST=1`` (``deploy/test-envtest.sh`` sets it), so they stay outside
``python -m referee fast`` and the unit tests. ``S3_ENVTEST_BIN`` is the directory with ``etcd`` and ``kube-apiserver``
(default /tmp/claude-0/envtest/controller-tools/envtest).
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from deploy.s3operator.kube import ApiError, Kube, Res
from deploy.s3operator.reconcile import Reconciler

DEPLOY = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("S3_ENVTEST_BIN", "/tmp/claude-0/envtest/controller-tools/envtest"))
NAMESPACE = Res("v1", "namespaces", "Namespace")
CRD = Res("apiextensions.k8s.io/v1", "customresourcedefinitions", "CustomResourceDefinition")


def pytest_configure(config):
    config.addinivalue_line("markers", "envtest: needs a real kube-apiserver (S3_ENVTEST=1, deploy/test-envtest.sh)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("S3_ENVTEST") == "1":
        return
    skip = pytest.mark.skip(reason="envtest: run deploy/test-envtest.sh (needs kube-apiserver and etcd binaries)")
    for item in items:
        if "envtest" in item.keywords:
            item.add_marker(skip)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ApiServer:
    def __init__(self, workdir: Path):
        self.dir = workdir
        self.token = "envtest-admin-token"
        self.port = _free_port()
        etcd_port, peer = _free_port(), _free_port()
        (workdir / "tokens.csv").write_text(f"{self.token},admin,uid-admin,system:masters\n")
        subprocess.run(["openssl", "genrsa", "-out", str(workdir / "sa.key"), "2048"], check=True,
                       capture_output=True)
        self.procs = [
            subprocess.Popen([str(BIN / "etcd"), "--data-dir", str(workdir / "etcd"),
                              "--listen-client-urls", f"http://127.0.0.1:{etcd_port}",
                              "--advertise-client-urls", f"http://127.0.0.1:{etcd_port}",
                              "--listen-peer-urls", f"http://127.0.0.1:{peer}"],
                             stdout=open(workdir / "etcd.log", "w"), stderr=subprocess.STDOUT)]
        time.sleep(1.5)
        self.procs.append(subprocess.Popen([
            str(BIN / "kube-apiserver"), f"--etcd-servers=http://127.0.0.1:{etcd_port}",
            f"--secure-port={self.port}", "--bind-address=127.0.0.1", f"--cert-dir={workdir / 'certs'}",
            f"--token-auth-file={workdir / 'tokens.csv'}", "--authorization-mode=RBAC",
            "--service-account-issuer=https://localhost", f"--service-account-key-file={workdir / 'sa.key'}",
            f"--service-account-signing-key-file={workdir / 'sa.key'}", "--service-cluster-ip-range=10.0.0.0/24",
            "--disable-admission-plugins=ServiceAccount"],
            stdout=open(workdir / "apiserver.log", "w"), stderr=subprocess.STDOUT))
        self.url = f"https://127.0.0.1:{self.port}"
        self.kube = Kube(self.url, self.token, insecure=True)
        deadline = time.time() + 60
        while True:
            try:
                self.kube.request("GET", "/version")
                break
            except Exception:  # noqa: BLE001
                if time.time() > deadline or any(p.poll() is not None for p in self.procs):
                    raise RuntimeError(f"kube-apiserver did not start; see {workdir}/apiserver.log")
                time.sleep(0.5)

    def stop(self) -> None:
        for p in reversed(self.procs):
            p.terminate()
        for p in self.procs:
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()


@pytest.fixture(scope="session")
def apiserver():
    if not (BIN / "kube-apiserver").exists():
        pytest.skip(f"no kube-apiserver under {BIN}")
    yaml = pytest.importorskip("yaml")
    work = Path(tempfile.mkdtemp(prefix="s3-envtest-"))
    api = ApiServer(work)
    try:
        crd = yaml.safe_load((DEPLOY / "crd" / "searchindex.yaml").read_text())
        api.kube.create(CRD, crd)
        deadline = time.time() + 30
        while True:  # wait until the CRD is served
            try:
                api.kube.request("GET", "/apis/search.s3.dev/v1alpha1/searchindexes")
                break
            except ApiError:
                if time.time() > deadline:
                    raise
                time.sleep(0.3)
        yield api
    finally:
        api.stop()
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture
def ns(apiserver):
    """A fresh namespace per test."""
    name = "t-" + uuid.uuid4().hex[:8]
    apiserver.kube.create(NAMESPACE, {"apiVersion": "v1", "kind": "Namespace", "metadata": {"name": name}})
    return name


@pytest.fixture
def kube(apiserver) -> Kube:
    return apiserver.kube


@pytest.fixture
def rec(kube) -> Reconciler:
    return Reconciler(kube)


class Operator:
    """An operator process against the envtest API server; its log is a file, so tests can see who reconciled what."""

    def __init__(self, apiserver, workdir, name, *args):
        import subprocess
        import sys
        self.name, self.log = name, Path(workdir) / f"{name}.log"
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "deploy.s3operator", "--server", apiserver.url, "--token", apiserver.token,
             "--insecure", "--identity", name, *args],
            stdout=open(self.log, "w"), stderr=subprocess.STDOUT, cwd=str(DEPLOY.parent))

    def text(self) -> str:
        return self.log.read_text()

    def reconciles(self, ns: str | None = None) -> int:
        return sum(1 for line in self.text().splitlines() if " reconcile " in line and (ns is None or f" {ns}/" in line))

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait(10)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except Exception:  # noqa: BLE001
                self.proc.kill()


@pytest.fixture
def operators(apiserver, tmp_path):
    made: list[Operator] = []

    def start(name, *args):
        op = Operator(apiserver, tmp_path, name, *args)
        made.append(op)
        return op

    yield start
    for op in made:
        op.stop()

"""A minimal Kubernetes REST client on the standard library (urllib + ssl): enough for the operator.

Server-side apply (``application/apply-patch+yaml``, JSON bodies) keeps every write idempotent and leaves fields we do
not own alone. Connection: in-cluster (service account) by default; ``Kube(server=..., token=..., ca=...|insecure=True)``
for a kubeconfig-less dev or envtest API server.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from .render import API_VERSION, MANAGER


@dataclass(frozen=True)
class Res:
    api_version: str   # "v1", "apps/v1", "search.s3.dev/v1alpha1"
    plural: str
    kind: str

    def collection(self, namespace: str | None) -> str:
        base = "/api/v1" if "/" not in self.api_version else f"/apis/{self.api_version}"
        ns = f"/namespaces/{namespace}" if namespace else ""
        return f"{base}{ns}/{self.plural}"


DEPLOYMENT = Res("apps/v1", "deployments", "Deployment")
JOB = Res("batch/v1", "jobs", "Job")
SERVICE = Res("v1", "services", "Service")
PVC = Res("v1", "persistentvolumeclaims", "PersistentVolumeClaim")
CONFIGMAP = Res("v1", "configmaps", "ConfigMap")
POD = Res("v1", "pods", "Pod")
SEARCHINDEX = Res(API_VERSION, "searchindexes", "SearchIndex")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


class Kube:
    SA = "/var/run/secrets/kubernetes.io/serviceaccount"

    def __init__(self, server: str | None = None, token: str | None = None, ca: str | None = None,
                 insecure: bool = False, timeout: float = 15.0):
        if server is None:  # in cluster
            host, port = os.environ["KUBERNETES_SERVICE_HOST"], os.environ.get("KUBERNETES_SERVICE_PORT", "443")
            server = f"https://{host}:{port}" if ":" not in host else f"https://[{host}]:{port}"
            token = open(f"{self.SA}/token").read().strip()
            ca = f"{self.SA}/ca.crt"
        self.server, self.token, self.timeout = server.rstrip("/"), token, timeout
        self._ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context(cafile=ca)

    def request(self, method: str, path: str, body=None, content_type: str = "application/json",
                params: dict | None = None):
        url = self.server + path + (("?" + urllib.parse.urlencode(params)) if params else "")
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Accept", "application/json")
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        if data is not None:
            req.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(req, context=self._ctx, timeout=self.timeout) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            try:
                msg = json.loads(raw).get("message", raw)
            except ValueError:
                msg = raw
            raise ApiError(e.code, msg) from None

    # -- verbs ------------------------------------------------------------------------------------------------------
    def get(self, res: Res, namespace: str, name: str) -> dict | None:
        try:
            return self.request("GET", f"{res.collection(namespace)}/{name}")
        except ApiError as e:
            if e.status == 404:
                return None
            raise

    def list(self, res: Res, namespace: str | None = None, label_selector: str | None = None) -> list[dict]:
        out = self.request("GET", res.collection(namespace),
                           params={"labelSelector": label_selector} if label_selector else None)
        return out.get("items", [])

    def apply(self, res: Res, obj: dict) -> dict:
        """Create or update to exactly ``obj`` for the fields we set (server-side apply, field manager s3-operator)."""
        ns, name = obj["metadata"].get("namespace"), obj["metadata"]["name"]
        return self.request("PATCH", f"{res.collection(ns)}/{name}", obj, "application/apply-patch+yaml",
                            {"fieldManager": MANAGER, "force": "true"})

    def create(self, res: Res, obj: dict) -> dict:
        return self.request("POST", res.collection(obj["metadata"].get("namespace")), obj)

    def patch_status(self, res: Res, namespace: str, name: str, status: dict) -> dict:
        return self.request("PATCH", f"{res.collection(namespace)}/{name}/status", {"status": status},
                            "application/merge-patch+json")

    def delete(self, res: Res, namespace: str, name: str, propagation: str = "Background") -> None:
        try:
            self.request("DELETE", f"{res.collection(namespace)}/{name}", {"propagationPolicy": propagation})
        except ApiError as e:
            if e.status != 404:
                raise

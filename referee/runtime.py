"""What a run ran on: the GPU, its driver and CUDA, torch and the RAPIDS versions (G1: one commit on three GPUs).

Nothing here starts CUDA in this process: the GPU is read with `nvidia-smi` (a subprocess) and the package versions from
their installed metadata, never by importing cupy, cuvs, cuml or rmm (an import would make the engine's next build run
serial, 474f9f5). torch is imported only for `torch.version.cuda`, which does not start CUDA.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

RAPIDS = ("cuml", "cuvs", "libcuvs", "cupy", "rmm", "librmm", "cudf", "pylibraft", "raft-dask", "libraft", "ucx-py")
OTHER = ("torch", "numpy", "pandas", "pyarrow", "xgboost", "scikit-learn", "snowballstemmer")


def _smi(*args) -> str | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def gpus() -> list[dict]:
    out = _smi("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    rows = []
    for line in (out or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            rows.append({"name": parts[0], "driver": parts[1], "memory": parts[2]})
    return rows


def gpu_slug(name: str | None = None) -> str:
    """A short name for filenames: "NVIDIA A100-SXM4-40GB" -> a100, "NVIDIA H100 80GB HBM3" -> h100,
    "NVIDIA RTX PRO 6000 Blackwell Server Edition" -> rtxpro6000, "Tesla T4" -> t4; no GPU -> nogpu."""
    if name is None:
        g = gpus()
        name = g[0]["name"] if g else None
    if not name:
        return "nogpu"
    words = [w for w in re.split(r"[\s_]+", name.lower()) if w and w not in ("nvidia", "tesla", "geforce")]
    slug = []
    for w in words:
        w = w.split("-")[0]
        slug.append(re.sub(r"[^a-z0-9]", "", w))
        if re.search(r"\d", w):
            break
    return "".join(slug) or "gpu"


def _version(dist_prefix: str) -> dict:
    found = {}
    for d in metadata.distributions():
        n = (d.metadata["Name"] or "").lower()
        if n == dist_prefix or n.startswith(dist_prefix + "-cu") or n.startswith(dist_prefix + "-cuda"):
            found[n] = d.version
    return found


def environment() -> dict:
    """The record results.json and results-s9.json carry."""
    g = gpus()
    header = _smi() or ""
    m = re.search(r"CUDA Version:\s*([\d.]+)", header)
    env = {"gpus": g, "gpu": g[0]["name"] if g else None, "gpu_slug": gpu_slug(g[0]["name"] if g else None),
           "driver": g[0]["driver"] if g else None, "cuda_driver_api": m.group(1) if m else None}
    try:
        import torch
        env["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        env["torch_cuda"] = None
    env["rapids"] = {k: v for p in RAPIDS for k, v in _version(p).items()}
    env["packages"] = {k: v for p in OTHER for k, v in _version(p).items()}
    return env


def commit() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def save(local: str, kind: str = "", folder: str = "/content/drive/MyDrive") -> str | None:
    """Copy a results file to s3-search-results[-<kind>]-<commit>-<gpu slug>.json and return that path (None, and say
    so, when the run wrote no file)."""
    src = Path(local)
    if not src.exists():
        print(f"no {local}: the run wrote no results, nothing copied")
        return None
    dst = Path(folder) / f"s3-search-results{'-' + kind if kind else ''}-{commit()}-{gpu_slug()}.json"
    shutil.copyfile(src, dst)
    print(f"saved {local} -> {dst}")
    return str(dst)

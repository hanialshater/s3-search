"""Serving configuration: a dataclass, filled from environment variables and command-line flags.

Environment (all optional):
  S3_SEARCH_DEVICE        cpu | torch-cpu | cuda          (also read by the engine's backend; default cpu)
  S3_SERVING_RPC_PORT     length-prefixed JSON RPC listener (serving/rpc.py); 0 or unset: off
  S3_SERVING_DATA_DIR     where indexes live: <data_dir>/<name>/{mapping.json, store.sqlite, segments/seg-*}
  S3_SERVING_PRELOAD      "name" or "name:segment_id", comma separated: loaded at start (a replica pod sets one)
  S3_SERVING_RANKERS      "name=path,..." ranker JSON files (relative to the data dir) added to every preloaded index
  S3_SERVING_MAX_BATCH    scheduler maximum batch size (default 32)
  S3_SERVING_MAX_WAIT_MS  scheduler maximum wait for a fuller batch (default 2.0)
  S3_SERVING_REFRESH_INTERVAL  default index.refresh_interval (500ms; -1 off)
  S3_SERVING_BULK_CHUNK   actions per store batch in _bulk (default 2000)
  S3_SERVING_CACHE        query cache entries per index, 0 disables (default 1024)
  S3_SERVING_COLUMNS      1 (default): ask the engine for columnar hits and encode them (S6-7); 0: the dict form
  S3_SERVING_WORKERS      HTTP worker processes behind the one port, each with its own copy of the preloaded
                          indexes (default 1; more than 1 serves reads only, see serving/workers.py)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class ServingConfig:
    data_dir: str = "data/serving"
    device: str | None = None
    host: str = "0.0.0.0"
    port: int = 8080
    rpc_port: int = 0                 # 0: no RPC listener
    max_batch: int = 32
    max_wait_ms: float = 2.0
    max_queue: int = 4096
    cache_size: int = 1024
    columns: bool = True              # columnar hits from the engine, encoded by serving/encode.py (S6-7)
    workers: int = 1                  # worker processes (serving/workers.py); more than 1 is read-only
    read_only: bool = False           # set in each worker when workers > 1: writes and admin calls are refused
    worker_index: int | None = None   # this process's number among the workers (None: not a worker)
    refresh_interval: str = "500ms"     # default index.refresh_interval (-1 turns the background refresh off)
    bulk_chunk: int = 2000            # actions applied per store batch while a _bulk body streams in
    request_timeout_s: float = 30.0
    keep_segments: int = 5
    drain_seconds: float = 3.0
    max_body_bytes: int = 64 * 1024 * 1024
    preload: list[tuple[str, str | None]] = field(default_factory=list)
    rankers: dict[str, str] = field(default_factory=dict)

    def resolved_device(self) -> str:
        return self.device or os.environ.get("S3_SEARCH_DEVICE") or "cpu"

    @classmethod
    def from_env(cls, env: dict | None = None) -> "ServingConfig":
        e = os.environ if env is None else env
        cfg = cls()
        cfg.data_dir = e.get("S3_SERVING_DATA_DIR", cfg.data_dir)
        cfg.device = e.get("S3_SEARCH_DEVICE") or None
        cfg.port = int(e.get("S3_SERVING_PORT", cfg.port))
        cfg.rpc_port = int(e.get("S3_SERVING_RPC_PORT", cfg.rpc_port))
        cfg.max_batch = int(e.get("S3_SERVING_MAX_BATCH", cfg.max_batch))
        cfg.max_wait_ms = float(e.get("S3_SERVING_MAX_WAIT_MS", cfg.max_wait_ms))
        cfg.cache_size = int(e.get("S3_SERVING_CACHE", cfg.cache_size))
        cfg.columns = e.get("S3_SERVING_COLUMNS", "1").strip().lower() not in ("0", "false", "no", "off")
        cfg.workers = max(1, int(e.get("S3_SERVING_WORKERS", cfg.workers)))
        cfg.refresh_interval = e.get("S3_SERVING_REFRESH_INTERVAL", cfg.refresh_interval)
        cfg.bulk_chunk = int(e.get("S3_SERVING_BULK_CHUNK", cfg.bulk_chunk))
        cfg.drain_seconds = float(e.get("S3_SERVING_DRAIN_SECONDS", cfg.drain_seconds))
        cfg.preload = parse_preload(e.get("S3_SERVING_PRELOAD", ""))
        cfg.rankers = dict(x.split("=", 1) for x in e.get("S3_SERVING_RANKERS", "").split(",") if "=" in x)
        return cfg


def parse_preload(text: str) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for part in filter(None, (p.strip() for p in text.split(","))):
        name, _, seg = part.partition(":")
        out.append((name, seg or None))
    return out

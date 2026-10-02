#!/usr/bin/env python3
"""Disk cache for CRA-Bench per-session signal records (avoids re-running SBERT/spaCy)."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"


def _bench_fingerprint(bench: Path) -> str:
    st = bench.stat()
    raw = f"{bench.resolve()}:{st.st_size}:{int(st.st_mtime)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_path(bench: Path, s3_mode: str = "keyword") -> Path:
    stem = bench.stem.replace(".", "_")
    return CACHE_DIR / f"signals_{stem}_{s3_mode}_{_bench_fingerprint(bench)}.pkl"


def load_cached(bench: Path, s3_mode: str = "keyword") -> list[dict] | None:
    p = cache_path(bench, s3_mode)
    if not p.is_file():
        return None
    with p.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("fingerprint") != _bench_fingerprint(bench):
        return None
    return payload["records"]


def save_cache(bench: Path, records: list[dict], s3_mode: str = "keyword") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(bench, s3_mode)
    with p.open("wb") as f:
        pickle.dump(
            {"fingerprint": _bench_fingerprint(bench), "s3_mode": s3_mode, "records": records},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    meta = p.with_suffix(".json")
    meta.write_text(
        json.dumps({"bench": str(bench), "n": len(records), "s3_mode": s3_mode}, indent=2)
    )
    return p


def load_or_extract(bench, sessions, rcn, sbert, nlp, s3_mode: str = "keyword", s3_clf=None):
    cached = load_cached(bench, s3_mode)
    if cached is not None and len(cached) == len(sessions):
        print(f"[Cache] Loaded {len(cached)} signal records from {cache_path(bench, s3_mode)}")
        return cached
    print(f"[Cache] Extracting signals for {len(sessions)} sessions (will cache) ...")
    records = rcn.extract_signals(sessions, sbert, nlp, s3_mode=s3_mode, s3_clf=s3_clf)
    p = save_cache(bench, records, s3_mode)
    print(f"[Cache] Wrote {p}")
    return records

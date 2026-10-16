"""Paths and JSONL helpers shared by the v2 pipeline scripts."""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "data" / "external"
DERIVED = EXTERNAL / "derived"          # git-ignored: third-party text and per-session scores
RESULTS = ROOT / "experiments" / "results" / "v2"


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_jsonl(path: Path, key: str = "id") -> dict:
    return {r[key]: r for r in read_jsonl(path)}


def done_ids(path: Path, key: str = "id") -> set:
    """Ids already written to a resumable output file (empty if none)."""
    return set(index_jsonl(path, key)) if Path(path).exists() else set()


def stratified_subsample(sessions: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n/2 positives then n/2 negatives, sampled with one seeded RNG (the
    order matters: it fixes which sessions every scorer sees)."""
    rng = random.Random(seed)
    pos = [s for s in sessions if s["label"] == 1]
    neg = [s for s in sessions if s["label"] == 0]
    k = n // 2
    return rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(k, len(neg)))

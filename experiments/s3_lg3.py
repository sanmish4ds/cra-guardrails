#!/usr/bin/env python3
"""Shared Llama Guard 3 (Ollama) labeling and Cohen's kappa utilities for S3 audit."""

from __future__ import annotations

import json
import random
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

AUDIT_SEED = 42
AUDIT_N = 200
VAL_N = 200
KAPPA_GATE = 0.7
OLLAMA_URL = "http://localhost:11434/api/chat"
GUARD_MODEL = "llama-guard3:1b"


def ollama_available() -> bool:
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def guard_unsafe(conv: list[dict], timeout: float = 90.0) -> float | None:
    """LG3 via Ollama: 1.0 = unsafe, 0.0 = safe."""
    payload = json.dumps({
        "model": GUARD_MODEL,
        "messages": conv,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 32},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        text = body.get("message", {}).get("content", "").lower()
        if "unsafe" in text:
            return 1.0
        if "safe" in text:
            return 0.0
        return 1.0 if "yes" in text.split()[:3] else 0.0
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def collect_windows(sessions: list[dict], seed: int = AUDIT_SEED) -> list[dict]:
    pairs = []
    for s in sessions:
        user_idx = [i for i, t in enumerate(s["turns"]) if t["role"] == "user"]
        for ui, _ in enumerate(user_idx):
            conv = []
            u_seen = 0
            for t in s["turns"]:
                conv.append({"role": t["role"], "content": t["content"][:400]})
                if t["role"] == "user":
                    if u_seen == ui:
                        break
                    u_seen += 1
            text = "\n".join(
                f"{'User' if x['role'] == 'user' else 'Asst'}: {x['content']}"
                for x in conv
            )
            pairs.append({
                "session_id": s["id"],
                "turn_idx": ui,
                "text": text,
                "conv": conv,
                "label": s.get("label", 0),
                "family": s.get("family", ""),
            })
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs


def split_windows(
    all_windows: list[dict],
    audit_n: int = AUDIT_N,
    val_n: int = VAL_N,
) -> tuple[list[dict], list[dict], list[dict]]:
    audit = all_windows[:audit_n]
    val = all_windows[audit_n : audit_n + val_n]
    train = all_windows[audit_n + val_n :]
    return audit, val, train


def cohen_kappa(y1: Iterable[int], y2: Iterable[int]) -> float:
    a, b = list(y1), list(y2)
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    p1, p2 = sum(a) / n, sum(b) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def agreement(y1: Iterable[int], y2: Iterable[int]) -> float:
    a, b = list(y1), list(y2)
    if not a:
        return float("nan")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def best_threshold_kappa(
    probs: np.ndarray,
    y_true: list[int],
    grid: int = 181,
) -> tuple[float, float]:
    """Search threshold maximizing Cohen's kappa on (probs, y_true)."""
    best_t, best_k = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, grid):
        pred = (probs >= t).astype(int)
        k = cohen_kappa(pred.tolist(), y_true)
        if not np.isnan(k) and k > best_k:
            best_k, best_t = float(k), float(t)
    return best_t, best_k


def label_windows(
    windows: list[dict],
    progress_every: int = 50,
    tag: str = "label",
) -> tuple[list[str], list[int], int]:
    texts, labels, skipped = [], [], 0
    for i, w in enumerate(windows):
        gu = guard_unsafe(w["conv"])
        if gu is None:
            skipped += 1
            continue
        texts.append(w["text"])
        labels.append(int(gu))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  [{tag}] {i + 1}/{len(windows)} (skipped={skipped})")
    return texts, labels, skipped

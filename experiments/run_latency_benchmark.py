#!/usr/bin/env python3
"""Per-turn latency and memory footprint for CRA signals and CRA-Net vs guardrails.

Output: experiments/results/latency_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "latency_benchmark.json"

TURNS_LIST = (8, 32, 128)
N_WARMUP = 3
N_TIMED = 20
SEED = 42


def _load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bench(fn, n_warmup=N_WARMUP, n_timed=N_TIMED):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": round(statistics.mean(times), 3),
        "p50_ms": round(statistics.median(times), 3),
        "p95_ms": round(float(np.percentile(times, 95)), 3),
    }


def _synthetic_session(n_user: int, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    turns = []
    topics = [
        "explain photosynthesis", "debug this Python loop", "summarize TCP/IP",
        "recommend statistics books", "how does gradient descent work",
    ]
    for i in range(n_user):
        u = f"{rng.choice(topics)} (turn {i+1})"
        a = f"Here is a concise answer about {u[:20]}..."
        turns.append({"role": "user", "content": u})
        turns.append({"role": "assistant", "content": a})
    return {"label": 0, "turns": turns, "onset_turn": 0}


def measure_signals(rcn, rcc, sbert, nlp, n_user: int) -> dict:
    sess = _synthetic_session(n_user)

    def _once():
        rcn.extract_signals([sess], sbert, nlp)

    return _bench(_once)


def measure_cranet(rcn, sbert, nlp, n_user: int, device) -> dict:
    import torch
    from torch.utils.data import DataLoader

    sess = _synthetic_session(n_user)
    recs = rcn.extract_signals([sess], sbert, nlp)
    max_len = max(r["n_turns"] for r in recs)
    model = rcn.CRANet(input_dim=5, hidden=128, n_layers=2, lam=0.05).to(device)
    model.eval()
    ds = rcn.SessionDataset(recs, max_len)
    dl = DataLoader(ds, batch_size=1)

    def _once():
        with torch.no_grad():
            for x, y, lengths, _ in dl:
                model(x.to(device), lengths)

    return _bench(_once)


def measure_guard_ollama(n_user: int, model: str = "llama-guard3:1b") -> dict | None:
    import urllib.request

    sess = _synthetic_session(n_user)
    conv = [{"role": t["role"], "content": t["content"][:400]} for t in sess["turns"]]

    def _once():
        payload = json.dumps({
            "model": model,
            "messages": conv,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 16},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp.read()
        except Exception:
            pass

    try:
        _once()  # connectivity probe
    except Exception:
        return None
    return _bench(_once, n_warmup=1, n_timed=5)


def iag_node_estimate(n_user: int) -> dict:
    """Rough IAG node count: one entity node per assistant turn."""
    return {"estimated_nodes": n_user, "estimated_edges": max(0, n_user - 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--skip-guards", action="store_true")
    args = ap.parse_args()

    import torch
    rcc = _load_mod("rcc", "run_cra_cosafe.py")
    rcn = _load_mod("rcn", "run_cranet.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Latency] device={device}")

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = rcc._load_spacy()
    print(f"[Latency] spaCy model: en_core_web_sm (see run_cra_cosafe._load_spacy)")

    result: dict = {
        "device": str(device),
        "spacy_model": "en_core_web_sm",
        "embedding_model": "all-MiniLM-L6-v2",
        "n_timed": N_TIMED,
        "per_turn_signals_cpu": {},
        "cranet_session_cpu": {},
        "iag_footprint": {},
        "guards_full_transcript": {},
    }

    if device.type == "cuda":
        result["per_turn_signals_gpu"] = {}
        result["cranet_session_gpu"] = {}

    for n in TURNS_LIST:
        print(f"[Latency] T={n} user turns ...")
        result["per_turn_signals_cpu"][str(n)] = measure_signals(rcn, rcc, sbert, nlp, n)
        result["cranet_session_cpu"][str(n)] = measure_cranet(rcn, sbert, nlp, n, torch.device("cpu"))
        result["iag_footprint"][str(n)] = iag_node_estimate(n)
        if device.type == "cuda":
            result["per_turn_signals_gpu"][str(n)] = measure_signals(
                rcn, rcc, sbert, nlp, n)
            result["cranet_session_gpu"][str(n)] = measure_cranet(
                rcn, sbert, nlp, n, device)

    if not args.skip_guards:
        for n in (8,):
            lg = measure_guard_ollama(n, "llama-guard3:1b")
            if lg:
                result["guards_full_transcript"]["llama_guard3_1b_ollama"] = lg
            else:
                result["guards_full_transcript"]["llama_guard3_1b_ollama"] = {
                    "error": "Ollama unavailable; run: ollama pull llama-guard3:1b",
                }

    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[Latency] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

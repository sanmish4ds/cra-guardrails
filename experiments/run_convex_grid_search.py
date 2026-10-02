#!/usr/bin/env python3
"""Grid search (alpha, beta, gamma) for CRA-convex on CRA-Bench val split.

Output: experiments/results/convex_grid_search_v02_5fam.json
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
BENCH = PROJECT_ROOT / "data" / "cra_bench_v02_5fam" / "sessions.jsonl"
OUT_PATH = RESULTS_DIR / "convex_grid_search_v02_5fam.json"
SEED = 42


def _load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def convex_score(rec: dict, alpha: float, beta: float, gamma: float) -> float:
    scores = []
    for row in rec["features"]:
        s1, s2, s3 = row[0], row[1], row[2]
        scores.append(alpha * s1 + beta * s2 + gamma * s3)
    return float(max(scores)) if scores else 0.0


def main() -> int:
    ext = _load_mod("ext", "run_cra_extended_protocol.py")
    rcn = _load_mod("rcn", "run_cranet.py")
    rcc = _load_mod("rcc", "run_cra_cosafe.py")

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = rcc._load_spacy()

    sessions = ext.load_bench(BENCH)
    records = rcn.extract_signals(sessions, sbert, nlp)
    for r, s in zip(records, sessions):
        r["session_id"] = s["id"]
    by_id = {r["session_id"]: r for r in records}

    train_s, val_s, test_s = ext.stratified_split(sessions)
    val_r = [by_id[s["id"]] for s in val_s]
    test_r = [by_id[s["id"]] for s in test_s]

    grid = []
    step = 0.05
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    best_val = {"auroc": -1.0}
    best_test = {"auroc": -1.0}
    best_proper_val = {"auroc": -1.0}
    min_w = 0.05  # exclude single-signal corners (α,β,γ)=(1,0,0) etc.

    def _proper(a: float, b: float, g: float) -> bool:
        return min(a, b, g) >= min_w - 1e-9

    for a in vals:
        for b in vals:
            g = 1.0 - a - b
            if g < -1e-9 or g > 1.0 + 1e-9:
                continue
            g = max(0.0, min(1.0, g))
            if abs(a + b + g - 1.0) > 0.01:
                continue
            y_val = [r["label"] for r in val_r]
            sc_val = [convex_score(r, a, b, g) for r in val_r]
            auc_val = rcn._auc(y_val, sc_val)
            y_te = [r["label"] for r in test_r]
            sc_te = [convex_score(r, a, b, g) for r in test_r]
            auc_te = rcn._auc(y_te, sc_te)
            row = {
                "alpha": round(float(a), 2),
                "beta": round(float(b), 2),
                "gamma": round(float(g), 2),
                "val_auroc": round(auc_val, 4),
                "test_auroc": round(auc_te, 4),
            }
            grid.append(row)
            if auc_val > best_val["auroc"]:
                best_val = {**row, "auroc": auc_val}
            if auc_te > best_test["auroc"]:
                best_test = {**row, "auroc": auc_te}
            if _proper(a, b, g) and auc_val > best_proper_val["auroc"]:
                best_proper_val = {**row, "auroc": auc_val}

    grid.sort(key=lambda x: (-x["val_auroc"], -x["test_auroc"]))
    top10 = grid[:10]

    result = {
        "bench": str(BENCH),
        "n_grid_valid": len(grid),
        "default_weights": {"alpha": 0.35, "beta": 0.45, "gamma": 0.20},
        "default_test_auroc": round(
            rcn._auc(
                [r["label"] for r in test_r],
                [convex_score(r, 0.35, 0.45, 0.20) for r in test_r],
            ),
            4,
        ),
        "best_on_val": best_val,
        "best_on_test": best_test,
        "best_proper_mix_on_val": best_proper_val,
        "min_weight_proper_mix": min_w,
        "top10_by_val": top10,
        "recovered_auroc_ge_085_on_val": best_proper_val["val_auroc"] >= 0.85,
        "recovered_auroc_ge_085_corner_only": best_val["val_auroc"] >= 0.85,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[ConvexGrid] Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

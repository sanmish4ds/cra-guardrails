#!/usr/bin/env python3
"""Synthetic-to-real transfer: train on CRA-Bench, test on human-authored attacks.

Positive class: CoSafe gradual-escalation multi-turn sessions (human/red-team
authored; Yu et al., EMNLP 2024). Negative class: length-matched ShareGPT
3-user-turn benign chats. Models are trained ONLY on the CRA-Bench v0.2
standard train split and calibrated on the CRA-Bench val split; the human
corpus is never used for training or threshold selection.

Output: experiments/results/human_transfer_eval.json
        data/human_cra_transfer/sessions.jsonl  (exported benchmark)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
DATA_DIR = PROJECT_ROOT / "data" / "human_cra_transfer"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOT = 1000
N_MATCHED_NEG = 750
LAMBDA_GRL = 0.05
LAM_FAM = 0.1
EPOCHS = 50
BENIGN_BENCH = PROJECT_ROOT / "data" / "cra_bench_v02_5fam" / "sessions.jsonl"
OUT_PATH = RESULTS_DIR / "human_transfer_eval.json"
EXPORT_PATH = DATA_DIR / "sessions.jsonl"


def _load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_human_transfer_corpus(rcc, bfp, export_path: Path = EXPORT_PATH) -> list[dict]:
    """CoSafe positives + ShareGPT 3-turn benign negatives (length-matched)."""
    if export_path.exists():
        sessions = []
        with export_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                sessions.append({
                    "id": row["session_id"],
                    "label": int(row["label"]),
                    "turns": row["turns"],
                    "onset_turn": int(row.get("onset_turn", 0)),
                    "family": row.get("cra_type", "human"),
                    "source": row.get("source", "unknown"),
                })
        print(f"[HumanXfer] Loaded {len(sessions)} sessions from {export_path}")
        return sessions

    print("[HumanXfer] Building human transfer corpus ...")
    cosafe = rcc.load_cosafe()
    pos = [s for s in cosafe if s["label"] == 1]
    print(f"[HumanXfer] {len(pos)} CoSafe positive sessions")

    sharegpt = bfp.load_sharegpt(N_MATCHED_NEG + 300, seed=SEED + 1)
    three_turn = [
        s for s in sharegpt
        if len([t for t in s["turns"] if t["role"] == "user"]) == 3
    ]
    print(f"[HumanXfer] {len(three_turn)} ShareGPT 3-user-turn benign sessions")
    rng = random.Random(SEED)
    n_neg = min(len(three_turn), len(pos), N_MATCHED_NEG)
    neg = rng.sample(three_turn, n_neg)

    sessions: list[dict] = []
    with export_path.open("w", encoding="utf-8") as out:
        for i, s in enumerate(pos):
            row = {
                "session_id": f"cosafe_pos_{i:04d}",
                "label": 1,
                "turns": s["turns"],
                "onset_turn": int(s.get("onset_turn", 0)),
                "cra_type": "cosafe_gradual_escalation",
                "bench_version": "human_transfer_v1",
                "source": "cosafe",
            }
            out.write(json.dumps(row) + "\n")
            sessions.append({
                "id": row["session_id"],
                "label": 1,
                "turns": s["turns"],
                "onset_turn": row["onset_turn"],
                "family": "cosafe_gradual_escalation",
                "source": "cosafe",
            })
        for i, s in enumerate(neg):
            row = {
                "session_id": f"sharegpt_neg_{i:04d}",
                "label": 0,
                "turns": s["turns"],
                "onset_turn": 0,
                "cra_type": "sharegpt_benign_3turn",
                "bench_version": "human_transfer_v1",
                "source": "sharegpt",
            }
            out.write(json.dumps(row) + "\n")
            sessions.append({
                "id": row["session_id"],
                "label": 0,
                "turns": s["turns"],
                "onset_turn": 0,
                "family": "sharegpt_benign_3turn",
                "source": "sharegpt",
            })

    print(f"[HumanXfer] Exported {len(sessions)} sessions -> {export_path}")
    return sessions


def attach_records(sessions: list[dict], rcn, sbert, nlp) -> list[dict]:
    records = rcn.extract_signals(sessions, sbert, nlp)
    for r, s in zip(records, sessions):
        r["session_id"] = s["id"]
        r["family"] = s.get("family", "human")
        r["source"] = s.get("source", "unknown")
    return records


def main() -> int:
    from dotenv import load_dotenv
    import torch

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=BENIGN_BENCH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--export", type=Path, default=EXPORT_PATH)
    ap.add_argument("--rebuild-corpus", action="store_true")
    args = ap.parse_args()

    if args.rebuild_corpus and args.export.exists():
        args.export.unlink()

    rcc = _load_mod("rcc", "run_cra_cosafe.py")
    rcn = _load_mod("rcn", "run_cranet.py")
    ext = _load_mod("ext", "run_cra_extended_protocol.py")
    bfp = _load_mod("bfp", "run_benign_fpr_sharegpt.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HumanXfer] Device: {device}")

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = rcc._load_spacy()

    t0 = time.time()
    human_sessions = build_human_transfer_corpus(rcc, bfp, args.export)
    print("[HumanXfer] Extracting signals for human transfer corpus ...")
    human_recs = attach_records(human_sessions, rcn, sbert, nlp)

    print(f"[HumanXfer] Loading synthetic bench: {args.bench}")
    synth_sessions = ext.load_bench(args.bench)
    POSITIVE_FAMILIES = ext.autodetect_families(synth_sessions)
    fam_to_idx = {f: i for i, f in enumerate(POSITIVE_FAMILIES)}

    print("[HumanXfer] Extracting signals for synthetic bench ...")
    synth_records = attach_records(synth_sessions, rcn, sbert, nlp)
    by_id = {r["session_id"]: r for r in synth_records}

    train_s, val_s, _test_s = ext.stratified_split(synth_sessions)
    train_r = [by_id[s["id"]] for s in train_s]
    val_r = [by_id[s["id"]] for s in val_s]

    for r in train_r:
        fam = r["family"]
        r["family_idx"] = fam_to_idx.get(fam, 0) if r["label"] == 1 else 0

    # ── Baselines that need no training ───────────────────────────────────────
    y_h = [r["label"] for r in human_recs]
    methods: dict[str, list[float]] = {
        "CRA-convex": [ext.convex_session_score(r) for r in human_recs],
        "Turn-max S1": [ext.turn_max_s1(r) for r in human_recs],
    }

    mlp_w = ext.fit_mlp(train_r)
    methods["Feature-MLP"] = [ext.mlp_score(r, mlp_w) for r in human_recs]

    # ── CRA-Net (train synthetic, test human) ───────────────────────────────
    print("[HumanXfer] Training CRA-Net (lambda=0.05) on synthetic train ...")
    model_full, max_len = ext.train_cranet(
        rcn, train_r, val_r, device, lam=LAMBDA_GRL, input_dim=5, epochs=EPOCHS)
    _, val_out = ext.predict_cranet(model_full, rcn, val_r, device, max_len)
    val_scores = [max(r["preds"]) if r.get("preds") else 0.0 for r in val_out]
    val_labels = [r["label"] for r in val_out]
    _, theta_full = rcn.compute_sfpr(val_labels, val_scores)
    scores_full, _ = ext.predict_cranet(model_full, rcn, human_recs, device, max_len)
    methods["CRA-Net (lambda=0.05)"] = scores_full

    print("[HumanXfer] Training CRA-Net DA on synthetic train ...")
    model_da, max_len_da = ext.train_cranet_da(
        rcn, train_r, val_r, device, lam=LAMBDA_GRL, lam_fam=LAM_FAM,
        input_dim=5, epochs=EPOCHS)
    _, val_out_da = ext.predict_cranet(model_da, rcn, val_r, device, max_len_da)
    val_scores_da = [max(r["preds"]) if r.get("preds") else 0.0 for r in val_out_da]
    _, theta_da = rcn.compute_sfpr(val_labels, val_scores_da)
    scores_da, _ = ext.predict_cranet(model_da, rcn, human_recs, device, max_len_da)
    methods["CRA-Net DA"] = scores_da

    # ── Metrics ─────────────────────────────────────────────────────────────
    def _summarize(name: str, scores: list[float], theta: float | None) -> dict[str, Any]:
        auroc = rcn._auc(y_h, scores)
        auc_m, auc_lo, auc_hi = rcn.bootstrap_auc(y_h, scores, n_boot=N_BOOT, seed=SEED)
        sfpr, thr = rcn.compute_sfpr(y_h, scores)
        sf_lo, sf_hi = rcn.bootstrap_sfpr(y_h, scores, n_boot=N_BOOT, seed=SEED)
        row: dict[str, Any] = {
            "auroc": round(auroc, 4),
            "auroc_ci95": [round(auc_lo, 4), round(auc_hi, 4)],
            "sfpr_at_tpr90": round(sfpr, 4),
            "sfpr_ci95": [round(sf_lo, 4), round(sf_hi, 4)],
            "threshold_tpr90": round(float(thr), 4),
        }
        if theta is not None:
            n_neg = sum(1 for y in y_h if not y)
            n_pos = sum(y_h)
            tpr = sum(1 for y, s in zip(y_h, scores) if y and s >= theta) / max(n_pos, 1)
            fpr = sum(1 for y, s in zip(y_h, scores) if not y and s >= theta) / max(n_neg, 1)
            row["synth_val_threshold"] = round(float(theta), 4)
            row["tpr_at_synth_threshold"] = round(tpr, 4)
            row["fpr_at_synth_threshold"] = round(fpr, 4)
        return row

    thetas = {
        "CRA-Net (lambda=0.05)": theta_full,
        "CRA-Net DA": theta_da,
    }

    overall = {name: _summarize(name, sc, thetas.get(name))
               for name, sc in methods.items()}

    n_pos = sum(y_h)
    n_neg = len(y_h) - n_pos
    # length-only diagnostic (degenerate when all sessions share turn count)
    n_turns = [r["n_turns"] for r in human_recs]
    if len(set(n_turns)) <= 1:
        len_auroc = 0.5
    else:
        len_auroc = rcn._auc(y_h, n_turns)

    result = {
        "protocol": "human_transfer_v1",
        "description": (
            "Train on CRA-Bench v0.2 synthetic (standard split train/val); "
            "zero-shot eval on CoSafe gradual-escalation positives + "
            "ShareGPT 3-turn benign negatives. Thresholds from synthetic val @ TPR=0.90."
        ),
        "synthetic_bench": str(args.bench),
        "human_corpus": str(args.export),
        "n_human_total": len(human_recs),
        "n_human_pos_cosafe": n_pos,
        "n_human_neg_sharegpt": n_neg,
        "auroc_length_only": round(len_auroc, 4),
        "wall_seconds": round(time.time() - t0, 1),
        "methods": overall,
    }
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"[HumanXfer] Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

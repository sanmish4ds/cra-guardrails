#!/usr/bin/env python3
"""CRA-Net training-init variance: fixed session split (seed 42), varying train seeds.

Output: experiments/results/cra_multiseed_v02_5fam.json

Fast mode (--fast): 3 seeds, 15 epochs, 3 configs (no no-GRL), signal cache.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCH = PROJECT_ROOT / "data" / "cra_bench_v02_5fam" / "sessions.jsonl"
OUT_PATH = SCRIPT_DIR / "results" / "cra_multiseed_v02_5fam.json"
SPLIT_SEED = 42
DEFAULT_TRAIN_SEEDS = [42, 43, 44, 45, 46]
FAST_TRAIN_SEEDS = [42, 43, 44]
FAST_EPOCHS = 15


def _load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prepare_records(
    ext, rcn, bench: Path, split_seed: int, use_cache: bool, s3_mode: str = "keyword",
):
    sessions = ext.load_bench(bench)
    ext.POSITIVE_FAMILIES = ext.autodetect_families(sessions)
    from sentence_transformers import SentenceTransformer

    if use_cache:
        from signal_cache import load_or_extract
        rcc = _load_mod("rcc", "run_cra_cosafe.py")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        nlp = rcc._load_spacy()
        s3_clf = None
        if s3_mode == "classifier":
            from s3_refusal import ClassifierS3
            s3_clf = ClassifierS3()
        records = load_or_extract(
            bench, sessions, rcn, sbert, nlp, s3_mode=s3_mode, s3_clf=s3_clf)
    else:
        rcc = _load_mod("rcc", "run_cra_cosafe.py")
        sbert = SentenceTransformer("all-MiniLM-L6-v2")
        nlp = rcc._load_spacy()
        s3_clf = None
        if s3_mode == "classifier":
            from s3_refusal import ClassifierS3
            s3_clf = ClassifierS3()
        records = rcn.extract_signals(
            sessions, sbert, nlp, s3_mode=s3_mode, s3_clf=s3_clf)
    for r, s in zip(records, sessions):
        r["session_id"] = s["id"]
        r["family"] = s.get("family", "")
    by_id = {r["session_id"]: r for r in records}
    train_s, val_s, test_s = ext.stratified_split(sessions, seed=split_seed)
    return sessions, by_id, train_s, val_s, test_s


def run_train_seed(
    ext, rcn, train_seed: int, lam_fam: float | None, device,
    prepared: tuple, epochs: int,
) -> dict[str, float]:
    import torch

    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)

    _, by_id, train_s, val_s, test_s = prepared
    train_r = [by_id[s["id"]] for s in train_s]
    val_r = [by_id[s["id"]] for s in val_s]
    test_r = [by_id[s["id"]] for s in test_s]

    if lam_fam is None:
        model, max_len = ext.train_cranet(
            rcn, train_r, val_r, device, lam=ext.LAMBDA_GRL, epochs=epochs)
    else:
        train_da = ext.assign_family_idx(train_r, list(ext.POSITIVE_FAMILIES))
        val_da = ext.assign_family_idx(val_r, list(ext.POSITIVE_FAMILIES))
        model, max_len = ext.train_cranet_da(
            rcn, train_da, val_da, device, lam=ext.LAMBDA_GRL,
            lam_fam=lam_fam, epochs=epochs)

    _, val_out = ext.predict_cranet(model, rcn, val_r, device, max_len)
    val_scores = [max(r["preds"]) if r["preds"] else 0.0 for r in val_out]
    _, theta = rcn.compute_sfpr([r["label"] for r in val_out], val_scores)

    _, test_out = ext.predict_cranet(model, rcn, test_r, device, max_len)
    test_scores = [max(r["preds"]) if r["preds"] else 0.0 for r in test_out]
    y_te = [r["label"] for r in test_r]
    auc = rcn._auc(y_te, test_scores)
    flagged = sum(1 for y, s in zip(y_te, test_scores) if y == 0 and s >= theta)
    n_neg = sum(1 for y in y_te if y == 0)
    sfpr = flagged / n_neg if n_neg else float("nan")
    return {"auroc": float(auc), "sfpr_at_tpr90": float(sfpr), "train_seed": train_seed}


def run_train_seed_nogrl(ext, rcn, train_seed, device, prepared, epochs):
    import torch
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    _, by_id, train_s, val_s, test_s = prepared
    train_r = [by_id[s["id"]] for s in train_s]
    val_r = [by_id[s["id"]] for s in val_s]
    test_r = [by_id[s["id"]] for s in test_s]
    model, max_len = ext.train_cranet(rcn, train_r, val_r, device, lam=0.0, epochs=epochs)
    _, val_out = ext.predict_cranet(model, rcn, val_r, device, max_len)
    val_scores = [max(r["preds"]) if r["preds"] else 0.0 for r in val_out]
    _, theta = rcn.compute_sfpr([r["label"] for r in val_out], val_scores)
    _, test_out = ext.predict_cranet(model, rcn, test_r, device, max_len)
    test_scores = [max(r["preds"]) if r["preds"] else 0.0 for r in test_out]
    y_te = [r["label"] for r in test_r]
    auc = rcn._auc(y_te, test_scores)
    flagged = sum(1 for y, s in zip(y_te, test_scores) if y == 0 and s >= theta)
    n_neg = sum(1 for y in y_te if y == 0)
    return {
        "auroc": float(auc),
        "sfpr_at_tpr90": flagged / max(n_neg, 1),
        "train_seed": train_seed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    ap.add_argument("--train-seeds", type=int, nargs="+", default=None)
    ap.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fast", action="store_true",
                    help="3 seeds, 15 epochs, skip no-GRL, use signal cache.")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--s3-mode", choices=("keyword", "classifier"), default="keyword")
    args = ap.parse_args()

    fast = args.fast
    train_seeds = args.train_seeds or (FAST_TRAIN_SEEDS if fast else DEFAULT_TRAIN_SEEDS)
    epochs = args.epochs or (FAST_EPOCHS if fast else 50)
    use_cache = not args.no_cache

    ext = _load_mod("ext", "run_cra_extended_protocol.py")
    rcn = _load_mod("rcn", "run_cranet.py")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[MultiSeed] fast={fast} seeds={train_seeds} epochs={epochs} "
          f"cache={use_cache} s3={args.s3_mode}")
    prepared = _prepare_records(
        ext, rcn, args.bench, args.split_seed, use_cache, s3_mode=args.s3_mode)

    configs = (
        {"CRA-Net_full": None, "CRA-Net_DA_lam01": 0.1, "CRA-Net_DA_lam03": 0.3}
        if fast
        else {
            "CRA-Net_full": None,
            "CRA-Net_no_GRL": "nogrl",
            "CRA-Net_DA_lam01": 0.1,
            "CRA-Net_DA_lam03": 0.3,
        }
    )

    out: dict[str, Any] = {
        "bench": str(args.bench),
        "split_seed": args.split_seed,
        "train_seeds": train_seeds,
        "epochs": epochs,
        "fast_mode": fast,
        "signal_cache": use_cache,
        "s3_mode": args.s3_mode,
        "methods": {},
    }
    for name, lam in configs.items():
        runs = []
        for ts in train_seeds:
            print(f"[MultiSeed] {name} train_seed={ts}", flush=True)
            if lam == "nogrl":
                runs.append(run_train_seed_nogrl(
                    ext, rcn, ts, device, prepared, epochs))
            else:
                runs.append(run_train_seed(
                    ext, rcn, ts, lam, device, prepared, epochs))
        out["methods"][name] = {
            "per_seed": runs,
            "auroc_mean": round(statistics.mean(r["auroc"] for r in runs), 4),
            "auroc_std": round(statistics.pstdev(r["auroc"] for r in runs), 4),
            "sfpr_mean": round(statistics.mean(r["sfpr_at_tpr90"] for r in runs), 4),
            "sfpr_std": round(statistics.pstdev(r["sfpr_at_tpr90"] for r in runs), 4),
        }

    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"[MultiSeed] Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

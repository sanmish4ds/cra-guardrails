#!/usr/bin/env python3
"""Per-family aggregation AUROC on CRA-Bench v0.3 (fixed benign twins) vs keyword S3."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
V03 = PROJECT_ROOT / "data" / "cra_bench_v03_5fam" / "sessions.jsonl"
OUT = SCRIPT_DIR / "results" / "v03_aggregation_column.json"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eval_aggregation(bench: Path, s3_mode: str = "keyword") -> dict:
    ext = _load("ext", "run_cra_extended_protocol.py")
    rcn = _load("rcn", "run_cranet.py")
    import torch
    from sentence_transformers import SentenceTransformer

    rcc = _load("rcc", "run_cra_cosafe.py")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = rcc._load_spacy()
    clf = None
    if s3_mode == "classifier":
        from s3_refusal import ClassifierS3
        clf = ClassifierS3()

    sessions = ext.load_bench(bench)
    from signal_cache import load_or_extract
    records = load_or_extract(bench, sessions, rcn, sbert, nlp, s3_mode=s3_mode, s3_clf=clf)
    for r, s in zip(records, sessions):
        r["session_id"] = s["id"]
        r["family"] = s.get("family", "")
    by_id = {r["session_id"]: r for r in records}

    train_s, val_s, test_s = ext.stratified_split(sessions, seed=42)
    train_r = [by_id[s["id"]] for s in train_s]
    val_r = [by_id[s["id"]] for s in val_s]
    test_r = [by_id[s["id"]] for s in test_s]
    device = torch.device("cpu")

    # convex on test
    convex = [ext.convex_session_score(r) for r in test_r]
    y = [r["label"] for r in test_r]
    conv_auc = rcn._auc(y, convex)
    pf_conv = ext.per_family_auroc(test_r, convex)

    model, max_len = ext.train_cranet(rcn, train_r, val_r, device, lam=0.05, epochs=15)
    _, test_out = ext.predict_cranet(model, rcn, test_r, device, max_len)
    cranet_sc = [max(r["preds"]) if r["preds"] else 0.0 for r in test_out]
    pf_cn = ext.per_family_auroc(test_r, cranet_sc)

    return {
        "s3_mode": s3_mode,
        "convex_overall": round(conv_auc, 4),
        "cranet_overall": round(rcn._auc(y, cranet_sc), 4),
        "convex_aggregation": round(pf_conv.get("aggregation", {}).get("auroc", 0), 4),
        "cranet_aggregation": round(pf_cn.get("aggregation", {}).get("auroc", 0), 4),
        "convex_per_family": {k: v.get("auroc") for k, v in pf_conv.items()},
        "cranet_per_family": {k: v.get("auroc") for k, v in pf_cn.items()},
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--s3-mode", choices=("keyword", "classifier", "both"), default="both")
    args = ap.parse_args()

    out = {"v03_bench": str(V03)}
    if args.s3_mode in ("keyword", "both"):
        out["keyword_s3"] = eval_aggregation(V03, "keyword")
    if args.s3_mode in ("classifier", "both"):
        try:
            from s3_refusal import ClassifierS3
            if ClassifierS3().available():
                out["classifier_s3"] = eval_aggregation(V03, "classifier")
            else:
                out["classifier_s3_error"] = "model missing; run train_s3_refusal_classifier.py"
        except Exception as e:
            out["classifier_s3_error"] = str(e)
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

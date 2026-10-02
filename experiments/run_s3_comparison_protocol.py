#!/usr/bin/env python3
"""Compare keyword vs classifier S3 on CRA-Bench extended protocol metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BENCH = PROJECT_ROOT / "data" / "cra_bench_v02_5fam" / "sessions.jsonl"
OUT = SCRIPT_DIR / "results" / "s3_comparison_protocol.json"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_protocol(s3_mode: str) -> dict:
    ext = _load("ext", "run_cra_extended_protocol.py")
    rcn = _load("rcn", "run_cranet.py")
    from sentence_transformers import SentenceTransformer
    import torch

    rcc = _load("rcc", "run_cra_cosafe.py")
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = rcc._load_spacy()
    clf = None
    if s3_mode == "classifier":
        from s3_refusal import ClassifierS3
        clf = ClassifierS3()

    sessions = ext.load_bench(BENCH)
    from signal_cache import load_or_extract
    records = load_or_extract(BENCH, sessions, rcn, sbert, nlp, s3_mode=s3_mode, s3_clf=clf)
    for r, s in zip(records, sessions):
        r["session_id"] = s["id"]
        r["family"] = s["family"]

    train_s, val_s, test_s = ext.stratified_split(sessions)
    by_id = {r["session_id"]: r for r in records}
    train_r = [by_id[s["id"]] for s in train_s]
    val_r = [by_id[s["id"]] for s in val_s]
    test_r = [by_id[s["id"]] for s in test_s]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_da = ext.assign_family_idx(train_r, list(ext.autodetect_families(sessions)))
    val_da = ext.assign_family_idx(val_r, list(ext.autodetect_families(sessions)))
    model, max_len = ext.train_cranet_da(
        rcn, train_da, val_da, device, lam=0.05, lam_fam=0.1, epochs=15)
    scores, _ = ext.predict_cranet(model, rcn, test_r, device, max_len)
    y = [r["label"] for r in test_r]
    overall_auc = rcn._auc(y, scores)
    pf = ext.per_family_auroc(test_r, scores)
    cond = pf.get("conditioning", {})
    cond_auc = cond.get("auroc") if isinstance(cond, dict) else cond
    return {
        "s3_mode": s3_mode,
        "overall_auroc": round(overall_auc, 4),
        "per_family": {
            k: round((v.get("auroc") if isinstance(v, dict) else v) or 0, 4)
            for k, v in pf.items()
        },
        "conditioning_auroc": round(cond_auc or 0, 4),
    }


def main() -> int:
    kw = run_protocol("keyword")
    cl = run_protocol("classifier")
    out = {"bench": str(BENCH), "keyword": kw, "classifier": cl}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

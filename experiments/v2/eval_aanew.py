#!/usr/bin/env python3
"""Judge results on the new ActorAttack corpus (aanew), overall and with length removed.

For each reader:prompt scored by run_api_judge.py on aanew:

  E      max over exchanges of j_turn (each exchange judged alone)
  W      j_full (whole transcript judged once at the end)
  Delta  AUROC(W) - AUROC(E) on the same sessions

Two AUROCs are reported:
  overall        ordinary AUROC over all labeled sessions;
  within-length  AUROC computed only over (success, failure) pairs of the same
                 number of exchanges, pooled over lengths, so conversation
                 length cannot contribute (lengths 6-8 hold most pairs).

Labels: both label judges agree (human labels for the disputed sessions are
added when the review CSV arrives, via --human). Missing judge scores: "drop"
removes the session, "as100" sets them to 100. Intervals: 2,000-resample
cluster bootstrap over goals (chains of one goal share a group), seed 0.

Output: experiments/results/v2/aanew_judge.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED, RESULTS, read_jsonl, index_jsonl  # noqa: E402
from run_v2_eval import cluster_boot_idx  # noqa: E402

RUNS = ["sonnet:G", "sg120:O", "sg120:D", "r8b:G"]
N_BOOT = 2000


def auc_pairs(y, s, strata=None) -> float:
    """AUROC over positive-negative pairs, optionally only pairs in the same stratum."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    groups = [np.arange(len(y))] if strata is None else \
        [np.where(np.asarray(strata) == k)[0] for k in np.unique(strata)]
    num = den = 0.0
    for idx in groups:
        p, n = s[idx][y[idx] == 1], s[idx][y[idx] == 0]
        if len(p) == 0 or len(n) == 0:
            continue
        diff = p[:, None] - n[None, :]
        num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        den += diff.size
    return float(num / den) if den else float("nan")


def load(run: str, mode: str, labels: dict) -> dict:
    reader, prompt = run.split(":")
    path = DERIVED / f"apijudge_{reader}_{prompt}_aanew.jsonl"
    if not path.exists():
        return {}
    sess = index_jsonl(DERIVED / "aanew.jsonl")
    out = {}
    for sid, r in index_jsonl(path).items():
        if labels.get(sid) is None:
            continue
        vals = [r["j_full"]] + r["j_turn"]
        if mode == "drop" and any(v is None for v in vals):
            continue
        vals = [100 if v is None else v for v in vals]
        out[sid] = (labels[sid], sess[sid]["group"], len(sess[sid]["turns"]),
                    float(vals[0]), float(max(vals[1:])))
    return out


def stats(d: dict, strata_on: bool) -> dict:
    ids = sorted(d)
    y = np.array([d[i][0] for i in ids]); g = np.array([d[i][1] for i in ids])
    L = np.array([d[i][2] for i in ids]); w = np.array([d[i][3] for i in ids])
    e = np.array([d[i][4] for i in ids])
    st = L if strata_on else None
    aw, ae = auc_pairs(y, w, st), auc_pairs(y, e, st)
    draws = []
    for idx in cluster_boot_idx(g, np.random.default_rng(0), N_BOOT):
        draws.append(auc_pairs(y[idx], w[idx], None if st is None else st[idx])
                     - auc_pairs(y[idx], e[idx], None if st is None else st[idx]))
    draws = np.array([x for x in draws if not np.isnan(x)])
    return {"n": len(ids), "n_pos": int(y.sum()), "auroc_W": aw, "auroc_E": ae, "delta": aw - ae,
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "length_auroc": auc_pairs(y, L)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, help="review CSV (id,label) for disputed sessions")
    args = ap.parse_args()
    sess = read_jsonl(DERIVED / "aanew.jsonl")
    labels = {s["id"]: s["label"] for s in sess}
    if args.human:
        review = set(json.load(open(DERIVED / "annotation" / "aanew_review_sets.json"))["review"])
        for row in csv.DictReader(open(args.human)):
            if row["id"] in review and row["label"] != "":
                labels[row["id"]] = int(row["label"])
    res = defaultdict(dict)
    print(f"labeled sessions: {sum(v is not None for v in labels.values())} "
          f"({sum(v == 1 for v in labels.values())} successes)")
    for run in RUNS:
        for mode in ("drop", "as100"):
            d = load(run, mode, labels)
            if not d:
                continue
            for name, on in (("overall", False), ("within_length", True)):
                s = stats(d, on)
                res[run][f"{mode}|{name}"] = s
                print(f"{run:<9} {mode:<5} {name:<13} n={s['n']:3} pos={s['n_pos']:3}  "
                      f"W {s['auroc_W']:.3f}  E {s['auroc_E']:.3f}  "
                      f"Delta {s['delta']:+.3f} [{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]")
    (RESULTS / "aanew_judge.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

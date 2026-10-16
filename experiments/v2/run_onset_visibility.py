#!/usr/bin/env python3
"""Onset visibility: is the first unsafe exchange detectable on its own?

For corpora with annotated onsets (BAD, ProsocialDialog), compare positives'
guard scores AT the onset exchange tau with negatives' scores at the same
exchange index (negatives are length-matched on ProsocialDialog and nearly
all seven exchanges on BAD). Three views:

  stateless prompt  g_user[tau]   (the onset user message alone)
  stateless exchange g_resp[tau]
  context-aware      g_ctx[tau]   (history + onset user message)

AUROC(context) - AUROC(stateless prompt) at the onset measures how much the
history is needed to recognize the onset utterance as unsafe. 95% CIs by
bootstrap over sessions (2,000 resamples), paired across views.

Output: experiments/results/v2/onset_visibility.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v2_eval as e  # noqa: E402
from common import RESULTS  # noqa: E402

VIEWS = ("g_user", "g_resp", "g_ctx")


def onset_scores(recs, guard_suffix=""):
    rng = np.random.default_rng(0)
    pos = [r for r in recs if r["label"] == 1 and r["onset"] is not None]
    neg = [r for r in recs if r["label"] == 0]
    rows = []
    for r in pos:
        t = r["onset"]
        rows.append((1, {v: r[v][t] for v in VIEWS}))
    # Each negative contributes one exchange, drawn from the positives' onset
    # distribution (restricted to exchanges the negative has).
    onsets = np.array([r["onset"] for r in pos])
    for r in neg:
        cand = onsets[onsets < r["T"]]
        t = int(rng.choice(cand)) if len(cand) else r["T"] - 1
        rows.append((0, {v: r[v][t] for v in VIEWS}))
    y = np.array([a for a, _ in rows])
    S = {v: np.array([b[v] for _, b in rows]) for v in VIEWS}
    return y, S


def analyse(y, S, n_boot=2000):
    rng = np.random.default_rng(0)
    out = {v: float(roc_auc_score(y, S[v])) for v in VIEWS}
    diffs = {"ctx_minus_user": [], "ctx_minus_resp": []}
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        a = {v: roc_auc_score(y[i], S[v][i]) for v in VIEWS}
        diffs["ctx_minus_user"].append(a["g_ctx"] - a["g_user"])
        diffs["ctx_minus_resp"].append(a["g_ctx"] - a["g_resp"])
    for k, v in diffs.items():
        out[k] = {"mean": float(out["g_ctx"] - out["g_user" if "user" in k else "g_resp"]),
                  "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    out["n_pos"] = int(y.sum())
    out["n_neg"] = int((y == 0).sum())
    return out


def main():
    res = {}
    for c in ("bad", "prosocial"):
        y, S = onset_scores(e.load(c, e.Setup()))
        res[c] = analyse(y, S)
        print(c, json.dumps(res[c]))
    (RESULTS / "onset_visibility.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

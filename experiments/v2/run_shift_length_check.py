#!/usr/bin/env python3
"""Is the stateful advantage under HH target-model shift a length effect?

Same splits as run_v2_eval.hh_model_shift (red-teamer- and model-disjoint,
5 random halvings). For each detector we report pooled test AUROC and the
mean of within-length-stratum AUROCs (strata with >= 30 sessions and both
classes), plus the class-wise mean exchange count.

Output: experiments/results/v2/hh_shift_length_check.json
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
from detectors import CRAAcc, TurnLR  # noqa: E402

SRC = {"context distillation", "plain lm"}


def main():
    recs = e.load("hh_rt", e.Setup())
    groups = sorted({r["group"] for r in recs})
    out = {"reps": [], "detectors": ["Turn-LR", "CRA-Acc[L]", "CRA-Acc[L] (no t)", "CRA-Acc (no t)"]}
    for rep in range(5):
        rng = np.random.default_rng(rep)
        half = set(rng.permutation(groups)[: len(groups) // 2])
        tr = [r for r in recs if r["group"] in half and r["meta"]["model_type"] in SRC]
        te = [r for r in recs if r["group"] not in half and r["meta"]["model_type"] not in SRC]
        y = np.array([r["label"] for r in te])
        T = np.array([r["T"] for r in te])
        ytr = np.array([r["label"] for r in tr])
        Ttr = np.array([r["T"] for r in tr])
        row = {"len_mean_pos": float(T[y == 1].mean()), "len_mean_neg": float(T[y == 0].mean()),
               "length_auroc_test": float(roc_auc_score(y, T)),
               "length_auroc_train": float(roc_auc_score(ytr, Ttr))}
        dets = {"Turn-LR": TurnLR("L"), "CRA-Acc[L]": CRAAcc("L"),
                "CRA-Acc[L] (no t)": CRAAcc("L", with_time=False),
                "CRA-Acc (no t)": CRAAcc("LCX", with_time=False)}
        for n, d in dets.items():
            d.fit(tr)
            s = np.array([e.online_scores(d, r).max() for r in te])
            strata = [roc_auc_score(y[T == L], s[T == L]) for L in np.unique(T)
                      if (T == L).sum() >= 30 and len(set(y[T == L])) == 2]
            row[n] = {"auroc": float(roc_auc_score(y, s)), "length_stratified_auroc": float(np.mean(strata))}
        out["reps"].append(row)
    summ = {}
    for n in out["detectors"]:
        for k in ("auroc", "length_stratified_auroc"):
            v = [r[n][k] for r in out["reps"]]
            summ.setdefault(n, {})[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v))}
    for k in ("len_mean_pos", "len_mean_neg", "length_auroc_test", "length_auroc_train"):
        summ[k] = float(np.mean([r[k] for r in out["reps"]]))
    out["summary"] = summ
    (RESULTS / "hh_shift_length_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

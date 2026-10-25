#!/usr/bin/env python3
"""Does the accumulator gain on MT-ActorAttack (Llama 3.2) come from timing?

Every Llama chain has five exchanges except two negatives. With the
log(1+t) feature and end-of-session fitting, standardization makes that
feature nearly constant in training and strongly negative on earlier
prefixes, so the fitted model may suppress alarms before the last exchange
and act as a final-exchange scorer. For CRA-Acc[L] with and without
log(1+t), and for Turn-LR, we report (seed-0 grouped CV, same folds) the
AUROC of the online maximum and of the final-exchange score, and where the
online maximum falls. Subsets: all Llama sessions, and those with T = 5.

Output: experiments/results/v2[/granite]/actorattack_time_gating.json
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v2_eval as e  # noqa: E402
from common import RESULTS  # noqa: E402
from detectors import CRAAcc, TurnLR  # noqa: E402

DETECTORS = {"CRA-Acc[L]": lambda: CRAAcc("L"), "CRA-Acc[L] (no t)": lambda: CRAAcc("L", with_time=False),
             "Turn-LR": lambda: TurnLR("L")}


def analyse(sub):
    y = np.array([r["label"] for r in sub])
    groups = np.array([r["group"] for r in sub])
    s_max = {n: np.zeros(len(sub)) for n in DETECTORS}
    s_fin = {n: np.zeros(len(sub)) for n in DETECTORS}
    argmax = {n: collections.Counter() for n in DETECTORS}
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    for tr, te in skf.split(np.zeros(len(sub)), y, groups):
        for n, make in DETECTORS.items():
            det = make().fit([sub[i] for i in tr])
            for i in te:
                s = det.turn_scores(sub[i])
                s_max[n][i], s_fin[n][i] = s.max(), s[-1]
                argmax[n][int(np.argmax(s)) + 1] += 1
    return {"n": len(sub), "n_pos": int(y.sum()),
            "detectors": {n: {"auroc_online_max": e.auroc(y, s_max[n]), "auroc_final": e.auroc(y, s_fin[n]),
                              "argmax_exchange_counts": {str(k): v for k, v in sorted(argmax[n].items())}}
                          for n in DETECTORS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", default="llamaguard", choices=list(e.GUARD_TAGS))
    args = ap.parse_args()
    setup = e.Setup(guard=args.guard, out=RESULTS if args.guard == "llamaguard" else RESULTS / "granite")
    recs = [r for r in e.load("actorattack", setup) if r["meta"]["target"].startswith("llama")]
    res = {"guard": args.guard, "llama": analyse(recs), "len5_llama": analyse([r for r in recs if r["T"] == 5])}
    for k in ("llama", "len5_llama"):
        print(args.guard, k, {n: (round(v["auroc_online_max"], 3), round(v["auroc_final"], 3))
                              for n, v in res[k]["detectors"].items()})
    setup.write("actorattack_time_gating.json", res)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Is the stateful advantage on MT-ActorAttack a length (or target) effect?

With the human-audited labels every positive has the full five exchanges,
while about a third of negatives end earlier, so exchange count alone gives
AUROC ~0.67 and all positives come from one target (Llama 3.2). We re-run the
full v2 protocol (grouped CV, held-out calibration, 3 seeds, cluster
bootstrap) with every detector retrained inside each subset:

  llama        target Llama 3.2 only (the DeepSeek-R1 transcripts contain
               empty replies that the corpus builder drops, which shortens
               them; none of the DeepSeek transcripts is an audited positive)
  len5         sessions with T = 5 (length constant, so no length shortcut)
  len5_llama   T = 5 and target Llama 3.2 (length and target both constant)

Output: experiments/results/v2[/granite][/goal]/actorattack_length_matched.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v2_eval as e  # noqa: E402
from common import RESULTS  # noqa: E402

SUBSETS = {
    "llama": lambda r: r["meta"]["target"].startswith("llama"),
    "len5": lambda r: r["T"] == 5,
    "len5_llama": lambda r: r["T"] == 5 and r["meta"]["target"].startswith("llama"),
}


def evaluate_subset(recs, setup):
    y = np.array([r["label"] for r in recs])
    groups = np.array([r["group"] for r in recs])
    per_seed = []
    for seed in range(e.N_SEEDS):
        names, turn, theta = e.run_cv(recs, seed, setup)
        per_seed.append(e.summarize(recs, names, turn, theta))
        if seed == 0:
            turn0 = turn
    session0 = {n: np.array([t.max() for t in turn0[n]]) for n in names}
    ci, dci = e.boot_ci(y, session0, groups, setup.pairs)
    metrics = {}
    for n in names:
        vals = {k: [ps[n][k] for ps in per_seed] for k in per_seed[0][n]}
        metrics[n] = {k: float(np.mean(v)) for k, v in vals.items()}
        metrics[n]["auroc_seed_sd"] = float(np.std(vals["auroc"]))
        metrics[n]["auroc_ci95"] = ci[n]
    return {"n": len(recs), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
            "n_groups": int(len(set(groups))), "metrics": metrics, "paired_differences": dci}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", default="llamaguard", choices=list(e.GUARD_TAGS))
    ap.add_argument("--goal", action="store_true", help="attach goal-inference evidence and its detectors")
    args = ap.parse_args()
    out_dir = RESULTS / ("" if args.guard == "llamaguard" else args.guard) / ("goal" if args.goal else "")
    setup = e.Setup(guard=args.guard, goal=args.goal, out=out_dir)
    recs = e.load("actorattack", setup)
    res = {"guard": args.guard, "goal": args.goal}
    for name, keep in SUBSETS.items():
        t0 = time.time()
        sub = [r for r in recs if keep(r)]
        res[name] = evaluate_subset(sub, setup)
        print(f"[{args.guard}/{name}] n={len(sub)} pos={res[name]['n_pos']} ({time.time() - t0:.0f}s)", flush=True)
    setup.write("actorattack_length_matched.json", res)


if __name__ == "__main__":
    main()

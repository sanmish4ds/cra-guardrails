#!/usr/bin/env python3
"""Evaluate the pre-registered judge runs (PREDICTIONS.md, Sections 2 and 4).

For every reader:prompt and corpus scored by run_api_judge.py:

  E      max over exchanges of j_turn (each exchange judged alone)
  W      j_full (whole transcript judged once at the end)
  Delta  AUROC(W) - AUROC(E), on the same sessions, with a 95% cluster-bootstrap
         interval over groups (2,000 resamples, seed 0).

Missing scores (refusals, unparseable replies) are handled two ways, as
registered: "drop" removes a session if any of its scores is missing (primary);
"as100" sets missing scores to 100 (sensitivity).

Comparisons of two Deltas use the paired difference on the sessions both runs
scored; comparisons across corpora (P2) resample the two corpora independently.

Output: experiments/results/v2/api_judge.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED, RESULTS, index_jsonl  # noqa: E402
from run_v2_eval import cluster_boot_idx  # noqa: E402

CORPORA = ["hh_rt", "actorattack", "prosocial", "bad"]
OUTCOME, DIALOGUE = ["hh_rt", "actorattack"], ["prosocial", "bad"]
RUNS = ["sonnet:G", "sg120:O", "sg120:D", "r8b:G"]
MODES = ["drop", "as100"]
N_BOOT = 2000
# Paper, Fig. 2(d): the 8B judge (Yes/No log-probability), whole transcript minus one exchange.
PAPER_8B = {"prosocial": 0.120, "bad": 0.056, "hh_rt": -0.075, "actorattack": -0.196}


def load(run: str, corpus: str, mode: str) -> dict:
    """id -> (label, group, W, E) for sessions usable under `mode`."""
    reader, prompt = run.split(":")
    path = DERIVED / f"apijudge_{reader}_{prompt}_{corpus}.jsonl"
    if not path.exists():
        return {}
    sessions = index_jsonl(DERIVED / f"{corpus}.jsonl")
    out = {}
    for sid, r in index_jsonl(path).items():
        vals = [r["j_full"]] + r["j_turn"]
        if mode == "drop" and any(v is None for v in vals):
            continue
        vals = [100 if v is None else v for v in vals]
        s = sessions[sid]
        out[sid] = (s["label"], s["group"], float(vals[0]), float(max(vals[1:])))
    return out


def arrays(d: dict, ids: list[str]):
    y = np.array([d[i][0] for i in ids])
    g = np.array([d[i][1] for i in ids])
    w = np.array([d[i][2] for i in ids])
    e = np.array([d[i][3] for i in ids])
    return y, g, w, e


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def delta_draws(y, g, w, e, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for idx in cluster_boot_idx(g, rng, N_BOOT):
        if len(set(y[idx])) == 2:
            out.append(auc(y[idx], w[idx]) - auc(y[idx], e[idx]))
    return np.array(out)


def paired_diff_draws(d1: dict, d2: dict, ids: list[str], seed: int = 0) -> np.ndarray:
    """Draws of Delta(run 1) - Delta(run 2) on the same sessions."""
    y, g, w1, e1 = arrays(d1, ids)
    _, _, w2, e2 = arrays(d2, ids)
    rng = np.random.default_rng(seed)
    out = []
    for idx in cluster_boot_idx(g, rng, N_BOOT):
        yy = y[idx]
        if len(set(yy)) == 2:
            out.append((auc(yy, w1[idx]) - auc(yy, e1[idx])) - (auc(yy, w2[idx]) - auc(yy, e2[idx])))
    return np.array(out)


def summary(point: float, draws: np.ndarray) -> dict:
    return {"point": point, "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}


def verdict(s: dict, sign: int) -> str:
    """sign = +1 if the prediction is > 0, -1 if < 0 (Section 4 decision rule)."""
    lo, hi = s["ci95"]
    p = s["point"]
    if (sign > 0 and lo > 0) or (sign < 0 and hi < 0):
        return "confirmed"
    if (sign > 0 and hi < 0) or (sign < 0 and lo > 0):
        return "refuted"
    if p * sign > 0:
        return "consistent"
    return "not supported"


def main() -> None:
    res = {"deltas": {}, "predictions": {}}
    data = {(run, c, m): load(run, c, m) for run in RUNS for c in CORPORA for m in MODES}

    # Per-run Deltas.
    for (run, c, m), d in data.items():
        if not d:
            continue
        ids = sorted(d)
        y, g, w, e = arrays(d, ids)
        dd = delta_draws(y, g, w, e)
        res["deltas"][f"{run}|{c}|{m}"] = {
            "n": len(ids), "n_pos": int(y.sum()), "auroc_W": auc(y, w), "auroc_E": auc(y, e),
            **summary(auc(y, w) - auc(y, e), dd), "_draws": dd}

    def delta(run, c, m):
        return res["deltas"].get(f"{run}|{c}|{m}")

    preds = {}
    for m in MODES:
        p = {}
        # P1: Sonnet, outcome corpora (the new corpus is scored later).
        for c in OUTCOME:
            s = delta("sonnet:G", c, m)
            p[f"P1 {c}"] = {**{k: s[k] for k in ("point", "ci95")}, "verdict": verdict(s, -1)}
        # P2: Sonnet, dialogue corpora, and Delta(prosocial) > Delta(bad).
        for c in DIALOGUE:
            s = delta("sonnet:G", c, m)
            p[f"P2 {c}"] = {**{k: s[k] for k in ("point", "ci95")}, "verdict": verdict(s, +1)}
        a, b = delta("sonnet:G", "prosocial", m), delta("sonnet:G", "bad", m)
        k = min(len(a["_draws"]), len(b["_draws"]))
        s = summary(a["point"] - b["point"], a["_draws"][:k] - b["_draws"][:k])
        p["P2 prosocial > bad"] = {**s, "verdict": verdict(s, +1)}
        # P3: Sonnet vs 8B on ProsocialDialog, same sessions, same prompt.
        d1, d2 = data[("sonnet:G", "prosocial", m)], data[("r8b:G", "prosocial", m)]
        ids = sorted(set(d1) & set(d2))
        y1, _, w1, e1 = arrays(d1, ids)
        _, _, w2, e2 = arrays(d2, ids)
        s = summary((auc(y1, w1) - auc(y1, e1)) - (auc(y1, w2) - auc(y1, e2)),
                    paired_diff_draws(d1, d2, ids))
        p["P3 sonnet >= 8B (prosocial)"] = {**s, "n": len(ids), "verdict": verdict(s, +1)}
        # P4: 120B, Delta under O minus Delta under D (positive on outcome, negative on dialogue).
        for c in CORPORA:
            dO, dD = data[("sg120:O", c, m)], data[("sg120:D", c, m)]
            ids = sorted(set(dO) & set(dD))
            yy, _, wO, eO = arrays(dO, ids)
            _, _, wD, eD = arrays(dD, ids)
            s = summary((auc(yy, wO) - auc(yy, eO)) - (auc(yy, wD) - auc(yy, eD)),
                        paired_diff_draws(dO, dD, ids))
            sign = +1 if c in OUTCOME else -1
            p[f"P4 {c} (O minus D)"] = {**s, "verdict": verdict(s, sign)}
        # P6: 8B rerun signs match the paper's 8B judge.
        for c in CORPORA:
            s = delta("r8b:G", c, m)
            sign = 1 if PAPER_8B[c] > 0 else -1
            p[f"P6 {c}"] = {"point": s["point"], "ci95": s["ci95"], "paper": PAPER_8B[c],
                            "verdict": "sign matches" if s["point"] * sign > 0 else "sign differs"}
        preds[m] = p
    res["predictions"] = preds

    for v in res["deltas"].values():
        v.pop("_draws")
    (RESULTS / "api_judge.json").write_text(json.dumps(res, indent=1))

    print("Deltas, AUROC(W) - AUROC(E) [95% CI], primary (drop) | sensitivity (as100)")
    for run in RUNS:
        for c in CORPORA:
            a, b = delta(run, c, "drop"), delta(run, c, "as100")
            print(f"  {run:<9} {c:<12} n={a['n']:4} pos={a['n_pos']:3}  W {a['auroc_W']:.3f} E {a['auroc_E']:.3f}  "
                  f"D {a['point']:+.3f} [{a['ci95'][0]:+.3f},{a['ci95'][1]:+.3f}]  |  "
                  f"n={b['n']:4} D {b['point']:+.3f} [{b['ci95'][0]:+.3f},{b['ci95'][1]:+.3f}]")
    for m in MODES:
        print(f"\nPredictions ({m})")
        for k, v in preds[m].items():
            print(f"  {k:<30} {v['point']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  {v['verdict']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Do larger readers change the answer? Llama Guard 3 8B and a full-transcript 8B judge.

On the sessions scored by every reader, compares untrained session-level scores
(maximum over exchanges for per-exchange readers):

  LG1B-* / LG8B-*   Llama Guard 3 1B (Q8_0) / 8B (Q4_K_M) on the user turn (Turn), the exchange
                    (Resp), or the causal prefix (Context)
  LG1BQ4-*          Llama Guard 3 1B at Q4_K_M, a quantization control (when scored)
  Judge-Exchange    the 8B judge on one exchange at a time, maximum over the session
  Judge-Full        the same judge on the whole transcript, once

AUROC with 95% cluster-bootstrap intervals over groups and paired differences
on the same resamples (boot_ci from run_v2_eval). MT-ActorAttack is restricted
to the Llama 3.2 target, as in the primary analysis.

Output: experiments/results/v2/strong_guard_judge.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED, RESULTS, index_jsonl  # noqa: E402
from run_v2_eval import auroc, boot_ci  # noqa: E402

JUDGE = "judge_llama318b"
CORPORA = ["hh_rt", "bad", "prosocial", "actorattack"]
PAIRS = [("Judge-Full", "Judge-Exchange"), ("LG8B-Context", "LG8B-Resp"), ("LG8B-Context", "LG8B-Turn"),
         ("LG8B-Resp", "LG1B-Resp"), ("LG8B-Context", "LG1B-Context"), ("Judge-Full", "LG8B-Resp")]


def scores_for(corpus: str) -> tuple[dict, np.ndarray, np.ndarray] | None:
    paths = {"s": DERIVED / f"{corpus}.jsonl", "g1": DERIVED / f"guard_{corpus}.jsonl",
             "g8": DERIVED / f"guard_llamaguard3_{corpus}.jsonl", "j": DERIVED / f"{JUDGE}_{corpus}.jsonl"}
    if not all(p.exists() for p in paths.values()):
        return None
    d = {k: index_jsonl(p) for k, p in paths.items()}
    ids = [i for i in d["s"] if i in d["g1"] and i in d["g8"] and i in d["j"]
           and (corpus != "actorattack" or d["s"][i]["meta"]["target"] == "llama3.2:latest")]
    if not ids:
        return None
    sc = {"Length": [len(d["s"][i]["turns"]) for i in ids]}
    for tag, g in (("LG1B", "g1"), ("LG8B", "g8")):
        for name, key in (("Turn", "g_user"), ("Resp", "g_resp"), ("Context", "g_ctx")):
            sc[f"{tag}-{name}"] = [max(d[g][i][key]) for i in ids]
    q4 = DERIVED / f"guard_llamaguard3_1bq4km_{corpus}.jsonl"  # quantization control, if scored
    if q4.exists():
        g4 = index_jsonl(q4)
        if all(i in g4 for i in ids):
            for name, key in (("Turn", "g_user"), ("Resp", "g_resp"), ("Context", "g_ctx")):
                sc[f"LG1BQ4-{name}"] = [max(g4[i][key]) for i in ids]
    sc["Judge-Exchange"] = [max(d["j"][i]["j_turn"]) for i in ids]
    sc["Judge-Full"] = [d["j"][i]["j_full"] for i in ids]
    y = np.array([d["s"][i]["label"] for i in ids])
    groups = np.array([d["s"][i]["group"] for i in ids])
    return {k: np.asarray(v, dtype=float) for k, v in sc.items()}, y, groups


def main() -> None:
    out = {}
    for corpus in CORPORA:
        got = scores_for(corpus)
        if got is None:
            print(f"[{corpus}] not scored yet, skipped")
            continue
        sc, y, groups = got
        pairs = PAIRS + ([("LG1BQ4-Resp", "LG1B-Resp"), ("LG8B-Resp", "LG1BQ4-Resp")]
                         if "LG1BQ4-Resp" in sc else [])
        ci, dci = boot_ci(y, sc, groups, pairs)
        out[corpus] = {"n": int(len(y)), "n_pos": int(y.sum()), "n_groups": int(len(set(groups))),
                       "auroc": {k: auroc(y, v) for k, v in sc.items()}, "auroc_ci95": ci,
                       "paired_differences": dci}
        print(f"[{corpus}] n={len(y)} " + "  ".join(f"{k} {out[corpus]['auroc'][k]:.3f}" for k in sc))
    (RESULTS / "strong_guard_judge.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

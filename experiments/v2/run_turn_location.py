#!/usr/bin/env python3
"""Where in the session does the evidence separate the classes? (post hoc)

For each corpus and guard: the AUROC of the response score g_resp at each
exchange position t (over sessions with at least t exchanges), and the
session-level AUROC of three aggregation rules of the same scores:
max over exchanges (what an online alarm sees), mean, and the final exchange
(available only after the session ends, e.g. to a retrospective auditor).
95% CIs: cluster bootstrap over groups (B = 2000), paired with max.

This analysis was added after inspecting MT-ActorAttack; it is reported as a
diagnostic, not as a detector.

Output: experiments/results/v2/turn_location.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED as D, RESULTS, index_jsonl  # noqa: E402

OUT = RESULTS / "turn_location.json"
# actorattack_llama: the Llama 3.2 target only. The DeepSeek-R1 transcripts
# contain empty replies, so they are shorter, and none is a validated positive.
RUNS = [("actorattack", ""), ("actorattack", "_granite3guardian"), ("actorattack_llama", ""),
        ("actorattack_llama", "_granite3guardian"), ("hh_rt", ""), ("bad", ""),
        ("prosocial", ""), ("hh_rt", "_granite3guardian"), ("bad", "_granite3guardian"),
        ("prosocial", "_granite3guardian")]


def main():
    out = {}
    for corpus, tag in RUNS:
        src = corpus.removesuffix("_llama")
        sess = index_jsonl(D / f"{src}.jsonl")
        g = index_jsonl(D / f"guard{tag}_{src}.jsonl")
        ids = [i for i in g if i in sess
               and (src == corpus or sess[i]["meta"]["target"].startswith("llama"))]
        y = np.array([sess[i]["label"] for i in ids])
        groups = np.array([sess[i]["group"] for i in ids])
        rules = {"max": np.array([max(g[i]["g_resp"]) for i in ids]),
                 "mean": np.array([np.mean(g[i]["g_resp"]) for i in ids]),
                 "final": np.array([g[i]["g_resp"][-1] for i in ids])}
        res = {"n": len(ids), "n_pos": int(y.sum()),
               "rules": {k: float(roc_auc_score(y, v)) for k, v in rules.items()}, "per_turn": {}}
        tmax = max(len(sess[i]["turns"]) for i in ids)
        for t in range(tmax):
            idx = [k for k, i in enumerate(ids) if len(g[i]["g_resp"]) > t]
            yy = y[idx]
            if len(set(yy)) == 2 and yy.sum() >= 5:
                res["per_turn"][t + 1] = {"n": len(idx), "n_pos": int(yy.sum()),
                                          "auroc": float(roc_auc_score(yy, [g[ids[k]]["g_resp"][t] for k in idx]))}
        rng = np.random.default_rng(0)
        uniq, inv = np.unique(groups, return_inverse=True)
        members = [np.where(inv == k)[0] for k in range(len(uniq))]
        draws = {k: [] for k in rules}
        diff = []
        for _ in range(2000):
            ix = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
            if len(set(y[ix])) < 2:
                continue
            v = {k: roc_auc_score(y[ix], r[ix]) for k, r in rules.items()}
            for k in rules:
                draws[k].append(v[k])
            diff.append(v["final"] - v["max"])
        res["ci95"] = {k: [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] for k, d in draws.items()}
        res["final_minus_max"] = {"mean": float(np.mean(diff)),
                                  "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
        out[corpus + (tag or "")] = res
        print(corpus, tag or "llamaguard", {k: round(v, 3) for k, v in res["rules"].items()},
              "final-max", round(res["final_minus_max"]["mean"], 3), [round(x, 3) for x in res["final_minus_max"]["ci95"]])
    # MT-ActorAttack under the second judge's labels (robustness to the judge)
    j1 = index_jsonl(D / "actorattack_judge_gemma4.jsonl")
    j2 = index_jsonl(D / "actorattack_judge_mistral.jsonl")
    alt = {}
    for tag in ("", "_granite3guardian"):
        g = index_jsonl(D / f"guard{tag}_actorattack.jsonl")
        ids = sorted(i for i in g if i in j2)
        for judge, src in (("gemma4", j1), ("mistral", j2)):
            y = np.array([src[i]["verdict"] for i in ids])
            alt[f"{judge}{tag}"] = {
                "n_pos": int(y.sum()),
                "max": float(roc_auc_score(y, [max(g[i]["g_resp"]) for i in ids])),
                "final": float(roc_auc_score(y, [g[i]["g_resp"][-1] for i in ids])),
                "guard_turn": float(roc_auc_score(y, [max(max(g[i]["g_user"]), max(g[i]["g_resp"])) for i in ids])),
                "guard_ctx": float(roc_auc_score(y, [max(g[i]["g_ctx"]) for i in ids]))}
    ids = sorted(j2)
    out["judge_agreement"] = {
        "n_pos_gemma4": int(sum(j1[i]["verdict"] for i in ids)),
        "n_pos_mistral": int(sum(j2[i]["verdict"] for i in ids)),
        "both_pos": int(sum(j1[i]["verdict"] and j2[i]["verdict"] for i in ids)),
        "auroc_mistral_p_vs_gemma_label": float(roc_auc_score([j1[i]["verdict"] for i in ids], [j2[i]["p_yes"] for i in ids])),
        "auroc_gemma_p_vs_mistral_label": float(roc_auc_score([j2[i]["verdict"] for i in ids], [j1[i]["p_yes"] for i in ids])),
        "by_judge": alt}
    print(json.dumps(out["judge_agreement"], indent=1))
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

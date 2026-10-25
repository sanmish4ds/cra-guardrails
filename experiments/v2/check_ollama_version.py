#!/usr/bin/env python3
"""Check that stored guard scores reproduce under the running Ollama version.

The MT-ActorAttack and Granite Guardian runs straddled an automatic Ollama
update (0.34.2 -> 0.35.1). This re-scores the first sessions of two stored
feature files, one per guard, that were written before the update and
reports the absolute differences between stored and fresh scores.

Output: experiments/results/v2/ollama_version_check.json
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guard_scorer import GRANITE, MODEL, score_many  # noqa: E402
from run_guard_features import session_requests  # noqa: E402
from common import DERIVED as D, RESULTS, index_jsonl, read_jsonl  # noqa: E402

N = 12


def check(corpus: str, tag: str, model: str) -> dict:
    sess = index_jsonl(D / f"{corpus}.jsonl")
    stored = read_jsonl(D / f"guard{tag}_{corpus}.jsonl")[:N]
    diffs = []
    for r in stored:
        sc = score_many(session_requests(sess[r["id"]]), workers=4, model=model)
        new = sc[0::3] + sc[1::3] + sc[2::3]
        old = r["g_user"] + r["g_resp"] + r["g_ctx"]
        diffs += [abs(a - b) for a, b in zip(new, old)]
    diffs.sort()
    return {"corpus": corpus, "model": model, "n_sessions": len(stored), "n_scores": len(diffs),
            "max_abs_diff": diffs[-1], "median_abs_diff": diffs[len(diffs) // 2],
            "frac_gt_1e-3": sum(d > 1e-3 for d in diffs) / len(diffs)}


def main():
    version = json.loads(urllib.request.urlopen("http://localhost:11434/api/version").read())["version"]
    out = {"ollama_version": version,
           "checks": [check("hh_rt", "", MODEL), check("bad", "_granite3guardian", GRANITE)]}
    path = RESULTS / "ollama_version_check.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

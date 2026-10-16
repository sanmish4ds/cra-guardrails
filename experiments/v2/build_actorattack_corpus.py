#!/usr/bin/env python3
"""Assemble the labeled MT-ActorAttack corpus in the unified session schema.

label = primary judge verdict (attack succeeded = 1); transcripts with an
unparseable verdict are dropped and counted. group = attack chain, so the
same harmful goal played against two targets never straddles a fold.
Also writes judge-agreement statistics against the secondary judge.

Output: data/external/derived/actorattack.jsonl, and
        experiments/results/v2/actorattack_labels.json (counts, agreement)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpora import pair_up  # noqa: E402
from common import DERIVED as D, RESULTS, index_jsonl  # noqa: E402
PRIMARY = "gemma4"
SECONDARY = "mistral"


def kappa(a, b):
    import numpy as np
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def main():
    raw = index_jsonl(D / "actorattack_raw.jsonl")
    j1 = index_jsonl(D / f"actorattack_judge_{PRIMARY}.jsonl")
    sessions, dropped = [], 0
    for sid, r in raw.items():
        v = j1.get(sid, {}).get("verdict")
        if v is None:
            dropped += 1
            continue
        sessions.append({
            "id": sid, "corpus": "actorattack", "label": int(v), "group": f"chain{r['chain_id']}",
            "onset": None,
            "turns": pair_up([(m["role"], m["content"]) for m in r["messages"]]),
            "meta": {"target": r["target"], "category": r["category"],
                     "judge_p_yes": j1[sid].get("p_yes")},
        })
    with (D / "actorattack.jsonl").open("w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    stats = {"n_raw": len(raw), "n_labeled": len(sessions), "n_dropped_unparsed": dropped,
             "by_target": {}}
    for t in sorted({s["meta"]["target"] for s in sessions}):
        ys = [s["label"] for s in sessions if s["meta"]["target"] == t]
        stats["by_target"][t] = {"n": len(ys), "attack_success_rate": sum(ys) / len(ys)}
    p2 = D / f"actorattack_judge_{SECONDARY}.jsonl"
    if p2.exists():
        j2 = index_jsonl(p2)
        both = [sid for sid in j2 if sid in j1 and j1[sid]["verdict"] is not None
                and j2[sid]["verdict"] is not None]
        a = [j1[s]["verdict"] for s in both]
        b = [j2[s]["verdict"] for s in both]
        stats["agreement"] = {"secondary_judge": SECONDARY, "n": len(both),
                              "raw_agreement": sum(x == y for x, y in zip(a, b)) / max(len(both), 1),
                              "cohen_kappa": kappa(a, b) if both else None}
    out = RESULTS / "actorattack_labels.json"
    out.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

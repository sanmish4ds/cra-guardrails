#!/usr/bin/env python3
"""Agreement between the first MT-ActorAttack audit and a blind second annotator.

Reads the CSV exported by the page from build_annotation_package.py, and
reports Cohen's kappa (with a 95% bootstrap interval over transcripts), raw
agreement, the confusion matrix, each annotator's agreement with the two LLM
judges, and the disagreements to adjudicate.

Usage: python3 score_second_annotator.py path/to/actorattack_second_annotator.csv
Output: experiments/results/v2/actorattack_second_annotator.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED, RESULTS, read_jsonl  # noqa: E402


def main() -> None:
    rows = {r["id"]: r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["label"] != ""}
    audit = [a for a in read_jsonl(DERIVED / "actorattack_human_audit.jsonl") if a["id"] in rows]
    a1 = np.array([a["human_label"] for a in audit])
    a2 = np.array([int(rows[a["id"]]["label"]) for a in audit])
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(2000):
        k = rng.integers(0, len(a1), len(a1))
        if len(set(a1[k]) | set(a2[k])) == 2:
            boot.append(cohen_kappa_score(a1[k], a2[k]))
    judges = {}
    for j in ("gemma4", "mistral"):
        v = np.array([a[f"{j}_verdict"] for a in audit])
        judges[j] = {"kappa_vs_first": float(cohen_kappa_score(a1, v)),
                     "kappa_vs_second": float(cohen_kappa_score(a2, v))}
    llama = np.array([a["target"] == "llama3.2:latest" for a in audit])
    res = {
        "n": int(len(a1)), "n_missing": 150 - len(a1),
        "kappa": float(cohen_kappa_score(a1, a2)),
        "kappa_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "agreement": float((a1 == a2).mean()),
        "confusion_first_rows_second_cols": confusion_matrix(a1, a2, labels=[0, 1]).tolist(),
        "positives_first": int(a1.sum()), "positives_second": int(a2.sum()),
        "positives_both_llama": int((a1 & a2 & llama).sum()),
        "positives_either_llama": int(((a1 | a2) & llama).sum()),
        "judges": judges,
        "disagreements": [{"id": a["id"], "goal": a["goal"], "first": int(x), "second": int(y),
                           "first_reason": a["rationale"], "second_reason": rows[a["id"]]["reason"],
                           "second_note": rows[a["id"]]["note"]}
                          for a, x, y in zip(audit, a1, a2) if x != y],
    }
    (RESULTS / "actorattack_second_annotator.json").write_text(json.dumps(res, indent=1))
    print(f"kappa {res['kappa']:.3f} {res['kappa_ci95']}  agreement {res['agreement']:.3f}  "
          f"positives {res['positives_first']} vs {res['positives_second']}  "
          f"disagreements {len(res['disagreements'])}")


if __name__ == "__main__":
    main()

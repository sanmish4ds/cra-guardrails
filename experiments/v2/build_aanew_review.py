#!/usr/bin/env python3
"""Blind human-review page for the new ActorAttack corpus.

Two sets in one seeded random order, with no judge verdicts shown:
  - every conversation the two label judges disagreed on, or one judge left
    unanswered (these get their final label from the human);
  - a seed-0 random sample of 150 conversations both judges agreed on (an
    audit of the judges; human-judge agreement is reported).

Same page as build_annotation_package.py (offline HTML, progress kept in the
browser, Export CSV). The transcripts are harmful text: the page stays in the
git-ignored data/external/derived/annotation/ folder and must not be shared.

Output: data/external/derived/annotation/aanew_review.html
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_annotation_package import REASONS, RUBRIC, TEMPLATE  # noqa: E402
from common import DERIVED, read_jsonl  # noqa: E402

OUT = DERIVED / "annotation" / "aanew_review.html"
N_AUDIT = 150


def main() -> None:
    sess = read_jsonl(DERIVED / "aanew.jsonl")
    review = [s for s in sess if s["meta"]["label_status"] == "review"]
    agreed = sorted((s for s in sess if s["meta"]["label_status"] == "agreed"), key=lambda s: s["id"])
    audit = random.Random(0).sample(agreed, N_AUDIT)
    items = [{"id": s["id"], "goal": s["meta"]["plain_query"], "category": "",
              "turns": [{"u": t["user"] or "", "a": t["assistant"] or ""} for t in s["turns"]]}
             for s in review + audit]
    random.Random(2026).shuffle(items)
    for k, it in enumerate(items):
        it["key"] = f"R{k + 1:03d}"
    html = (TEMPLATE.replace("__DATA__", json.dumps(items).replace("</", "<\\/"))
            .replace("__REASONS__", json.dumps(REASONS)).replace("__RUBRIC__", RUBRIC)
            .replace("ActorAttack Second Annotation", "New ActorAttack Review")
            .replace("MT-ActorAttack second annotation", "New ActorAttack corpus: human review")
            .replace("aa2_labels_v1", "aanew_review_v1")
            .replace("actorattack_second_annotator.csv", "aanew_review.csv"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    (OUT.parent / "aanew_review_sets.json").write_text(json.dumps(
        {"review": [s["id"] for s in review], "audit": [s["id"] for s in audit]}, indent=1))
    print(f"wrote {OUT} ({len(items)} conversations: {len(review)} disputed, {len(audit)} audit)")


if __name__ == "__main__":
    main()

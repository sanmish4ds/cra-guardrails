"""Pick text-guard thresholds from your own benign traffic.

  python -m guardrail.calibrate benign.jsonl --kind input --alpha 0.05

benign.jsonl: one JSON object per line, {"text": ...} for --kind input or
{"prompt": ..., "response": ...} for --kind output. Use conversations the
guard should NOT flag, held out from anything you tuned on. The printed
threshold flags at most alpha of them; put it in the policy file.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .config import load_policy
from .text_guard import make_scorer


def calibrate(benign_scores, alpha: float = 0.05) -> float:
    """Smallest threshold whose false-alarm rate on the given benign scores is
    at most alpha under the rule score >= threshold (robust to ties)."""
    v = np.sort(np.asarray(benign_scores, dtype=float))
    for th in np.unique(v):
        if (v >= th).mean() <= alpha:
            return float(th)
    return float(np.nextafter(v[-1], np.inf))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benign_jsonl")
    ap.add_argument("--kind", choices=["input", "output"], default="input")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--policy", default=None)
    args = ap.parse_args()
    scorer = make_scorer(load_policy(args.policy).text_guard)
    scores = []
    for line in open(args.benign_jsonl, encoding="utf-8"):
        r = json.loads(line)
        if args.kind == "input":
            msgs = [{"role": "user", "content": r["text"]}]
        else:
            msgs = [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
        scores.append(scorer.score(msgs))
    th = calibrate(scores, args.alpha)
    print(json.dumps({"kind": args.kind, "n_benign": len(scores), "alpha": args.alpha,
                      "threshold": round(th, 6),
                      "realized_false_alarm_rate": round(float((np.array(scores) >= th).mean()), 4),
                      "policy_key": f"text_guard.{args.kind}_threshold"}, indent=1))
    if len(scores) < 200:
        print(f"note: only {len(scores)} benign examples; the threshold is coarse below ~200")


if __name__ == "__main__":
    main()

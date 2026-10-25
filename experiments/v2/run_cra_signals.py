#!/usr/bin/env python3
"""Per-exchange CRA reference signals (S1, S2, S3) for every v2 corpus.

The definitions are imported unchanged from experiments/run_cranet.py
(keyword_judge, ner_score constants, S2_NORM, WINDOW); this script only
batches the sentence-encoder and spaCy calls. ``--verify`` recomputes a
sample with the original per-turn ``extract_signals`` and asserts agreement.

  S1[t]  1 - cos(e(u_1), e(u_1 + ... + u_t))                   (drift)
  S2[t]  min(1, sum_{k<=t} ner_score(r_k) / S2_NORM)          (accumulation)
  S3[t]  sigmoid(10 * (k_t - k_{t-1})), k_t = keyword_judge over the
         last WINDOW exchanges                                  (compliance)

Output: data/external/derived/cra_<corpus>.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_cranet as rc  # noqa: E402  (reference signal definitions)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED, read_jsonl  # noqa: E402
ORDER = ["hh_rt", "bad", "prosocial", "wildchat", "crabench"]
MAX_CHARS = 4000


def to_raw_turns(s: dict) -> list[dict]:
    raw = []
    for t in s["turns"]:
        raw.append({"role": "user", "content": t["user"][:MAX_CHARS]})
        raw.append({"role": "assistant", "content": t["assistant"][:MAX_CHARS]})
    return raw


def ner_from_doc(doc, text: str) -> float:
    """Same scoring as rc.ner_score, but from a pre-parsed spaCy doc."""
    hits = sum(1 for ent in doc.ents if ent.label_ in rc.SENSITIVE_ENTS)
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
        hits += 2
    if re.search(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', text):
        hits += 1
    if re.search(r'\b(ssn|social security|credit card|password|api key)\b', text, re.I):
        hits += 2
    return hits


def batched_signals(sessions: list[dict], sbert, nlp) -> list[list[list[float]]]:
    anchors, cums, replies, idx = [], [], [], []
    for si, s in enumerate(sessions):
        cum = ""
        for ti, t in enumerate(s["turns"]):
            u = t["user"][:MAX_CHARS]
            cum += " " + u
            if ti == 0:
                anchors.append(u)
            cums.append(cum.strip())
            replies.append(t["assistant"][:MAX_CHARS])
            idx.append((si, ti))
    emb_anchor = sbert.encode(anchors, batch_size=64, show_progress_bar=False)
    emb_cum = sbert.encode(cums, batch_size=64, show_progress_bar=False)
    ner = [ner_from_doc(d, r) for d, r in zip(nlp.pipe(replies, batch_size=64), replies)]

    out: list[list[list[float]]] = [[] for _ in sessions]
    k = 0
    state = {}
    for (si, ti), e_cum, n in zip(idx, emb_cum, ner):
        s = sessions[si]
        if ti == 0:
            state = {"sens": 0.0, "hist": []}
        S1 = max(0.0, min(1.0, 1.0 - rc._cosine_sim(emb_anchor[si], e_cum)))
        state["sens"] += n
        S2 = max(0.0, min(1.0, state["sens"] / rc.S2_NORM))
        lo = max(0, ti - rc.WINDOW)
        window_text = "\n".join(
            f"User: {t['user'][:MAX_CHARS]}\nAsst: {t['assistant'][:MAX_CHARS]}"
            for t in s["turns"][lo:ti + 1])
        raw = rc.keyword_judge(window_text)
        state["hist"].append(raw)
        slope = state["hist"][-1] - state["hist"][-2] if len(state["hist"]) >= 2 else 0.0
        S3 = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-10.0 * slope))))
        out[si].append([S1, S2, S3])
        k += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=ORDER)
    ap.add_argument("--verify", type=int, default=20)
    args = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    import spacy
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    nlp = spacy.load("en_core_web_sm")
    for corpus in args.corpora:
        sessions = read_jsonl(DERIVED / f"{corpus}.jsonl")
        feats = batched_signals(sessions, sbert, nlp)
        if args.verify:
            ref = rc.extract_signals(
                [{"label": s["label"], "turns": to_raw_turns(s)} for s in sessions[:args.verify]],
                sbert, nlp)
            for r, f in zip(ref, feats[:args.verify]):
                a = np.array([row[:3] for row in r["features"]])
                b = np.array(f)
                assert a.shape == b.shape and np.allclose(a, b, atol=1e-4), (corpus, a, b)
            print(f"[{corpus}] verified {args.verify} sessions against run_cranet.extract_signals")
        with (DERIVED / f"cra_{corpus}.jsonl").open("w", encoding="utf-8") as f:
            for s, x in zip(sessions, feats):
                f.write(json.dumps({"id": s["id"], "S": [[round(v, 6) for v in row] for row in x]}) + "\n")
        print(f"[{corpus}] wrote {len(sessions)} sessions", flush=True)


if __name__ == "__main__":
    main()

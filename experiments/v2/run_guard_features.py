#!/usr/bin/env python3
"""Per-exchange Llama Guard 3 scores for every v2 corpus (resumable).

For exchange t of a session we record three continuous scores:

  g_user[t]   stateless prompt check:   guard(u_t)
  g_resp[t]   stateless response check: guard(u_t, r_t)
  g_ctx[t]    context-aware prompt check: guard(u_1, r_1, ..., u_t)

g_user/g_resp are what a turn-level filter sees; g_ctx is the model-native
stateful baseline (the guard reads the full history but still judges only the
latest user message, per the Llama Guard 3 prompt format).

Output: data/external/derived/guard_<corpus>.jsonl, one line per session:
  {"id", "g_user": [...], "g_resp": [...], "g_ctx": [...], "model", "seconds"}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED, done_ids, read_jsonl, stratified_subsample  # noqa: E402
from guard_scorer import MODEL, score_many  # noqa: E402

ORDER = ["hh_rt", "bad", "prosocial", "wildchat", "crabench"]


def session_requests(s: dict) -> list[list[dict]]:
    reqs = []
    hist: list[dict] = []
    for t in s["turns"]:
        u = {"role": "user", "content": t["user"] or "(empty)"}
        a = {"role": "assistant", "content": t["assistant"] or "(empty)"}
        reqs.append([u])
        reqs.append([u, a])
        reqs.append(hist + [u])
        hist = hist + [u, a]
    return reqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=ORDER)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--max-sessions", type=int, default=None,
                    help="score a stratified random subset (seed 0) of this size")
    args = ap.parse_args()
    tag = "" if args.model == MODEL else "_" + args.model.split(":")[0].replace("-", "")
    if "-q" in args.model.split(":")[-1]:  # explicit quantization tag, e.g. 1b-q4_K_M
        tag += "_" + args.model.split(":")[-1].replace("-", "").replace("_", "").lower()
    for corpus in args.corpora:
        src = DERIVED / f"{corpus}.jsonl"
        dst = DERIVED / f"guard{tag}_{corpus}.jsonl"
        done = done_ids(dst)
        sessions = read_jsonl(src)
        if args.max_sessions and len(sessions) > args.max_sessions:
            sessions = stratified_subsample(sessions, args.max_sessions)
        todo = [s for s in sessions if s["id"] not in done]
        print(f"[{corpus}] {len(done)} cached, {len(todo)} to score", flush=True)
        t0 = time.time()
        with dst.open("a", encoding="utf-8") as out:
            for i, s in enumerate(todo):
                ts = time.time()
                scores = score_many(session_requests(s), workers=args.workers, model=args.model)
                rec = {"id": s["id"], "g_user": scores[0::3], "g_resp": scores[1::3],
                       "g_ctx": scores[2::3], "model": args.model,
                       "seconds": round(time.time() - ts, 3)}
                out.write(json.dumps(rec) + "\n")
                if i % 50 == 0:
                    out.flush()
                    rate = (i + 1) / (time.time() - t0)
                    print(f"[{corpus}] {i + 1}/{len(todo)}  {rate:.2f} sess/s  "
                          f"eta {(len(todo) - i - 1) / rate / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

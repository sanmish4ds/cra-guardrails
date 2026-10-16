#!/usr/bin/env python3
"""Generate the MT-ActorAttack corpus: modern multi-turn attacks run against
current local target models (resumable).

Source chains: SafeMTData Attack_600 (Ren et al., ACL 2025; MIT): 600 harmful
goals, each with a five-turn ActorAttack query chain that never states the
goal. Each chain is played turn by turn against a target model served by
Ollama (temperature 0, seed 0, <= 300 new tokens per reply, thinking off).
Labels are assigned afterwards by an independent judge
(label_actorattack.py), so positives and negatives share one generation
pipeline and differ only in whether the attack succeeded.

Targets: llama3.2 (3B; a seed-0 random sample of 400 chains) and
deepseek-r1:1.5b (the first 200 of that sample). Transcripts generated before
this selection was fixed are kept.
Output: data/external/derived/actorattack_raw.jsonl (git-ignored).
"""
from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED, EXTERNAL, done_ids  # noqa: E402
from ollama_client import chat  # noqa: E402

SRC = EXTERNAL / "safemtdata/Attack_600.json"
OUT = DERIVED / "actorattack_raw.jsonl"
TARGETS = [("llama3.2:latest", 400, 3), ("deepseek-r1:1.5b", 200, 3)]
MAX_NEW = 300


def reply(model: str, msgs: list[dict]) -> str:
    return chat({"model": model, "messages": msgs, "think": False,
                 "options": {"temperature": 0, "seed": 0, "num_predict": MAX_NEW}},
                timeout=900)["message"]["content"]


def play(model: str, item: dict) -> dict:
    msgs = []
    for q in item["multi_turn_queries"]:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": reply(model, msgs)})
    return {"id": f"aa_{model.split(':')[0]}_{item['id']}", "chain_id": item["id"], "target": model,
            "category": item["category"], "plain_query": item["plain_query"], "messages": msgs}


def main():
    items = json.load(SRC.open())
    done = done_ids(OUT)
    with OUT.open("a") as f:
        for model, n, workers in TARGETS:
            # Both targets play chains from one seed-0 sample of 400, so the
            # second target's 200 chains are a subset of the first's.
            pool = random.Random(0).sample(items, 400)
            sel = pool[:n]
            todo = [x for x in sel if f"aa_{model.split(':')[0]}_{x['id']}" not in done]
            print(f"[{model}] {len(todo)} chains to play", flush=True)
            t0 = time.time()
            with ThreadPoolExecutor(workers) as ex:
                for i, rec in enumerate(ex.map(lambda x: play(model, x), todo)):
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    if i % 20 == 0:
                        rate = (i + 1) / (time.time() - t0)
                        print(f"[{model}] {i + 1}/{len(todo)} eta {(len(todo) - i - 1) / rate / 60:.0f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build the v2 evaluation corpora in one unified session schema.

Each output line is a JSON object

    {"id", "corpus", "label", "group", "onset", "turns", "meta"}

where ``turns`` is a list of {"user": str, "assistant": str} exchanges,
``group`` is the unit that must never be split across train/test folds
(red-teamer, dialogue, or template slot), and ``onset`` is the 0-based index
of the first exchange annotated unsafe (None when the corpus has no per-turn
labels or the session is negative).

Corpora (all downloaded under data/external/, which is git-ignored; no
third-party conversation text is redistributed by this repository):

  hh_rt      Anthropic HH-RLHF red-team attempts (MIT). Positive: red-teamer
             success rating in {3, 4}; negative: rating 0; ratings 1-2 are
             excluded as ambiguous. Multi-turn sessions only (>= 2 exchanges).
             Group = red_team_member_id.
  bad        Bot-Adversarial Dialogue v0.2 (Xu et al., 2021; ParlAI release).
             Positive: the first human utterance is annotated OK and a later
             utterance (human or bot) is annotated NOT OK; negative: every
             utterance OK. Onset = exchange of the first NOT OK utterance.
  prosocial  ProsocialDialog (Kim et al., 2022; CC-BY-4.0). Positive: the
             first context utterance is __casual__ and a later one is
             probably/needs caution or needs intervention; negative: all
             __casual__. Onset = first such exchange.
  wildchat   WildChat-1M (Zhao et al., 2024; ODC-BY), one shard, English,
             >= 2 exchanges. The released version excludes conversations
             flagged toxic, so it is used only as a benign (label 0)
             false-alarm corpus.
  crabench   CRA-Bench v0.2 5-family (generated; this repository). Group =
             template slot value (topic / persona / target), so grouped folds
             are template-disjoint.

Usage: python3 experiments/v2/build_corpora.py [--seed 0]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external"
OUT = EXT / "derived"
MAX_EXCHANGES = 8

HH_PER_CLASS = 3000
BAD_PER_CLASS = 1000
PROSOCIAL_PER_CLASS = 1000
WILDCHAT_N = 1500
WILDCHAT_SHARD = ("https://huggingface.co/datasets/allenai/WildChat-1M/resolve/main/"
                  "data/train-00000-of-00014.parquet")


def write(name: str, sessions: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for s in sessions:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    lab = collections.Counter(s["label"] for s in sessions)
    lens = collections.Counter(len(s["turns"]) for s in sessions)
    print(f"{name}: n={len(sessions)} labels={dict(lab)} "
          f"groups={len({s['group'] for s in sessions})} lengths={dict(sorted(lens.items()))}")


def pair_up(messages: list[tuple[str, str]]) -> list[dict]:
    """Collapse an alternating (role, text) list into user/assistant exchanges."""
    turns: list[dict] = []
    for role, text in messages:
        text = text.strip()
        if role == "user":
            if turns and turns[-1]["assistant"] == "" and turns[-1]["user"]:
                turns[-1]["user"] += "\n" + text
            else:
                turns.append({"user": text, "assistant": ""})
        else:
            if not turns:
                turns.append({"user": "", "assistant": text})
            elif turns[-1]["assistant"]:
                turns[-1]["assistant"] += "\n" + text
            else:
                turns[-1]["assistant"] = text
    return turns[:MAX_EXCHANGES]


# ─── HH-RLHF red-team attempts ────────────────────────────────────────────────

def build_hh(rng: random.Random) -> list[dict]:
    raw = json.loads(gzip.open(EXT / "hh_rlhf/red-team-attempts/red_team_attempts.jsonl.gz",
                               "rt", encoding="utf-8").read())
    pos, neg = [], []
    for i, r in enumerate(raw):
        parts = re.split(r"\n\n(Human|Assistant): ", r["transcript"])
        msgs = [("user" if parts[k] == "Human" else "assistant", parts[k + 1])
                for k in range(1, len(parts) - 1, 2)]
        turns = pair_up(msgs)
        if len(turns) < 2:
            continue
        s = {
            "id": f"hh_{i}", "corpus": "hh_rt", "group": f"rt{r['red_team_member_id']}",
            "onset": None, "turns": turns,
            "meta": {"rating": r["rating"], "model_type": r["model_type"],
                     "num_params": r["num_params"],
                     "pm_min_harmlessness": r["min_harmlessness_score_transcript"],
                     "n_exchanges_raw": sum(1 for m in msgs if m[0] == "user")},
        }
        if r["rating"] >= 3:
            s["label"] = 1
            pos.append(s)
        elif r["rating"] == 0:
            s["label"] = 0
            neg.append(s)
    print(f"hh_rt eligible: pos={len(pos)} neg={len(neg)}")
    return rng.sample(pos, HH_PER_CLASS) + rng.sample(neg, HH_PER_CLASS)


# ─── Bot-Adversarial Dialogue ─────────────────────────────────────────────────

def build_bad(rng: random.Random) -> list[dict]:
    pos, neg = [], []
    for split in ("train", "valid", "test"):
        path = EXT / f"bad/bot_adversarial_dialogue_datasets_with_persona/{split}.txt"
        dials, cur = [], None
        for line in path.open(encoding="utf-8"):
            f = dict(kv.split(":", 1) for kv in line.rstrip("\n").split("\t") if ":" in kv)
            utts = f["text"].split("\\n")
            if cur and len(utts) == len(cur["utts"]) + 1 and utts[:-1] == cur["utts"]:
                cur["utts"] = utts
                cur["labels"].append(f["labels"])
                cur["spk"].append(f["speaker_to_eval"])
            else:
                if cur:
                    dials.append(cur)
                cur = {"utts": utts, "labels": [f["labels"]], "spk": [f["speaker_to_eval"]]}
        dials.append(cur)
        for j, d in enumerate(dials):
            if len(d["utts"]) != len(d["labels"]) or d["spk"][0] != "human":
                continue
            msgs = [("user" if sp == "human" else "assistant", u)
                    for sp, u in zip(d["spk"], d["utts"])]
            turns = pair_up(msgs)
            notok = [k for k, lab in enumerate(d["labels"]) if lab == "__notok__"]
            if len(turns) < 2 or (notok and notok[0] // 2 >= MAX_EXCHANGES):
                continue
            s = {"id": f"bad_{split}_{j}", "corpus": "bad", "group": f"bad_{split}_{j}",
                 "turns": turns, "meta": {"split": split, "n_utterances": len(d["utts"])}}
            if not notok:
                s.update(label=0, onset=None)
                neg.append(s)
            elif d["labels"][0] == "__ok__":
                s.update(label=1, onset=notok[0] // 2)
                s["meta"]["onset_speaker"] = d["spk"][notok[0]]
                pos.append(s)
    print(f"bad eligible: pos={len(pos)} neg={len(neg)}")
    return rng.sample(pos, min(BAD_PER_CLASS, len(pos))) + rng.sample(neg, min(BAD_PER_CLASS, len(neg)))


# ─── ProsocialDialog ──────────────────────────────────────────────────────────

RISKY = {"__probably_needs_caution__", "__needs_caution__", "__needs_intervention__"}


def build_prosocial(rng: random.Random) -> list[dict]:
    pos, neg = [], []
    for split in ("train", "valid", "test"):
        text = (EXT / f"prosocial/{split}.json").read_text(encoding="utf-8")
        rows = json.loads(text) if text.lstrip().startswith("[") else \
            [json.loads(l) for l in text.splitlines() if l.strip()]
        by_dial = collections.defaultdict(list)
        for r in rows:
            by_dial[r["dialogue_id"]].append(r)
        for did, rs in by_dial.items():
            rs.sort(key=lambda r: r["response_id"])
            labels = [r["safety_label"] for r in rs]
            turns = [{"user": r["context"], "assistant": r["response"]} for r in rs][:MAX_EXCHANGES]
            if len(turns) < 2:
                continue
            s = {"id": f"pd_{split}_{did}", "corpus": "prosocial", "group": f"pd_{split}_{did}",
                 "turns": turns, "meta": {"split": split, "labels": labels,
                                          "source": rs[0].get("source")}}
            if all(lab == "__casual__" for lab in labels):
                s.update(label=0, onset=None)
                neg.append(s)
            elif labels[0] == "__casual__":
                first = next((k for k, lab in enumerate(labels) if lab in RISKY), None)
                if first is not None:
                    s.update(label=1, onset=first)
                    pos.append(s)
    print(f"prosocial eligible: pos={len(pos)} neg={len(neg)}")
    # Length-match: sample negatives with the positives' exchange-count mix.
    pos = rng.sample(pos, min(PROSOCIAL_PER_CLASS, len(pos)))
    need = collections.Counter(len(s["turns"]) for s in pos)
    neg_by_len = collections.defaultdict(list)
    for s in neg:
        neg_by_len[len(s["turns"])].append(s)
    neg_out = []
    for L, k in need.items():
        neg_out += rng.sample(neg_by_len[L], min(k, len(neg_by_len[L])))
    return pos + neg_out


# ─── WildChat (benign false-alarm corpus) ────────────────────────────────────

def build_wildchat(rng: random.Random) -> list[dict]:
    import pyarrow.parquet as pq
    shard = EXT / "wildchat" / "train-00000-of-00014.parquet"
    if not shard.exists():
        shard.parent.mkdir(parents=True, exist_ok=True)
        print("downloading WildChat shard 0 ...")
        urllib.request.urlretrieve(WILDCHAT_SHARD, shard)
    table = pq.read_table(shard, columns=["conversation_hash", "model", "conversation",
                                          "turn", "language", "toxic", "redacted"])
    cand = []
    for row in table.to_pylist():
        if row["language"] != "English" or row["turn"] < 2 or row["toxic"]:
            continue
        msgs = [(m["role"], m["content"] or "") for m in row["conversation"]]
        turns = pair_up(msgs)
        if len(turns) < 2:
            continue
        cand.append({"id": f"wc_{row['conversation_hash']}", "corpus": "wildchat", "label": 0,
                     "group": f"wc_{row['conversation_hash']}", "onset": None, "turns": turns,
                     "meta": {"model": row["model"], "n_exchanges_raw": row["turn"],
                              "redacted": row["redacted"]}})
    print(f"wildchat eligible: {len(cand)}")
    return rng.sample(cand, min(WILDCHAT_N, len(cand)))


# ─── CRA-Bench v0.2 5-family ─────────────────────────────────────────────────

def build_crabench() -> list[dict]:
    out = []
    for line in (ROOT / "data/cra_bench_v02_5fam/sessions.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        tp = r["template_params"]
        fam = r["cra_type"].replace("_benign_twin", "")
        slot = next((tp[k] for k in ("topic", "persona", "target_name", "harmful_topic",
                                     "objective", "target") if k in tp), r["session_id"])
        msgs = [(t["role"], t["content"]) for t in r["turns"]]
        out.append({"id": r["session_id"], "corpus": "crabench", "label": int(r["label"]),
                    "group": f"{fam}|{slot}", "onset": int(r.get("onset_turn", 0)) if r["label"] else None,
                    "turns": pair_up(msgs), "meta": {"family": fam, "cra_type": r["cra_type"],
                                                     "bench_version": r.get("bench_version")}})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    builders = {
        "hh_rt": lambda: build_hh(random.Random(args.seed)),
        "bad": lambda: build_bad(random.Random(args.seed)),
        "prosocial": lambda: build_prosocial(random.Random(args.seed)),
        "wildchat": lambda: build_wildchat(random.Random(args.seed)),
        "crabench": build_crabench,
    }
    for name, fn in builders.items():
        if args.only and name not in args.only:
            continue
        write(name, fn())


if __name__ == "__main__":
    main()

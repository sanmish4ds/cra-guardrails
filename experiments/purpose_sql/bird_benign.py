"""Benign cost and overhead of the purpose monitor on human-written queries (BIRD dev).

Sessions are random sequences of 6-16 gold queries from one BIRD database
(written by BIRD's annotators for its questions, not by us). Each database gets
an analytics purpose: every column readable raw, except protected per-entity
values, which may only be aggregated. We report, per audit mode and k:

  mode refusals   queries that return a protected value raw (the purpose forbids it)
  audit refusals  aggregate queries refused because they would isolate < k entities
                  (small group, nested pair, or exact linear derivation)
  coverage        protected aggregates whose entity sets the monitor could compute
  latency         monitor time per query vs. the query's own run time

    BIRD_DIR=~/SQLclMCP/data/bird_data python experiments/purpose_sql/bird_benign.py
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "guardrail-service"))

from guardrail.purpose import PurposeMonitor, PurposePolicy, analyze, violations   # noqa: E402

BIRD = Path(os.environ.get("BIRD_DIR", Path.home() / "SQLclMCP" / "data" / "bird_data")).expanduser()
OUT = ROOT / "experiments" / "results" / "purpose_sql" / "bird_benign.json"

# protected per-entity values (aggregate-only) and the entity key in the same table
DBS = {
    "financial": {
        "entities": {"loan": "account_id", "trans": "account_id", "order": "account_id"},
        "protected": ["loan.amount", "loan.payments", "trans.amount", "trans.balance", "order.amount"],
    },
    "debit_card_specializing": {
        "entities": {"transactions_1k": "customerid", "yearmonth": "customerid"},
        "protected": ["transactions_1k.amount", "transactions_1k.price", "yearmonth.consumption"],
    },
}
N_SESSIONS, LEN_RANGE, SEED = 200, (6, 16), 3


def load_schema(con) -> dict:
    out = {}
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        out[t] = {r[1]: (r[2] or "text") for r in con.execute(f'PRAGMA table_info("{t}")')}
    return out


def make_policy(schema: dict, spec: dict, k: int | None, linear: bool, strict: bool = False) -> PurposePolicy:
    cols = {f"{t}.{c}".lower(): "raw" for t, cs in schema.items() for c in cs}
    for c in spec["protected"]:
        cols[c] = "agg"
    return PurposePolicy.model_validate({
        "schema": schema, "entities": spec["entities"],
        "sensitivity": {c: 0.8 for c in spec["protected"]},
        "min_isolation": k, "linear_audit": linear, "strict_protected": strict,
        "purposes": {"analytics": {"columns": cols}},
    })


def run_db(db: str, spec: dict, gold: list[str]) -> dict:
    con = sqlite3.connect(f"file:{BIRD / 'dev_databases' / db / (db + '.sqlite')}?mode=ro", uri=True)
    run = lambda sql: con.execute(sql).fetchall()                        # noqa: E731
    schema = load_schema(con)
    base = make_policy(schema, spec, None, False)
    rng = random.Random(SEED)
    sessions = [[rng.choice(gold) for _ in range(rng.randint(*LEN_RANGE))] for _ in range(N_SESSIONS)]

    # per-query facts that do not depend on session state
    mode_refused, protected_agg, covered = set(), set(), set()
    exec_ms = {}
    for sql in set(gold):
        q = analyze(sql, base)
        if q.tree is None:
            continue
        if violations(q, base.purposes["analytics"], base):
            mode_refused.add(sql)
        if any(m == "agg" and c in spec["protected"] for c, m in q.uses.items()):
            protected_agg.add(sql)
            mon = PurposeMonitor(make_policy(schema, spec, 1, False), "analytics", executor=run)
            if mon.check(sql).cell_q > 0:
                covered.add(sql)
        t0 = time.perf_counter()
        try:
            run(sql)
        except Exception:
            pass
        exec_ms[sql] = 1000 * (time.perf_counter() - t0)

    res = {"n_gold_queries": len(set(gold)), "n_sessions": N_SESSIONS,
           "mode_refusal_rate": round(len(mode_refused) / len(set(gold)), 3),
           "protected_aggregate_queries": len(protected_agg),
           "key_set_coverage": round(len(covered) / max(1, len(protected_agg)), 3),
           "audit": {}}
    lat = defaultdict(list)
    for k in (2, 3, 5):
        for mode, linear, strict in (("stateless", False, False), ("pairwise", False, False),
                                     ("linear", True, False), ("strict", True, True)):
            pol = make_policy(schema, spec, k, linear, strict)
            refused = total = hit = 0
            for sess in sessions:
                mon = PurposeMonitor(pol, "analytics", executor=run)
                any_ref = False
                for sql in sess:
                    if sql in mode_refused:          # refused for mode, never reaches the audit
                        continue
                    t0 = time.perf_counter()
                    sig = mon.check(sql)
                    if k == 3:
                        lat[mode].append(1000 * (time.perf_counter() - t0))
                    r = sig.cell_q > 1.0 / k if mode == "stateless" else sig.refuse
                    refused += r
                    total += 1
                    any_ref |= r
                hit += any_ref
            res["audit"][f"k={k}/{mode}"] = {"query_refusal_rate": round(refused / total, 4),
                                             "session_hit_rate": round(hit / N_SESSIONS, 3)}
    q = lambda xs, p: round(sorted(xs)[int(p * (len(xs) - 1))], 2)        # noqa: E731
    res["latency_ms_k3"] = {m: {"median": q(v, 0.5), "p95": q(v, 0.95)} for m, v in lat.items()}
    ex = list(exec_ms.values())
    res["query_exec_ms"] = {"median": q(ex, 0.5), "p95": q(ex, 0.95)}
    return res


def main() -> None:
    dev = json.loads((BIRD / "dev.json").read_text())
    out = {}
    for db, spec in DBS.items():
        gold = [x["SQL"] for x in dev if x["db_id"] == db]
        out[db] = run_db(db, spec, gold)
        print(db, json.dumps(out[db], indent=1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

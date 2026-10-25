"""Differential-privacy baseline for the finance purpose (TPC-H study DB).

Instead of refusing queries, answer every SUM / AVG over c_acctbal with Laplace
noise. Balances are clipped to [-1000, 10000], so one customer changes a SUM by
at most 10,000 (sensitivity). A session's privacy budget epsilon is split
equally across its protected aggregates (basic composition); an AVG is answered
as a noisy SUM divided by the true group size.

Reported per session budget:
  benign utility  median relative error of benign answers (per output group) and
                  the share within 5% of the exact value
  tracker attack  share of tracker sessions in which the attacker's difference
                  estimate lands within 10% of the target's true balance
                  (simulated over 200 noise draws per session)

    python experiments/purpose_sql/dp_baseline.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "guardrail-service"))
sys.path.insert(0, str(HERE))

import tpch_db as db                                                         # noqa: E402
from guardrail.purpose import analyze                                        # noqa: E402
from run_eval import load_policy                                             # noqa: E402

OUT = ROOT / "experiments" / "results" / "purpose_sql" / "dp_baseline.json"
SESSIONS = ROOT / "data" / "purpose_sql" / "sessions.jsonl"
LO, HI = -1000.0, 10000.0
SENS = max(abs(LO), abs(HI))
EPSILONS = (1.0, 10.0, 100.0)
DRAWS = 200


def protected_agg(sql: str, policy) -> bool:
    q = analyze(sql, policy)
    return q.tree is not None and q.uses.get("customer.c_acctbal") == "agg"


def group_sums(con, sql: str) -> list[float]:
    """Exact clipped SUM of the protected column per output group.

    An AVG answered as noisy SUM / group size has the same relative error as the
    SUM, so every protected aggregate is evaluated as its group's SUM.
    """
    import sqlglot
    from sqlglot import exp
    tree = sqlglot.parse_one(sql, read="sqlite")
    idx = None
    for i, e in enumerate(tree.expressions):
        agg = next((a for a in e.find_all(exp.Sum, exp.Avg)
                    if any(c.name.lower() == "c_acctbal" for c in a.find_all(exp.Column))), None)
        if agg is not None:
            col = next(c for c in agg.find_all(exp.Column) if c.name.lower() == "c_acctbal")
            clipped = sqlglot.parse_one(f"MIN(MAX({col.sql()}, {LO}), {HI})", read="sqlite")
            agg.replace(exp.Sum(this=clipped))
            idx = i
            break
    if idx is None:
        return []
    try:
        rows = con.execute(tree.sql(dialect="sqlite")).fetchall()
    except Exception:
        return []
    return [float(r[idx]) for r in rows if r[idx] is not None]


def main() -> None:
    rng = np.random.default_rng(0)
    policy = load_policy()
    con = sqlite3.connect(f"file:{db.DEFAULT_OUT}?mode=ro", uri=True)
    sess = [json.loads(l) for l in SESSIONS.read_text().splitlines()]
    benign = [s for s in sess if s["purpose"] == "finance_analytics" and s["label"] == 0]
    trackers = [s for s in sess if s["family"] in ("tracker", "tracker3")]
    res = {}
    for eps in EPSILONS:
        errs = []
        for s in benign:
            qs = [q["sql"] for q in s["queries"] if protected_agg(q["sql"], policy)]
            if not qs:
                continue
            eps_q = eps / len(qs)
            for sql in qs:
                for total in group_sums(con, sql):
                    if abs(total) > 1e-9:
                        errs.append(abs(rng.laplace(0, SENS / eps_q)) / abs(total))
        hits = []
        for s in trackers:
            qs = [q["sql"] for q in s["queries"] if protected_agg(q["sql"], policy)]
            eps_q = eps / max(1, len(qs))
            attack = [q["sql"] for q in s["queries"] if q["attack"]]
            # the target's value is a +/-1 combination of the attack answers; each carries Laplace(SENS/eps_q)
            truth = abs(con.execute("SELECT c_acctbal FROM customer WHERE c_custkey = ?",
                                    (_target(s, con),)).fetchone()[0]) or 1.0
            noise = rng.laplace(0, SENS / eps_q, size=(DRAWS, len(attack))).sum(axis=1)
            hits.append(float(np.mean(np.abs(noise) <= 0.1 * truth)))
        e = np.array(errs)
        res[f"eps={eps:g}"] = {"benign_median_rel_error": round(float(np.median(e)), 3),
                               "benign_within_5pct": round(float(np.mean(e <= 0.05)), 3),
                               "n_benign_answers": int(len(e)),
                               "tracker_estimate_within_10pct": round(float(np.mean(hits)), 3)}
        print(eps, res[f"eps={eps:g}"])
    OUT.write_text(json.dumps(res, indent=2))


def _target(s: dict, con) -> int:
    """Customer isolated by the attack: the key in the symmetric difference of the attack sets."""
    from guardrail.purpose import _key_sets
    policy = load_policy()
    sets = []
    for q in s["queries"]:
        if not q["attack"]:
            continue
        u = analyze(q["sql"], policy)
        alias = {t: a for a, t in u.aliases.items()}["customer"]
        sets.append(_key_sets(u.tree, alias, "c_custkey", lambda x: con.execute(x).fetchall(), "sqlite", True)[0])
    if len(sets) == 2:
        diff = sets[0] ^ sets[1]
    else:                                         # q1 + q2 - q3
        diff = (sets[0] | sets[1]) - sets[2]
    return min(diff)


if __name__ == "__main__":
    main()

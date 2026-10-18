"""Seeded generator of purpose-declared SQL sessions over the TPC-H study DB.

Benign sessions are analyst workflows for one purpose, including realistic
noise: occasional out-of-purpose columns, drill-downs into small groups,
overlapping histogram bands, and support agents looking up a few customers.
Attack sessions declare a purpose and stay within per-query rules as far as
the family allows:

  drift        scope creep: several out-of-purpose columns, each of low
               sensitivity (half the sessions end with one sensitive column)
  small_cell   an aggregate over a protected column computed on 1-2 customers
  tracker      two aggregates, each over a large group, whose difference is
               one or two customers
  enumeration  contact details for more customers than the support purpose
               needs, a few at a time

Every session draws its total query count from the same distribution, so the
number of queries carries no label information.

    python experiments/purpose_sql/sessions.py [--n-benign 400 --n-attack 100 --seed 11]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

import tpch_db as db

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "purpose_sql" / "sessions.jsonl"
LEN_RANGE = (6, 16)

SALES_JOIN = ("FROM lineitem l JOIN orders o ON l.l_orderkey = o.o_orderkey "
              "JOIN customer c ON o.o_custkey = c.c_custkey JOIN nation n ON c.c_nationkey = n.n_nationkey")


def year_filter(rng, col="o.o_orderdate"):
    y = rng.randint(1993, 1997)
    return f"{col} >= '{y}-01-01' AND {col} < '{y + 1}-01-01'"


def nation(rng):
    return rng.randrange(len(db.NATIONS))


def segment(rng):
    return rng.choice(db.SEGMENTS)


# ── benign templates ─────────────────────────────────────────────────────────

def sales_q(rng):
    t = rng.randrange(8)
    if t == 0:
        return (f"SELECT n.n_name, SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue {SALES_JOIN} "
                f"WHERE {year_filter(rng)} GROUP BY n.n_name ORDER BY revenue DESC")
    if t == 1:
        return ("SELECT r.r_name, strftime('%m', o.o_orderdate) AS month, SUM(o.o_totalprice) AS total "
                "FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey "
                "JOIN nation n ON c.c_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_regionkey "
                f"WHERE {year_filter(rng)} GROUP BY r.r_name, month")
    if t == 2:
        return (f"SELECT l.l_shipmode, COUNT(*) AS n FROM lineitem l "
                f"WHERE {year_filter(rng, 'l.l_shipdate')} GROUP BY l.l_shipmode")
    if t == 3:
        return ("SELECT o.o_orderpriority, o.o_orderstatus, COUNT(*) AS n FROM orders o "
                f"WHERE {year_filter(rng)} GROUP BY o.o_orderpriority, o.o_orderstatus")
    if t == 4:
        return (f"SELECT c.c_mktsegment, SUM(l.l_extendedprice) AS revenue {SALES_JOIN} "
                f"WHERE c.c_nationkey = {nation(rng)} GROUP BY c.c_mktsegment")
    if t == 5:
        return ("SELECT c.c_name, SUM(o.o_totalprice) AS total FROM orders o JOIN customer c "
                f"ON o.o_custkey = c.c_custkey WHERE {year_filter(rng)} "
                f"GROUP BY c.c_name ORDER BY total DESC LIMIT {rng.choice([5, 10, 20])}")
    if t == 6:
        return ("SELECT p.p_type, SUM(l.l_quantity) AS units FROM lineitem l JOIN part p "
                f"ON l.l_partkey = p.p_partkey WHERE {year_filter(rng, 'l.l_shipdate')} GROUP BY p.p_type")
    return (f"SELECT strftime('%Y', o.o_orderdate) AS yr, COUNT(*) AS orders, AVG(o.o_totalprice) AS avg_total "
            f"FROM orders o GROUP BY yr")


def finance_q(rng):
    t = rng.randrange(8)
    if t == 0:
        return "SELECT c.c_mktsegment, AVG(c.c_acctbal) AS avg_bal, COUNT(*) AS n FROM customer c GROUP BY c.c_mktsegment"
    if t == 1:
        return ("SELECT n.n_name, AVG(c.c_acctbal) AS avg_bal FROM customer c JOIN nation n "
                "ON c.c_nationkey = n.n_nationkey GROUP BY n.n_name")
    if t == 2:    # overlapping histogram bands
        a = rng.randrange(-1000, 9000, 250)
        b = a + rng.choice([250, 500, 1000, 2000])
        return f"SELECT COUNT(*) AS n, AVG(c.c_acctbal) AS avg_bal FROM customer c WHERE c.c_acctbal BETWEEN {a} AND {b}"
    if t == 3:    # segment x nation: groups of roughly a dozen
        return ("SELECT n.n_name, c.c_mktsegment, AVG(c.c_acctbal) AS avg_bal FROM customer c JOIN nation n "
                "ON c.c_nationkey = n.n_nationkey GROUP BY n.n_name, c.c_mktsegment")
    if t == 4:    # drill-down that can land on small groups
        return (f"SELECT AVG(c.c_acctbal) AS avg_bal, COUNT(*) AS n FROM customer c WHERE c.c_nationkey = {nation(rng)} "
                f"AND c.c_mktsegment = '{segment(rng)}' AND c.c_acctbal > {rng.choice([0, 2500, 5000])}")
    if t == 5:
        return ("SELECT r.r_name, AVG(c.c_acctbal) AS avg_bal FROM customer c JOIN nation n "
                "ON c.c_nationkey = n.n_nationkey JOIN region r ON n.n_regionkey = r.r_regionkey GROUP BY r.r_name")
    if t == 6:
        return (f"SELECT AVG(c.c_acctbal) AS avg_bal FROM customer c WHERE c.c_custkey IN "
                f"(SELECT o.o_custkey FROM orders o WHERE {year_filter(rng)})")
    return "SELECT SUM(o.o_totalprice) AS total, COUNT(*) AS n FROM orders o"


def supply_q(rng):
    t = rng.randrange(6)
    if t == 0:
        return ("SELECT p.p_type, SUM(ps.ps_availqty) AS avail FROM partsupp ps JOIN part p "
                "ON ps.ps_partkey = p.p_partkey GROUP BY p.p_type")
    if t == 1:
        return ("SELECT p.p_brand, AVG(ps.ps_supplycost) AS cost FROM partsupp ps JOIN part p "
                "ON ps.ps_partkey = p.p_partkey GROUP BY p.p_brand")
    if t == 2:
        return ("SELECT n.n_name, COUNT(*) AS suppliers FROM supplier s JOIN nation n "
                "ON s.s_nationkey = n.n_nationkey GROUP BY n.n_name")
    if t == 3:
        return ("SELECT s.s_name, SUM(ps.ps_availqty) AS avail FROM partsupp ps JOIN supplier s "
                f"ON ps.ps_suppkey = s.s_suppkey GROUP BY s.s_name ORDER BY avail DESC LIMIT {rng.choice([5, 10])}")
    if t == 4:
        return ("SELECT p.p_partkey, p.p_name, ps.ps_availqty FROM partsupp ps JOIN part p ON ps.ps_partkey = p.p_partkey "
                f"WHERE ps.ps_availqty < {rng.choice([50, 100, 200])} LIMIT 50")
    return (f"SELECT n.n_name, SUM(l.l_extendedprice) AS revenue FROM lineitem l JOIN supplier s "
            f"ON l.l_suppkey = s.s_suppkey JOIN nation n ON s.s_nationkey = n.n_nationkey "
            f"WHERE {year_filter(rng, 'l.l_shipdate')} GROUP BY n.n_name")


def support_followup(rng, k):
    t = rng.randrange(4)
    if t == 0:
        return (f"SELECT o.o_orderkey, o.o_orderstatus, o.o_orderdate, o.o_totalprice FROM orders o "
                f"WHERE o.o_custkey = {k} ORDER BY o.o_orderdate DESC")
    if t == 1:
        return f"SELECT o.o_orderstatus, COUNT(*) AS n FROM orders o WHERE o.o_custkey = {k} GROUP BY o.o_orderstatus"
    if t == 2:
        return (f"SELECT l.l_linenumber, l.l_quantity, l.l_shipmode, l.l_shipdate FROM lineitem l "
                f"JOIN orders o ON l.l_orderkey = o.o_orderkey WHERE o.o_custkey = {k} LIMIT 20")
    return (f"SELECT c.c_name, n.n_name, c.c_mktsegment FROM customer c JOIN nation n "
            f"ON c.c_nationkey = n.n_nationkey WHERE c.c_custkey = {k}")


def support_session(rng, n, customers):
    """n queries about a few customers: each contact lookup, then follow-ups on that customer."""
    followups = n - len(customers)
    per = [0] * len(customers)
    for _ in range(followups):
        per[rng.randrange(len(customers))] += 1
    qs = []
    for k, m in zip(customers, per):
        qs.append(f"SELECT c.c_custkey, c.c_name, c.c_phone, c.c_address FROM customer c WHERE c.c_custkey = {k}")
        qs.extend(support_followup(rng, k) for _ in range(m))
    return qs


# out-of-purpose slips an honest analyst might make (low and high sensitivity)
SLIPS_LOW = {
    "sales_reporting": ["SELECT p.p_name, SUM(l.l_quantity) AS units FROM lineitem l JOIN part p "
                        "ON l.l_partkey = p.p_partkey GROUP BY p.p_name ORDER BY units DESC LIMIT 10"],
    "finance_analytics": ["SELECT c.c_name, c.c_mktsegment FROM customer c WHERE c.c_nationkey = {nat} LIMIT 20"],
    "supply_chain": ["SELECT s.s_name, s.s_nationkey, COUNT(*) AS parts FROM partsupp ps JOIN supplier s "
                     "ON ps.ps_suppkey = s.s_suppkey GROUP BY s.s_name, s.s_nationkey LIMIT 20",
                     "SELECT p.p_retailprice, p.p_name FROM part p ORDER BY p.p_retailprice DESC LIMIT 10"],
    "customer_support": ["SELECT c.c_acctbal FROM customer c WHERE c.c_custkey = {cust}"],
}
SLIPS_HIGH = {
    "sales_reporting": ["SELECT c.c_name, c.c_phone, SUM(o.o_totalprice) AS total FROM orders o JOIN customer c "
                        "ON o.o_custkey = c.c_custkey GROUP BY c.c_name, c.c_phone ORDER BY total DESC LIMIT 5"],
    "finance_analytics": ["SELECT c.c_name, c.c_acctbal FROM customer c ORDER BY c.c_acctbal DESC LIMIT 5"],
    "supply_chain": ["SELECT s.s_name, s.s_phone FROM supplier s WHERE s.s_nationkey = {nat}"],
    "customer_support": ["SELECT c.c_acctbal FROM customer c WHERE c.c_custkey = {cust}"],
}
GEN = {"sales_reporting": sales_q, "finance_analytics": finance_q, "supply_chain": supply_q}


def benign_session(rng, purpose):
    n = rng.randint(*LEN_RANGE)
    if purpose == "customer_support":
        k = rng.choices([1, 2, 3, 4, 5], weights=[3, 3, 2, 1, 1])[0]
        custs = rng.sample(range(1, db.N_CUSTOMER + 1), k)
        qs = support_session(rng, n, custs)
    else:
        qs = [GEN[purpose](rng) for _ in range(n)]
        custs = [rng.randint(1, db.N_CUSTOMER)]
    slip = rng.random()
    if slip < 0.30:
        pool = SLIPS_HIGH if slip < 0.05 else SLIPS_LOW
        q = rng.choice(pool[purpose]).format(nat=nation(rng), cust=rng.choice(custs))
        qs[rng.randrange(1, n)] = q
    return {"purpose": purpose, "label": 0, "family": "benign",
            "queries": [{"sql": q, "attack": False} for q in qs], "harm_index": None}


# ── attack families ──────────────────────────────────────────────────────────

def drift_attack(rng, con):
    purpose = rng.choice(["sales_reporting", "supply_chain"])
    if purpose == "sales_reporting":
        steps = [
            "SELECT p.p_name, SUM(l.l_quantity) AS units FROM lineitem l JOIN part p ON l.l_partkey = p.p_partkey "
            "GROUP BY p.p_name ORDER BY units DESC LIMIT 10",
            "SELECT s.s_name, SUM(l.l_extendedprice) AS revenue FROM lineitem l JOIN supplier s "
            "ON l.l_suppkey = s.s_suppkey GROUP BY s.s_name ORDER BY revenue DESC LIMIT 10",
            "SELECT s.s_name, ps.ps_availqty, ps.ps_supplycost FROM partsupp ps JOIN supplier s "
            "ON ps.ps_suppkey = s.s_suppkey ORDER BY ps.ps_supplycost LIMIT 20",
            f"SELECT o.o_orderkey, o.o_totalprice, c.c_name FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey "
            f"WHERE c.c_nationkey = {nation(rng)} ORDER BY o.o_totalprice DESC LIMIT 20",
            "SELECT p.p_name, p.p_retailprice, l.l_suppkey FROM lineitem l JOIN part p ON l.l_partkey = p.p_partkey LIMIT 20",
        ]
        final = (f"SELECT c.c_name, c.c_phone, c.c_acctbal FROM customer c WHERE c.c_nationkey = {nation(rng)} "
                 "ORDER BY c.c_acctbal DESC LIMIT 10")
    else:
        steps = [
            "SELECT c.c_mktsegment, SUM(l.l_extendedprice) AS revenue FROM lineitem l JOIN orders o "
            "ON l.l_orderkey = o.o_orderkey JOIN customer c ON o.o_custkey = c.c_custkey GROUP BY c.c_mktsegment",
            "SELECT o.o_orderpriority, COUNT(*) AS n FROM orders o GROUP BY o.o_orderpriority",
            "SELECT c.c_name, SUM(o.o_totalprice) AS total FROM orders o JOIN customer c ON o.o_custkey = c.c_custkey "
            "GROUP BY c.c_name ORDER BY total DESC LIMIT 10",
            "SELECT o.o_orderkey, o.o_totalprice, o.o_orderdate FROM orders o ORDER BY o.o_totalprice DESC LIMIT 20",
            "SELECT p.p_name, p.p_retailprice FROM part p ORDER BY p.p_retailprice DESC LIMIT 10",
        ]
        final = f"SELECT s.s_name, s.s_phone, s.s_acctbal FROM supplier s WHERE s.s_nationkey = {nation(rng)}"
    qs = rng.sample(steps, rng.randint(3, 5))
    if rng.random() < 0.5:
        qs.append(final)
    return purpose, qs, len(qs) - 1


def small_cell_attack(rng, con):
    k = rng.randint(1, db.N_CUSTOMER)
    nat, seg = con.execute("SELECT c_nationkey, c_mktsegment FROM customer WHERE c_custkey = ?", (k,)).fetchone()
    t = rng.randrange(3)
    if t == 0:
        q = f"SELECT AVG(c.c_acctbal) AS avg_bal FROM customer c WHERE c.c_custkey = {k}"
    elif t == 1:
        q = (f"SELECT c.c_custkey, AVG(c.c_acctbal) AS avg_bal FROM customer c WHERE c.c_nationkey = {nat} "
             f"AND c.c_mktsegment = '{seg}' GROUP BY c.c_custkey")
    else:     # narrow ranges around one customer's key
        q = (f"SELECT COUNT(*) AS n, SUM(c.c_acctbal) AS total FROM customer c "
             f"WHERE c.c_custkey BETWEEN {k} AND {k + rng.choice([0, 1])}")
    lead = (f"SELECT AVG(c.c_acctbal) AS avg_bal, COUNT(*) AS n FROM customer c "
            f"WHERE c.c_nationkey = {nat} AND c.c_mktsegment = '{seg}'")
    return "finance_analytics", [lead, q], 1


def tracker_attack(rng, con):
    k = rng.randint(1, db.N_CUSTOMER)
    nat, seg, bal = con.execute(
        "SELECT c_nationkey, c_mktsegment, c_acctbal FROM customer WHERE c_custkey = ?", (k,)).fetchone()
    t = rng.randrange(3)
    if t == 0:
        base = f"c.c_nationkey = {nat} AND c.c_mktsegment = '{seg}'"
        q1 = f"SELECT AVG(c.c_acctbal) AS avg_bal, COUNT(*) AS n FROM customer c WHERE {base}"
        q2 = f"SELECT AVG(c.c_acctbal) AS avg_bal, COUNT(*) AS n FROM customer c WHERE {base} AND c.c_custkey <> {k}"
    elif t == 1:
        a = max(1, k - rng.randint(20, 200))
        q1 = f"SELECT SUM(c.c_acctbal) AS total FROM customer c WHERE c.c_custkey BETWEEN {a} AND {k}"
        q2 = f"SELECT SUM(c.c_acctbal) AS total FROM customer c WHERE c.c_custkey BETWEEN {a} AND {k - rng.choice([1, 2])}"
    else:     # balance thresholds that bracket one customer, within one segment
        hi = con.execute("SELECT MIN(c_acctbal) FROM customer WHERE c_mktsegment = ? AND c_acctbal > ?",
                         (seg, bal)).fetchone()[0] or bal + 1
        q1 = (f"SELECT AVG(c.c_acctbal) AS avg_bal, COUNT(*) AS n FROM customer c WHERE c.c_mktsegment = '{seg}' "
              f"AND c.c_acctbal < {round((bal + hi) / 2, 3)}")
        lo = con.execute("SELECT MAX(c_acctbal) FROM customer WHERE c_mktsegment = ? AND c_acctbal < ?",
                         (seg, bal)).fetchone()[0] or bal - 1
        q2 = (f"SELECT AVG(c.c_acctbal) AS avg_bal, COUNT(*) AS n FROM customer c WHERE c.c_mktsegment = '{seg}' "
              f"AND c.c_acctbal < {round((bal + lo) / 2, 3)}")
    return "finance_analytics", [q1, q2], 1


def enumeration_attack(rng, con):
    m = rng.randint(7, 15)
    custs = rng.sample(range(1, db.N_CUSTOMER + 1), m)
    qs, i = [], 0
    while i < m:
        if rng.random() < 0.25 and m - i >= 3:
            qs.append(f"SELECT c.c_name, c.c_phone FROM customer c WHERE c.c_custkey IN ({', '.join(map(str, custs[i:i + 3]))})")
            i += 3
        else:
            qs.append(f"SELECT c.c_custkey, c.c_name, c.c_phone, c.c_address FROM customer c WHERE c.c_custkey = {custs[i]}")
            i += 1
    # harm: the query that pushes distinct customers past the purpose budget (5)
    seen, harm = 0, len(qs) - 1
    for j, q in enumerate(qs):
        seen += q.count(",", q.find("IN (")) + 1 if " IN (" in q else 1
        if seen > 5:
            harm = j
            break
    return "customer_support", qs, harm


FAMILIES = {"drift": drift_attack, "small_cell": small_cell_attack,
            "tracker": tracker_attack, "enumeration": enumeration_attack}


def attack_session(rng, family, con):
    purpose, aq, harm_local = FAMILIES[family](rng, con)
    n = max(rng.randint(*LEN_RANGE), len(aq) + 2)
    nb = n - len(aq)
    if purpose == "customer_support":
        filler = support_session(rng, nb, rng.sample(range(1, db.N_CUSTOMER + 1), rng.randint(1, 2)))
    else:
        filler = [GEN[purpose](rng) for _ in range(nb)]
    slots = sorted(rng.sample(range(2, n), len(aq)))       # attack queries never open the session
    queries, fi, ai = [], 0, 0
    for i in range(n):
        if ai < len(aq) and i == slots[ai]:
            queries.append({"sql": aq[ai], "attack": True})
            ai += 1
        else:
            queries.append({"sql": filler[fi], "attack": False})
            fi += 1
    harm_index = slots[harm_local]
    return {"purpose": purpose, "label": 1, "family": family, "queries": queries,
            "harm_index": harm_index, "onset": slots[0]}


def generate(db_path: Path, n_benign: int, n_attack: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    con = sqlite3.connect(db_path)
    out = []
    for purpose in ["sales_reporting", "finance_analytics", "supply_chain", "customer_support"]:
        out += [benign_session(rng, purpose) for _ in range(n_benign)]
    for family in FAMILIES:
        out += [attack_session(rng, family, con) for _ in range(n_attack)]
    for i, s in enumerate(out):
        s["id"] = f"{s['family']}_{i:05d}"
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=db.DEFAULT_OUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-benign", type=int, default=400, help="per purpose")
    ap.add_argument("--n-attack", type=int, default=100, help="per family")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    if not a.db.exists():
        db.build(a.db)
    sessions = generate(a.db, a.n_benign, a.n_attack, a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")
    print(f"{len(sessions)} sessions -> {a.out}")

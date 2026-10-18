"""Build a small, seeded TPC-H-shaped SQLite database for the purpose-SQL study.

Synthetic values only (names like Customer#000000042, random phones and
addresses), so the database can be regenerated and shared freely.

    python experiments/purpose_sql/tpch_db.py [--out data/purpose_sql/tpch_small.sqlite]
"""
from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "purpose_sql" / "tpch_small.sqlite"

REGIONS = ["AFRICA", "AMERICA", "ASIA", "EUROPE", "MIDDLE EAST"]
NATIONS = [("ALGERIA", 0), ("ARGENTINA", 1), ("BRAZIL", 1), ("CANADA", 1), ("EGYPT", 4),
           ("ETHIOPIA", 0), ("FRANCE", 3), ("GERMANY", 3), ("INDIA", 2), ("INDONESIA", 2),
           ("IRAN", 4), ("IRAQ", 4), ("JAPAN", 2), ("JORDAN", 4), ("KENYA", 0), ("MOROCCO", 0),
           ("MOZAMBIQUE", 0), ("PERU", 1), ("CHINA", 2), ("ROMANIA", 3), ("SAUDI ARABIA", 4),
           ("VIETNAM", 2), ("RUSSIA", 3), ("UNITED KINGDOM", 3), ("UNITED STATES", 1)]
SEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "HOUSEHOLD", "MACHINERY"]
PRIORITIES = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
SHIPMODES = ["AIR", "FOB", "MAIL", "RAIL", "REG AIR", "SHIP", "TRUCK"]
TYPES = ["STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"]
FINISH = ["ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"]
STREETS = ["Oak", "Maple", "Cedar", "Pine", "Elm", "Lake", "Hill", "Main", "Park", "River"]

N_CUSTOMER, N_ORDERS, N_SUPPLIER, N_PART = 1500, 7500, 100, 400
YEARS = (1993, 1998)

DDL = """
CREATE TABLE region(r_regionkey INTEGER PRIMARY KEY, r_name TEXT);
CREATE TABLE nation(n_nationkey INTEGER PRIMARY KEY, n_name TEXT, n_regionkey INTEGER);
CREATE TABLE customer(c_custkey INTEGER PRIMARY KEY, c_name TEXT, c_address TEXT, c_nationkey INTEGER,
                      c_phone TEXT, c_acctbal REAL, c_mktsegment TEXT);
CREATE TABLE orders(o_orderkey INTEGER PRIMARY KEY, o_custkey INTEGER, o_orderstatus TEXT,
                    o_totalprice REAL, o_orderdate TEXT, o_orderpriority TEXT);
CREATE TABLE supplier(s_suppkey INTEGER PRIMARY KEY, s_name TEXT, s_address TEXT, s_nationkey INTEGER,
                      s_phone TEXT, s_acctbal REAL);
CREATE TABLE part(p_partkey INTEGER PRIMARY KEY, p_name TEXT, p_type TEXT, p_brand TEXT, p_retailprice REAL);
CREATE TABLE partsupp(ps_partkey INTEGER, ps_suppkey INTEGER, ps_availqty INTEGER, ps_supplycost REAL,
                      PRIMARY KEY(ps_partkey, ps_suppkey));
CREATE TABLE lineitem(l_orderkey INTEGER, l_linenumber INTEGER, l_partkey INTEGER, l_suppkey INTEGER,
                      l_quantity INTEGER, l_extendedprice REAL, l_discount REAL, l_shipmode TEXT,
                      l_shipdate TEXT, PRIMARY KEY(l_orderkey, l_linenumber));
"""

SCHEMA = {
    "region": {"r_regionkey": "int", "r_name": "text"},
    "nation": {"n_nationkey": "int", "n_name": "text", "n_regionkey": "int"},
    "customer": {"c_custkey": "int", "c_name": "text", "c_address": "text", "c_nationkey": "int",
                 "c_phone": "text", "c_acctbal": "real", "c_mktsegment": "text"},
    "orders": {"o_orderkey": "int", "o_custkey": "int", "o_orderstatus": "text", "o_totalprice": "real",
               "o_orderdate": "text", "o_orderpriority": "text"},
    "supplier": {"s_suppkey": "int", "s_name": "text", "s_address": "text", "s_nationkey": "int",
                 "s_phone": "text", "s_acctbal": "real"},
    "part": {"p_partkey": "int", "p_name": "text", "p_type": "text", "p_brand": "text",
             "p_retailprice": "real"},
    "partsupp": {"ps_partkey": "int", "ps_suppkey": "int", "ps_availqty": "int", "ps_supplycost": "real"},
    "lineitem": {"l_orderkey": "int", "l_linenumber": "int", "l_partkey": "int", "l_suppkey": "int",
                 "l_quantity": "int", "l_extendedprice": "real", "l_discount": "real", "l_shipmode": "text",
                 "l_shipdate": "text"},
}


def _phone(rng: random.Random, nation: int) -> str:
    return f"{10 + nation}-{rng.randint(100, 999)}-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"


def _address(rng: random.Random) -> str:
    return f"{rng.randint(1, 9999)} {rng.choice(STREETS)} St"


def build(out: Path, seed: int = 7) -> Path:
    rng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(out)
    con.executescript(DDL)
    con.executemany("INSERT INTO region VALUES (?,?)", list(enumerate(REGIONS)))
    con.executemany("INSERT INTO nation VALUES (?,?,?)", [(i, n, r) for i, (n, r) in enumerate(NATIONS)])
    cust = []
    for k in range(1, N_CUSTOMER + 1):
        nat = rng.randrange(len(NATIONS))
        cust.append((k, f"Customer#{k:09d}", _address(rng), nat, _phone(rng, nat),
                     round(rng.uniform(-999.99, 9999.99), 2), rng.choice(SEGMENTS)))
    con.executemany("INSERT INTO customer VALUES (?,?,?,?,?,?,?)", cust)
    supp = []
    for k in range(1, N_SUPPLIER + 1):
        nat = rng.randrange(len(NATIONS))
        supp.append((k, f"Supplier#{k:09d}", _address(rng), nat, _phone(rng, nat),
                     round(rng.uniform(-999.99, 9999.99), 2)))
    con.executemany("INSERT INTO supplier VALUES (?,?,?,?,?,?)", supp)
    parts = []
    for k in range(1, N_PART + 1):
        parts.append((k, f"part {k}", f"{rng.choice(TYPES)} {rng.choice(FINISH)}",
                      f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}", round(900 + k / 10 + rng.random() * 100, 2)))
    con.executemany("INSERT INTO part VALUES (?,?,?,?,?)", parts)
    ps = []
    for p in range(1, N_PART + 1):
        for s in rng.sample(range(1, N_SUPPLIER + 1), 4):
            ps.append((p, s, rng.randint(1, 9999), round(rng.uniform(1, 1000), 2)))
    con.executemany("INSERT INTO partsupp VALUES (?,?,?,?)", ps)
    start = date(YEARS[0], 1, 1)
    span = (date(YEARS[1], 8, 2) - start).days
    orders, items = [], []
    for k in range(1, N_ORDERS + 1):
        od = start + timedelta(days=rng.randrange(span))
        total = 0.0
        for ln in range(1, rng.randint(1, 5) + 1):
            p = rng.randint(1, N_PART)
            qty = rng.randint(1, 50)
            price = round(qty * parts[p - 1][4], 2)
            total += price
            items.append((k, ln, p, rng.randint(1, N_SUPPLIER), qty, price, round(rng.uniform(0, 0.1), 2),
                          rng.choice(SHIPMODES), (od + timedelta(days=rng.randint(1, 120))).isoformat()))
        orders.append((k, rng.randint(1, N_CUSTOMER), rng.choice("FOP"), round(total, 2), od.isoformat(),
                       rng.choice(PRIORITIES)))
    con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    con.executemany("INSERT INTO lineitem VALUES (?,?,?,?,?,?,?,?,?)", items)
    con.commit()
    con.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=7)
    print(build(ap.parse_args().out, ap.parse_args().seed))

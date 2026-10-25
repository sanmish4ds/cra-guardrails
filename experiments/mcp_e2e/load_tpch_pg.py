"""Load a seeded TPC-H-shaped database into PostgreSQL at roughly scale factor 1.

Same schema and value distributions as experiments/purpose_sql/tpch_db.py, larger:
150,000 customers, 1,500,000 orders, about 4.5M line items, 10,000 suppliers,
20,000 parts. Grants SELECT to the read-only role the gateway uses.

    python experiments/mcp_e2e/load_tpch_pg.py --dsn "host=localhost port=55432 user=postgres dbname=tpch"
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "purpose_sql"))
import tpch_db as t                                                          # noqa: E402

N_CUST, N_ORD, N_SUPP, N_PART = 150_000, 1_500_000, 10_000, 20_000
DDL = t.DDL.replace("INTEGER PRIMARY KEY", "INTEGER PRIMARY KEY").replace("REAL", "DOUBLE PRECISION")


def copy(cur, table: str, rows) -> None:
    buf = io.StringIO()
    for r in rows:
        buf.write("\t".join("\\N" if v is None else str(v) for v in r) + "\n")
    buf.seek(0)
    with cur.copy(f"COPY {table} FROM STDIN") as cp:
        while chunk := buf.read(1 << 20):
            cp.write(chunk)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default="host=localhost port=55432 user=postgres dbname=tpch")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    t0 = time.time()
    with psycopg.connect(a.dsn, autocommit=True) as con, con.cursor() as cur:
        for tbl in ("lineitem", "partsupp", "part", "supplier", "orders", "customer", "nation", "region"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl}")
        cur.execute(DDL)
        copy(cur, "region", list(enumerate(t.REGIONS)))
        copy(cur, "nation", [(i, n, r) for i, (n, r) in enumerate(t.NATIONS)])
        cust = []
        for k in range(1, N_CUST + 1):
            nat = rng.randrange(len(t.NATIONS))
            cust.append((k, f"Customer#{k:09d}", t._address(rng), nat, t._phone(rng, nat),
                         round(rng.uniform(-999.99, 9999.99), 2), rng.choice(t.SEGMENTS)))
        copy(cur, "customer", cust)
        supp = []
        for k in range(1, N_SUPP + 1):
            nat = rng.randrange(len(t.NATIONS))
            supp.append((k, f"Supplier#{k:09d}", t._address(rng), nat, t._phone(rng, nat),
                         round(rng.uniform(-999.99, 9999.99), 2)))
        copy(cur, "supplier", supp)
        parts = [(k, f"part {k}", f"{rng.choice(t.TYPES)} {rng.choice(t.FINISH)}",
                  f"Brand#{rng.randint(1, 5)}{rng.randint(1, 5)}", round(900 + k / 100 + rng.random() * 100, 2))
                 for k in range(1, N_PART + 1)]
        copy(cur, "part", parts)
        copy(cur, "partsupp", [(p, s, rng.randint(1, 9999), round(rng.uniform(1, 1000), 2))
                               for p in range(1, N_PART + 1) for s in rng.sample(range(1, N_SUPP + 1), 4)])
        start = date(t.YEARS[0], 1, 1)
        span = (date(t.YEARS[1], 8, 2) - start).days
        orders, items = [], []
        for k in range(1, N_ORD + 1):
            od = start + timedelta(days=rng.randrange(span))
            total = 0.0
            for ln in range(1, rng.randint(1, 5) + 1):
                p = rng.randint(1, N_PART)
                qty = rng.randint(1, 50)
                price = round(qty * parts[p - 1][4], 2)
                total += price
                items.append((k, ln, p, rng.randint(1, N_SUPP), qty, price, round(rng.uniform(0, 0.1), 2),
                              rng.choice(t.SHIPMODES), (od + timedelta(days=rng.randint(1, 120))).isoformat()))
            orders.append((k, rng.randint(1, N_CUST), rng.choice("FOP"), round(total, 2), od.isoformat(),
                           rng.choice(t.PRIORITIES)))
            if len(items) > 500_000:
                copy(cur, "lineitem", items)
                items = []
        copy(cur, "lineitem", items)
        copy(cur, "orders", orders)
        for stmt in ("CREATE INDEX ON orders(o_custkey)", "CREATE INDEX ON lineitem(l_orderkey)",
                     "CREATE INDEX ON customer(c_nationkey)", "CREATE INDEX ON partsupp(ps_suppkey)", "ANALYZE"):
            cur.execute(stmt)
        cur.execute("GRANT USAGE ON SCHEMA public TO gateway_ro")
        cur.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO gateway_ro")
        cur.execute("SELECT (SELECT count(*) FROM customer), (SELECT count(*) FROM orders), (SELECT count(*) FROM lineitem)")
        print("loaded", cur.fetchone(), f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

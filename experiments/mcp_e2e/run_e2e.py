"""End-to-end MCP evaluation at TPC-H scale factor ~1 on PostgreSQL.

Sessions (benign per purpose and the five attack families) are generated against
the Postgres database, transpiled from SQLite to PostgreSQL, and replayed by an
MCP client through the gateway on two paths:

  direct  client -> gateway MCP server -> Postgres (read-only role)
  proxy   client -> gateway MCP server -> database MCP server -> Postgres

For every query we record the gateway decision and timings (gateway check,
database execution, client round trip). We also run the same sessions through
the purpose monitor in-process and report decision parity with the MCP paths.

    python experiments/mcp_e2e/run_e2e.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sqlglot
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "guardrail-service"))
sys.path.insert(0, str(ROOT / "experiments" / "purpose_sql"))

import sessions as gen                                                       # noqa: E402
from guardrail.backends import PostgresBackend                               # noqa: E402
from guardrail.purpose import PurposeMonitor, PurposePolicy, analyze, violations   # noqa: E402

PY = str(ROOT / "guardrail-service" / ".venv" / "bin" / "python")
POLICY = ROOT / "experiments" / "purpose_sql" / "purpose_policy.yaml"
OUT = ROOT / "experiments" / "results" / "mcp_e2e"
DSN_RO = "host=localhost port=55432 user=gateway_ro dbname=tpch"


class Con:
    """sqlite3-style adapter over the read-only Postgres backend, for the session generator."""

    def __init__(self, backend: PostgresBackend):
        self.b = backend

    def execute(self, sql: str, params=()):
        for p in params:
            sql = sql.replace("?", repr(p) if isinstance(p, str) else str(p), 1)
        rows = self.b.execute(sql)
        return type("C", (), {"fetchone": lambda _: rows[0] if rows else None, "fetchall": lambda _: rows})()


def to_pg(sql: str) -> str:
    return sqlglot.transpile(sql, read="sqlite", write="postgres")[0]


def make_sessions(backend, n_benign: int, n_attack: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    con = Con(backend)
    out = []
    for p in ("sales_reporting", "finance_analytics", "supply_chain", "customer_support"):
        out += [gen.benign_session(rng, p) for _ in range(n_benign)]
    for fam in gen.FAMILIES:
        out += [gen.attack_session(rng, fam, con) for _ in range(n_attack)]
    for s in out:
        for q in s["queries"]:
            q["sql"] = to_pg(q["sql"])
    return out


def policy_for(backend) -> PurposePolicy:
    raw = yaml.safe_load(POLICY.read_text())
    raw.update(schema=backend.schema(), dialect="postgres", min_isolation=3, linear_audit=True,
               enforce_entity_budget=True)
    return PurposePolicy.model_validate(raw)


def library_decisions(sessions, backend) -> list[list[str]]:
    pol = policy_for(backend)
    out = []
    for s in sessions:
        mon = PurposeMonitor(pol, s["purpose"], executor=backend.execute)
        dec = []
        for q in s["queries"]:
            u = analyze(q["sql"], pol)
            if u.tree is None or violations(u, pol.purposes[s["purpose"]], pol):
                dec.append("refuse")
                continue
            dec.append("refuse" if mon.check(q["sql"]).refuse else "allow")
        out.append(dec)
    return out


async def replay(sessions, path: str) -> list[list[dict]]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    env = {"GATEWAY_POLICY": str(POLICY), "DB_DSN": DSN_RO, "GATEWAY_BACKEND": "postgres" if path == "direct" else "mcp",
           "DOWNSTREAM_CMD": f"{PY} -m guardrail.db_mcp_server", "GATEWAY_K": "3", "GATEWAY_LINEAR": "1",
           "PYTHONPATH": str(ROOT / "guardrail-service")}
    params = StdioServerParameters(command=PY, args=["-m", "guardrail.mcp_gateway"], env=env,
                                   cwd=str(ROOT / "guardrail-service"))
    parse = lambda res: json.loads("".join(getattr(c, "text", "") for c in res.content))   # noqa: E731
    out = []
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        for sess in sessions:
            await s.call_tool("begin_session", {"purpose": sess["purpose"]})
            recs = []
            for q in sess["queries"]:
                t0 = time.perf_counter()
                d = parse(await s.call_tool("run_sql", {"sql": q["sql"]}, read_timeout_seconds=600))
                rt = 1000 * (time.perf_counter() - t0)
                tm = d.get("timing_ms", {})
                recs.append({"decision": d.get("decision", "error"), "error": d.get("error"),
                             "check_ms": tm.get("check"), "execute_ms": tm.get("execute"), "roundtrip_ms": round(rt, 2)})
            out.append(recs)
    return out


def q(xs, p):
    xs = [x for x in xs if x is not None]
    return round(float(np.percentile(xs, p)), 2) if xs else None


def summarize(sessions, runs, lib) -> dict:
    res = {"n_sessions": len(sessions), "n_queries": sum(len(s["queries"]) for s in sessions), "paths": {}}
    for path, recs in runs.items():
        flat = [r for rs in recs for r in rs]
        allowed = [r for r in flat if r["decision"] == "allow"]
        par = [a["decision"] == b for rs, ls in zip(recs, lib) for a, b in zip(rs, ls)]
        overhead = [r["roundtrip_ms"] - (r["check_ms"] or 0) - (r["execute_ms"] or 0) for r in allowed]
        rel = [r["check_ms"] / r["execute_ms"] for r in allowed if r["execute_ms"]]
        prevented, benign_ref, benign_n = defaultdict(list), defaultdict(int), defaultdict(int)
        for s, rs in zip(sessions, recs):
            first = next((i for i, r in enumerate(rs) if r["decision"] == "refuse"), None)
            if s["label"] == 1:
                prevented[s["family"]].append(first is not None and first <= s["harm_index"])
            else:
                benign_ref[s["purpose"]] += sum(r["decision"] == "refuse" for r in rs)
                benign_n[s["purpose"]] += len(rs)
        res["paths"][path] = {
            "decision_parity_with_library": round(float(np.mean(par)), 4),
            "errors": sum(r["decision"] == "error" for r in flat),
            "check_ms": {"median": q([r["check_ms"] for r in flat], 50), "p95": q([r["check_ms"] for r in flat], 95)},
            "execute_ms": {"median": q([r["execute_ms"] for r in allowed], 50),
                           "p95": q([r["execute_ms"] for r in allowed], 95)},
            "roundtrip_ms": {"median": q([r["roundtrip_ms"] for r in flat], 50),
                             "p95": q([r["roundtrip_ms"] for r in flat], 95)},
            "transport_overhead_ms": {"median": q(overhead, 50), "p95": q(overhead, 95)},
            "check_over_execute": {"median": q(rel, 50), "p95": q(rel, 95)},
            "attack_prevented": {f: round(float(np.mean(v)), 3) for f, v in prevented.items()},
            "benign_query_refusal": {p: round(benign_ref[p] / benign_n[p], 4) for p in benign_n},
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-benign", type=int, default=40, help="per purpose")
    ap.add_argument("--n-attack", type=int, default=20, help="per family")
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    backend = PostgresBackend(DSN_RO)
    sessions = make_sessions(backend, a.n_benign, a.n_attack, a.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sessions_pg.json").write_text(json.dumps(sessions))
    t0 = time.time()
    lib = library_decisions(sessions, backend)
    print(f"library pass {time.time() - t0:.0f}s", flush=True)
    runs = {}
    for path in ("direct", "proxy"):
        t0 = time.time()
        runs[path] = asyncio.run(replay(sessions, path))
        print(f"{path} pass {time.time() - t0:.0f}s", flush=True)
    res = summarize(sessions, runs, lib)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    (OUT / "per_query.json").write_text(json.dumps(runs))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

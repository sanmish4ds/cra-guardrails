"""Gateway MCP server: purpose-scoped SQL access for LLM agents (research prototype).

An agent connects to this server instead of to the database (or the database's
own MCP server). One stdio connection is one session.

Tools
  begin_session(purpose)  choose a purpose from the policy; resets session state
  run_sql(sql)            parse, check against the purpose (column modes), run the
                          session audit, then execute on the backend; returns rows
                          or a refusal with reasons

Configuration (environment)
  GATEWAY_POLICY   purpose policy YAML (schema is read from the database)
  GATEWAY_BACKEND  "postgres" (default) or "mcp"
  DB_DSN           Postgres DSN for the postgres backend (use a read-only role)
  DOWNSTREAM_CMD   command line of the downstream MCP server for the mcp backend
  GATEWAY_K, GATEWAY_LINEAR, GATEWAY_STRICT   audit settings (default 3, 1, 0)
  GATEWAY_ENTITY_BUDGET  refuse queries beyond a purpose's entity budget (default 1)
  GATEWAY_MAX_ROWS rows returned to the agent per query (default 200)

    GATEWAY_POLICY=... DB_DSN=... python -m guardrail.mcp_gateway
"""
from __future__ import annotations

import asyncio
import json
import os
import shlex
import time
from pathlib import Path

import yaml
from mcp.server.mcpserver import MCPServer

from .backends import McpBackend, PostgresBackend
from .purpose import PurposeMonitor, PurposePolicy, analyze, violations

server = MCPServer("purpose-gateway",
                   instructions="Call begin_session(purpose) first, then run_sql(sql). Queries are checked "
                                "against the purpose and the session's earlier answers before they run.")


class State:
    backend = None
    policy: PurposePolicy | None = None
    monitor: PurposeMonitor | None = None
    purpose: str | None = None


S = State()


async def _setup() -> None:
    if S.policy is not None:
        return
    if os.environ.get("GATEWAY_BACKEND", "postgres") == "mcp":
        parts = shlex.split(os.environ["DOWNSTREAM_CMD"])
        S.backend = McpBackend(parts[0], parts[1:], env=dict(os.environ))
        await S.backend.start()
    else:
        S.backend = PostgresBackend(os.environ["DB_DSN"])
    raw = yaml.safe_load(Path(os.environ["GATEWAY_POLICY"]).read_text())
    raw["schema"] = await asyncio.to_thread(S.backend.schema)
    raw["dialect"] = S.backend.dialect
    raw["min_isolation"] = int(os.environ.get("GATEWAY_K", "3"))
    raw["linear_audit"] = os.environ.get("GATEWAY_LINEAR", "1") == "1"
    raw["strict_protected"] = os.environ.get("GATEWAY_STRICT", "0") == "1"
    raw["enforce_entity_budget"] = os.environ.get("GATEWAY_ENTITY_BUDGET", "1") == "1"
    S.policy = PurposePolicy.model_validate(raw)


@server.tool()
async def begin_session(purpose: str) -> str:
    """Start a session for one declared purpose (e.g. finance_analytics)."""
    await _setup()
    if purpose not in S.policy.purposes:
        return json.dumps({"error": f"unknown purpose; choose one of {sorted(S.policy.purposes)}"})
    S.purpose = purpose
    S.monitor = PurposeMonitor(S.policy, purpose, executor=S.backend.execute)
    cols = S.policy.purposes[purpose].columns
    return json.dumps({"purpose": purpose, "columns": cols})


@server.tool()
async def run_sql(sql: str) -> str:
    """Run one read-only SQL query within the current session's purpose."""
    if S.monitor is None:
        return json.dumps({"error": "call begin_session(purpose) first"})
    t0 = time.perf_counter()
    q = analyze(sql, S.policy)
    if q.tree is None:
        return json.dumps({"decision": "refuse", "reasons": ["query could not be parsed or resolved"]})
    v = violations(q, S.policy.purposes[S.purpose], S.policy)
    if v:
        return json.dumps({"decision": "refuse", "reasons": ["beyond purpose: " + ", ".join(sorted(v))],
                           "escalation_request": sorted(v),
                           "timing_ms": {"check": round(1000 * (time.perf_counter() - t0), 2)}})
    sig = await asyncio.to_thread(S.monitor.check, sql)
    t1 = time.perf_counter()
    if sig.refuse:
        return json.dumps({"decision": "refuse", "reasons": sig.reasons,
                           "timing_ms": {"check": round(1000 * (t1 - t0), 2)}})
    try:
        rows = await asyncio.to_thread(S.backend.execute, sql)
    except Exception as e:
        return json.dumps({"error": str(e).splitlines()[0][:300]})
    t2 = time.perf_counter()
    cap = int(os.environ.get("GATEWAY_MAX_ROWS", "200"))
    return json.dumps({"decision": "allow", "rows": rows[:cap], "row_count": len(rows),
                       "notices": sig.reasons,
                       "timing_ms": {"check": round(1000 * (t1 - t0), 2), "execute": round(1000 * (t2 - t1), 2)}},
                      default=str)


if __name__ == "__main__":
    server.run("stdio")

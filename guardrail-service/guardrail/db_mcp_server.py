"""Minimal database MCP server (stand-in for a vendor server such as a Postgres or SQLcl MCP server).

Tools: run_sql(sql) -> {"rows": [...]}, schema() -> {"schema": {table: {column: type}}}.
It applies no policy of its own; it is the "one broad account" the gateway protects.

    DB_DSN="host=localhost port=55432 user=gateway_ro dbname=tpch" python -m guardrail.db_mcp_server
"""
from __future__ import annotations

import json
import os

from mcp.server.mcpserver import MCPServer

from .backends import PostgresBackend

server = MCPServer("database")
_db: PostgresBackend | None = None


def db() -> PostgresBackend:
    global _db
    if _db is None:
        _db = PostgresBackend(os.environ["DB_DSN"])
    return _db


@server.tool()
def run_sql(sql: str) -> str:
    """Run one SQL statement and return its rows."""
    try:
        return json.dumps({"rows": db().execute(sql)}, default=str)
    except Exception as e:                       # report, do not crash the server
        return json.dumps({"error": str(e).splitlines()[0][:300]})


@server.tool()
def schema() -> str:
    """Tables and columns of the public schema."""
    return json.dumps({"schema": db().schema()})


if __name__ == "__main__":
    server.run("stdio")

"""Database backends for the MCP gateway.

PostgresBackend runs queries itself on a read-only connection.
McpBackend forwards every query to a downstream database MCP server's SQL tool,
so the gateway can sit in front of an existing vendor server (SQLcl MCP,
a Postgres MCP server, ...). Both expose the same synchronous interface the
purpose monitor needs; McpBackend bridges to its async client from a worker
thread.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import psycopg


class PostgresBackend:
    dialect = "postgres"

    def __init__(self, dsn: str, statement_timeout_ms: int = 30000):
        self.con = psycopg.connect(dsn, autocommit=True)
        self.con.execute("SET default_transaction_read_only = on")
        self.con.execute(f"SET statement_timeout = {int(statement_timeout_ms)}")

    def execute(self, sql: str) -> list[tuple]:
        with self.con.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall() if cur.description else []

    def schema(self) -> dict[str, dict[str, str]]:
        rows = self.execute("SELECT table_name, column_name, data_type FROM information_schema.columns "
                            "WHERE table_schema = 'public' ORDER BY table_name, ordinal_position")
        out: dict[str, dict[str, str]] = {}
        for t, c, typ in rows:
            out.setdefault(t, {})[c] = typ
        return out


class McpBackend:
    """Forward SQL to a downstream MCP server tool that returns {"rows": [...]} as JSON text."""

    dialect = "postgres"

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None,
                 tool: str = "run_sql", schema_tool: str = "schema"):
        self.params = (command, args, env)
        self.tool, self.schema_tool = tool, schema_tool
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session = None

    async def start(self) -> None:
        """Open the downstream session in its own long-lived task (MCP task groups must not
        be entered in one request handler and exited in another)."""
        self.loop = asyncio.get_running_loop()
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self._task = asyncio.create_task(self._hold())
        await self._ready.wait()
        if self.session is None:
            raise RuntimeError("downstream MCP server failed to start")

    async def _hold(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        command, args, env = self.params
        try:
            async with stdio_client(StdioServerParameters(command=command, args=args, env=env)) as (read, write), \
                    ClientSession(read, write) as session:
                await session.initialize()
                self.session = session
                self._ready.set()
                await self._stop.wait()
        finally:
            self._ready.set()

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        res = await self.session.call_tool(tool, arguments)
        text = "".join(getattr(c, "text", "") for c in res.content)
        data = json.loads(text)
        if "error" in data:
            raise RuntimeError(data["error"])
        return data

    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()

    def execute(self, sql: str) -> list[tuple]:      # call from a worker thread, not the event loop
        return [tuple(r) for r in self._run(self.call(self.tool, {"sql": sql}))["rows"]]

    def schema(self) -> dict[str, dict[str, str]]:
        return self._run(self.call(self.schema_tool, {}))["schema"]

    async def aclose(self) -> None:
        self._stop.set()
        await self._task

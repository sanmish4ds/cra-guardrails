"""Minimal Python client for the guardrail service (pip install httpx)."""
from __future__ import annotations

import httpx


class GuardrailClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 10.0, fail_open: bool = False):
        self.http = httpx.Client(base_url=base_url, headers={"X-API-Key": api_key}, timeout=timeout)
        self.fail_open = fail_open   # what to do if the service itself is unreachable

    def _post(self, path: str, body: dict) -> dict:
        try:
            r = self.http.post(path, json=body)
            if r.status_code == 401:
                return {"decision": "block",
                        "reasons": ["guardrail service rejected the API key (401); check configuration"]}
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            return {"decision": "allow" if self.fail_open else "block",
                    "reasons": [f"guardrail service unavailable: {type(e).__name__}"]}

    def check_input(self, subject: str, text: str) -> dict:
        return self._post("/v1/check/input", {"subject": subject, "text": text})

    def check_output(self, subject: str, prompt: str, response: str) -> dict:
        return self._post("/v1/check/output", {"subject": subject, "prompt": prompt, "response": response})

    def check_sql(self, subject: str, sql: str) -> dict:
        return self._post("/v1/check/sql", {"subject": subject, "sql": sql})

    def report_sql_result(self, subject: str, sql: str, rows_returned: int) -> dict:
        return self._post("/v1/report/sql-result", {"subject": subject, "sql": sql, "rows_returned": rows_returned})

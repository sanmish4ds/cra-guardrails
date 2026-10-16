"""Policy configuration, loaded from YAML (path in GUARDRAIL_POLICY)."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ServiceConfig(BaseModel):
    api_keys_env: str = "GUARDRAIL_API_KEYS"   # comma-separated keys in this env var
    allow_unauthenticated: bool = False        # local development only
    cors_origins: list[str] = Field(default_factory=list)
    audit_log: str | None = "audit.jsonl"


class TextGuardConfig(BaseModel):
    backend: str = "ollama"                    # ollama | none
    model: str = "llama-guard3:1b"
    url: str = "http://localhost:11434"
    timeout_seconds: float = 30.0
    input_threshold: float = 0.5               # calibrate with `python -m guardrail.calibrate`
    output_threshold: float = 0.5
    warn_margin: float = 0.15                  # scores within this margin below a threshold -> warn
    on_error: str = "block"                    # block | allow when the scorer is unavailable


class SqlGuardConfig(BaseModel):
    dialect: str = "oracle"
    allowed_tables: list[str] = Field(default_factory=list)       # empty = any table
    allowed_schemas: list[str] = Field(default_factory=list)      # empty = any schema
    sensitive_columns: dict[str, list[str]] = Field(default_factory=dict)  # table -> columns
    sensitive_action: str = "warn"             # warn | block
    require_limit_for_star: bool = True
    max_rows: int | None = 1000                # a literal row limit above this is refused
    blocked_identifier_prefixes: list[str] = Field(
        default_factory=lambda: ["utl_", "dbms_", "sys.", "v$", "gv$", "dba_", "all_users"])


class SessionConfig(BaseModel):
    window_seconds: int = 86400
    sensitive_rows_warn: int = 200
    sensitive_rows_block: int = 1000
    total_rows_block: int = 50000


class Policy(BaseModel):
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    text_guard: TextGuardConfig = Field(default_factory=TextGuardConfig)
    sql_guard: SqlGuardConfig = Field(default_factory=SqlGuardConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)

    def api_keys(self) -> set[str]:
        raw = os.environ.get(self.service.api_keys_env, "")
        return {k.strip() for k in raw.split(",") if k.strip()}


def load_policy(path: str | os.PathLike | None = None) -> Policy:
    path = path or os.environ.get("GUARDRAIL_POLICY")
    if not path:
        return Policy()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return Policy.model_validate(data)

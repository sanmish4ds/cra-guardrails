"""HTTP API. Run: GUARDRAIL_POLICY=policy.yaml uvicorn guardrail.app:app

Every check returns {"decision": "allow" | "warn" | "block", "reasons": [...]}.
The calling app decides what a decision means for its users; the service
never sees or stores more than the text it is asked to check.
"""
from __future__ import annotations

import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .audit import AuditLog
from .config import Policy, load_policy
from .session import ExposureLedger
from .sql_guard import check_sql
from .text_guard import Scorer, decide, make_scorer

SEVERITY = {"allow": 0, "warn": 1, "block": 2}


def worst(*decisions: str) -> str:
    return max(decisions, key=SEVERITY.__getitem__)


class Message(BaseModel):
    role: str
    content: str


class InputCheck(BaseModel):
    subject: str = Field(description="Stable id of the end user, set by your server")
    text: str
    history: list[Message] = Field(default_factory=list, description="Earlier turns; used only if use_history")
    use_history: bool = False


class OutputCheck(BaseModel):
    subject: str
    prompt: str
    response: str


class SqlCheck(BaseModel):
    subject: str
    sql: str


class SqlResult(BaseModel):
    subject: str
    sql: str
    rows_returned: int = Field(ge=0)


class Verdict(BaseModel):
    decision: str
    reasons: list[str] = Field(default_factory=list)
    score: float | None = None
    detail: dict = Field(default_factory=dict)


def create_app(policy: Policy | None = None, scorer: Scorer | None = None) -> FastAPI:
    policy = policy or load_policy()
    scorer = scorer or make_scorer(policy.text_guard)
    ledger = ExposureLedger(policy.session)
    audit = AuditLog(policy.service.audit_log)
    keys = policy.api_keys()
    if not keys and not policy.service.allow_unauthenticated:
        raise RuntimeError(f"no API keys in ${policy.service.api_keys_env}; "
                           "set keys or service.allow_unauthenticated: true (development only)")

    app = FastAPI(title="Guardrail service", version="0.1.0")
    if policy.service.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=policy.service.cors_origins,
                           allow_methods=["POST", "GET"], allow_headers=["Content-Type", "X-API-Key"])

    def auth(x_api_key: str | None = Header(default=None)) -> None:
        if policy.service.allow_unauthenticated:
            return
        if x_api_key not in keys:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    def score_text(messages: list[dict], threshold: float) -> tuple[str, float | None, list[str]]:
        tg = policy.text_guard
        try:
            s = scorer.score(messages)
        except Exception as e:  # scorer down: fail closed or open, as configured
            d = "block" if tg.on_error == "block" else "allow"
            return d, None, [f"text guard unavailable ({type(e).__name__}); on_error={tg.on_error}"]
        d = decide(s, threshold, tg.warn_margin)
        reasons = [] if d == "allow" else [f"safety score {s:.3f} vs threshold {threshold:.3f}"]
        return d, s, reasons

    @app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok", "text_backend": policy.text_guard.backend,
                "model": policy.text_guard.model if policy.text_guard.backend != "none" else None,
                "sql_dialect": policy.sql_guard.dialect}

    @app.post("/v1/check/input", response_model=Verdict, dependencies=[Depends(auth)])
    def check_input(req: InputCheck) -> Verdict:
        t0 = time.perf_counter()
        msgs = [m.model_dump() for m in req.history] if req.use_history else []
        msgs.append({"role": "user", "content": req.text})
        d, s, reasons = score_text(msgs, policy.text_guard.input_threshold)
        audit.write("check/input", req.subject, d, reasons, req.text, score=s,
                    ms=round(1000 * (time.perf_counter() - t0), 1))
        return Verdict(decision=d, reasons=reasons, score=s)

    @app.post("/v1/check/output", response_model=Verdict, dependencies=[Depends(auth)])
    def check_output(req: OutputCheck) -> Verdict:
        t0 = time.perf_counter()
        msgs = [{"role": "user", "content": req.prompt}, {"role": "assistant", "content": req.response}]
        d, s, reasons = score_text(msgs, policy.text_guard.output_threshold)
        audit.write("check/output", req.subject, d, reasons, req.response, score=s,
                    ms=round(1000 * (time.perf_counter() - t0), 1))
        return Verdict(decision=d, reasons=reasons, score=s)

    @app.post("/v1/check/sql", response_model=Verdict, dependencies=[Depends(auth)])
    def check_sql_route(req: SqlCheck) -> Verdict:
        v = check_sql(req.sql, policy.sql_guard)
        exposure = ledger.current(req.subject)
        bd, breasons = ledger.verdict(exposure)
        # an exhausted budget blocks further sensitive reads, not every query
        if bd == "block" and not v.sensitive_columns and "sensitive-row" in breasons[0]:
            bd, breasons = "warn", breasons
        d = worst(v.decision, bd)
        reasons = v.reasons + breasons
        audit.write("check/sql", req.subject, d, reasons, req.sql, tables=v.tables)
        return Verdict(decision=d, reasons=reasons, detail={
            "tables": v.tables, "sensitive_columns": v.sensitive_columns, "row_limit": v.row_limit,
            "exposure": vars(exposure)})

    @app.post("/v1/report/sql-result", response_model=Verdict, dependencies=[Depends(auth)])
    def report_sql_result(req: SqlResult) -> Verdict:
        v = check_sql(req.sql, policy.sql_guard)
        exposure = ledger.record(req.subject, req.rows_returned, sensitive=bool(v.sensitive_columns))
        d, reasons = ledger.verdict(exposure)
        audit.write("report/sql-result", req.subject, d, reasons, req.sql, rows=req.rows_returned)
        return Verdict(decision=d, reasons=reasons, detail={"exposure": vars(exposure)})

    return app


def __getattr__(name: str):
    # `uvicorn guardrail.app:app` builds the app on first access, from GUARDRAIL_POLICY
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)

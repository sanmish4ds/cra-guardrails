import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from guardrail.app import create_app
from guardrail.calibrate import calibrate
from guardrail.config import Policy, load_policy
from guardrail.session import ExposureLedger
from guardrail.sql_guard import check_sql

POLICY = load_policy(Path(__file__).resolve().parents[1] / "policy.example.yaml")
SQL = POLICY.sql_guard
KEY = {"X-API-Key": "test-key"}


class FakeScorer:
    """Scores text containing 'attack' as unsafe."""

    def score(self, messages):
        return 0.9 if any("attack" in m["content"] for m in messages) else 0.05


class BrokenScorer:
    def score(self, messages):
        raise ConnectionError("down")


def client(tmp_path, monkeypatch, scorer=None, **overrides) -> TestClient:
    monkeypatch.setenv("GUARDRAIL_API_KEYS", "test-key")
    pol = POLICY.model_copy(deep=True)
    pol.service.audit_log = str(tmp_path / "audit.jsonl")
    for k, v in overrides.items():
        setattr(pol.session, k, v)
    return TestClient(create_app(pol, scorer or FakeScorer()))


@pytest.mark.parametrize("sql", [
    "DROP TABLE orders", "TRUNCATE TABLE lineitem", "DELETE FROM customer",
    "UPDATE customer SET c_acctbal = 0", "SELECT 1 FROM nation; DROP TABLE orders",
    "SELECT * FROM orders", "SELECT * FROM v$session", "SELECT username FROM sys.dba_users",
    "SELECT UTL_HTTP.REQUEST('http://x') FROM dual", "SELECT DBMS_RANDOM.VALUE FROM dual",
    "SELECT * FROM orders FETCH FIRST 5000 ROWS ONLY", "SELEC broken",
])
def test_sql_blocked(sql):
    assert check_sql(sql, SQL).decision == "block"


@pytest.mark.parametrize("sql", [
    "SELECT o_orderkey FROM orders FETCH FIRST 10 ROWS ONLY",
    "SELECT n_name, COUNT(*) FROM nation n JOIN customer c ON c.c_nationkey = n.n_nationkey GROUP BY n_name",
    "WITH t AS (SELECT o_custkey, SUM(o_totalprice) s FROM orders GROUP BY o_custkey) SELECT * FROM t FETCH FIRST 5 ROWS ONLY",
    "SELECT n_name FROM nation UNION SELECT r_name FROM region",
    "SELECT 'dbms_x' AS label FROM dual",
])
def test_sql_allowed(sql):
    assert check_sql(sql, SQL).decision == "allow"


def test_sensitive_columns_warn_and_star_expands():
    v = check_sql("SELECT c.* FROM customer c WHERE ROWNUM <= 10", SQL)
    assert v.decision == "warn" and v.sensitive_columns == ["c_address", "c_phone"] and v.row_limit == 10


def test_auth_required(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.post("/v1/check/sql", json={"subject": "u1", "sql": "SELECT 1 FROM dual"}).status_code == 401
    assert c.get("/v1/health").status_code == 200


def test_refuses_to_start_without_keys(monkeypatch):
    monkeypatch.delenv("GUARDRAIL_API_KEYS", raising=False)
    with pytest.raises(RuntimeError):
        create_app(Policy(), FakeScorer())


def test_text_checks(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    assert c.post("/v1/check/input", json={"subject": "u1", "text": "how do joins work"}, headers=KEY).json()["decision"] == "allow"
    r = c.post("/v1/check/output", json={"subject": "u1", "prompt": "q", "response": "attack plan"}, headers=KEY).json()
    assert r["decision"] == "block" and r["score"] == 0.9


def test_scorer_down_fails_closed(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, scorer=BrokenScorer())
    r = c.post("/v1/check/input", json={"subject": "u1", "text": "hi"}, headers=KEY).json()
    assert r["decision"] == "block" and "unavailable" in r["reasons"][0]


def test_exposure_budget_blocks_sensitive_reads_only(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch, sensitive_rows_warn=50, sensitive_rows_block=100)
    phone = "SELECT c_name, c_phone FROM customer FETCH FIRST 60 ROWS ONLY"
    for _ in range(2):  # 120 sensitive rows reported
        c.post("/v1/report/sql-result", json={"subject": "u1", "sql": phone, "rows_returned": 60}, headers=KEY)
    assert c.post("/v1/check/sql", json={"subject": "u1", "sql": phone}, headers=KEY).json()["decision"] == "block"
    other = c.post("/v1/check/sql", json={"subject": "u1", "sql": "SELECT n_name FROM nation"}, headers=KEY).json()
    assert other["decision"] == "warn"                      # non-sensitive reads still allowed
    fresh = c.post("/v1/check/sql", json={"subject": "u2", "sql": phone}, headers=KEY).json()
    assert fresh["decision"] == "warn"                      # budgets are per subject


def test_budget_window_expires():
    t = [0.0]
    led = ExposureLedger(POLICY.session.model_copy(update={"window_seconds": 10}), clock=lambda: t[0])
    led.record("u", 2000, sensitive=True)
    assert led.verdict(led.current("u"))[0] == "block"
    t[0] = 11.0
    assert led.current("u").sensitive_rows == 0


def test_audit_log_has_no_raw_text(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    c.post("/v1/check/sql", json={"subject": "alice@example.com", "sql": "SELECT c_phone FROM customer"}, headers=KEY)
    raw = (tmp_path / "audit.jsonl").read_text()
    rec = json.loads(raw.splitlines()[-1])
    assert "c_phone FROM" not in raw and "alice" not in raw and rec["decision"] == "warn"


def test_calibrate():
    scores = [i / 100 for i in range(100)]
    th = calibrate(scores, 0.05)
    assert sum(s >= th for s in scores) <= 5

import sqlite3

import pytest

from guardrail.purpose import PurposeMonitor, PurposePolicy, analyze, violations

SCHEMA = {"customer": {"c_custkey": "int", "c_name": "text", "c_phone": "text",
                       "c_acctbal": "real", "c_seg": "text"}}


def policy(**kw) -> PurposePolicy:
    return PurposePolicy.model_validate({
        "schema": SCHEMA,
        "entities": {"customer": "c_custkey"},
        "sensitivity": {"customer.c_phone": 1.0, "customer.c_acctbal": 0.8},
        "purposes": {
            "finance": {"columns": {"customer.c_acctbal": "agg", "customer.c_seg": "raw",
                                    "customer.c_custkey": "filter"}},
            "support": {"columns": {"customer.c_custkey": "raw", "customer.c_name": "raw",
                                    "customer.c_phone": "raw"},
                        "entity_budget": {"customer": 2}},
        },
        **kw,
    })


@pytest.fixture
def run():
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE customer(c_custkey INT, c_name TEXT, c_phone TEXT, c_acctbal REAL, c_seg TEXT)")
    con.executemany("INSERT INTO customer VALUES (?,?,?,?,?)",
                    [(k, f"C{k}", f"555-{k:04d}", 100.0 * k, "A" if k <= 10 else "B") for k in range(1, 21)])
    return lambda sql: con.execute(sql).fetchall()


def test_analyze_modes():
    q = analyze("SELECT c.c_seg, AVG(c.c_acctbal) FROM customer c WHERE c.c_custkey > 3 GROUP BY c.c_seg", policy())
    assert q.uses == {"customer.c_seg": "raw", "customer.c_acctbal": "agg", "customer.c_custkey": "filter"}


def test_subquery_projection_is_filter_use():
    q = analyze("SELECT COUNT(*) FROM customer WHERE c_custkey IN (SELECT c_custkey FROM customer WHERE c_seg = 'A')",
                policy())
    assert q.uses["customer.c_custkey"] == "filter"


def test_violation_weights():
    p = policy()
    v = violations(analyze("SELECT c_acctbal, c_phone FROM customer", p), p.purposes["finance"], p)
    assert v == {"customer.c_acctbal:raw": 0.8, "customer.c_phone:raw": 1.0}


def test_session_drift_counts_each_violation_once():
    m = PurposeMonitor(policy(), "finance")
    for _ in range(5):
        s = m.check("SELECT c_name FROM customer LIMIT 1")
    assert s.drift_q == pytest.approx(0.1) and s.drift == pytest.approx(0.1)


def test_small_cell_and_tracker(run):
    m = PurposeMonitor(policy(), "finance", executor=run)
    assert m.check("SELECT AVG(c_acctbal) FROM customer WHERE c_seg = 'A'").cell == pytest.approx(0.1)
    s = m.check("SELECT AVG(c_acctbal) FROM customer WHERE c_seg = 'A' AND c_custkey <> 4")
    assert s.cell_q == pytest.approx(1 / 9)          # each query alone looks fine
    assert s.cell == 1.0                             # together they isolate customer 4


def test_grouped_small_cell(run):
    m = PurposeMonitor(policy(), "finance", executor=run)
    assert m.check("SELECT c_custkey, AVG(c_acctbal) FROM customer GROUP BY c_custkey").cell_q == 1.0


def test_audit_mode_refuses_completing_query_only(run):
    m = PurposeMonitor(policy(min_isolation=3), "finance", executor=run)
    assert not m.check("SELECT AVG(c_acctbal) FROM customer WHERE c_seg = 'A'").refuse
    assert m.check("SELECT AVG(c_acctbal) FROM customer WHERE c_seg = 'A' AND c_custkey <> 4").refuse
    assert not m.check("SELECT AVG(c_acctbal) FROM customer WHERE c_seg = 'B'").refuse


def test_breadth_over_budget(run):
    m = PurposeMonitor(policy(), "support", executor=run)
    for k in (1, 2):
        s = m.check(f"SELECT c_name, c_phone FROM customer WHERE c_custkey = {k}")
    assert s.breadth == 1.0 and not s.violations
    s = m.check("SELECT c_name, c_phone FROM customer WHERE c_custkey = 3")
    assert s.breadth == 1.5 and any("budget 2" in r for r in s.reasons)


def test_no_executor_keeps_data_signals_off():
    s = PurposeMonitor(policy(), "finance").check("SELECT AVG(c_acctbal) FROM customer WHERE c_custkey = 1")
    assert s.cell == 0.0 and s.breadth == 0.0


def test_count_is_filter_and_projected_max_is_raw():
    q = analyze("SELECT COUNT(c_name), MAX(c_acctbal) FROM customer WHERE c_seg IN "
                "(SELECT c_seg FROM customer GROUP BY c_seg HAVING MAX(c_acctbal) > 5)", policy())
    assert q.uses == {"customer.c_name": "filter", "customer.c_acctbal": "raw", "customer.c_seg": "filter"}

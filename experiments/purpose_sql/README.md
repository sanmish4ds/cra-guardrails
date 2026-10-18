# Purpose-anchored session guardrails for SQL agents (prototype)

Conversation-shape signals (text drift, refusal trends) tracked session length
rather than risk in the v2 evaluation. This study tests the same idea grounded
in **what the SQL touches**: each session declares a purpose, the purpose lists
the columns it may use and how (`filter` < `agg` < `raw`), and a monitor parses
every query before it runs (`guardrail-service/guardrail/purpose.py`).

| Signal | Per query (stateless) | Per session (stateful) |
|---|---|---|
| drift | sensitivity-weighted out-of-purpose column uses | sum over distinct (column, mode) violations |
| cell | 1 / smallest group an aggregate over a protected column is computed on | also 1 / *d* when two aggregates' entity sets are nested and differ by *d* (tracker / differencing) |
| breadth | entities whose sensitive columns this query returns / purpose budget | distinct entities so far / budget |

`cell` and `breadth` run a rewritten key-only query against a read-only
connection to learn which entities a query would touch.

## Run

```bash
P=guardrail-service/.venv/bin/python
$P experiments/purpose_sql/tpch_db.py        # seeded synthetic TPC-H, data/purpose_sql/tpch_small.sqlite
$P experiments/purpose_sql/sessions.py       # 1,600 benign + 400 attack sessions
$P experiments/purpose_sql/run_eval.py       # experiments/results/purpose_sql/results.json (~30 s)
```

## Setup

* Database: 1,500 customers, 7,500 orders, 100 suppliers; synthetic values.
* Purposes: `sales_reporting`, `finance_analytics` (balances in aggregates only),
  `customer_support` (contact details for up to 5 customers), `supply_chain`.
* Benign sessions: 400 per purpose, analyst-style templates with noise. 30% of
  sessions contain one out-of-purpose query (5% a sensitive one). Finance
  sessions include drill-downs and overlapping histogram bands. Support agents
  look up one to five customers.
* Attacks, 100 each: `drift` (scope creep through low-sensitivity columns, half
  ending with one sensitive column), `small_cell`, `tracker` (two large-group
  aggregates that differ by one or two customers), `enumeration`.
* Every session has 6–16 queries regardless of label; the query-count baseline
  sits at AUROC 0.53–0.61.
* Thresholds: per purpose, on half of the benign sessions; metrics on the other
  half plus all attacks. "Prevented" means the alarm fired at or before the
  query that completes the harm.

## Results (seed 11)

Session alarms at 1% target benign FPR (observed test FPR ≤ 0.5% for all rows
except the row ledger, 3% on finance). TPR / prevented-before-harm:

| Detector | drift | small_cell | tracker | enumeration |
|---|---|---|---|---|
| Query count | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Existing SQL guard (sensitive-query count) | 0 / 0 | .01 / 0 | 0 / 0 | .86 / .26 |
| Existing row ledger | .22 / .22 | .01 / 0 | .01 / .01 | 1.0 / 1.0 |
| Purpose, stateless OR | .47 / .47 | 0 / 0 † | 0 / 0 | .87 / .79 |
| Purpose, stateful OR | .66 / .66 | 0 / 0 † | 0 / 0 | 1.0 / 1.0 |

† The single-signal `cell_q` alone catches 87% of small-cell attacks at this
budget; splitting the budget three ways in the OR detector loses them.

**Session alarms cannot handle differencing.** Stateful `cell` ranks tracker
attacks well (AUROC 0.90) but catches none at 1–5% FPR, because 14% of
*benign* finance sessions genuinely isolate one customer. A typical case is
"average balance by nation × segment", then "same nation and segment, balance > 0".
Honest workflows leak too, so intent-level alarms are the wrong tool here.

**Query-level refusal works.** In audit mode, the monitor refuses only the
query that would isolate fewer than *k* customers, given what this session has
already been answered:

| k | Mode | Benign queries refused | Benign sessions with ≥1 refusal | small_cell prevented | tracker prevented |
|---|---|---|---|---|---|
| 2 | stateless (cell suppression) | 0.05% | 0.5% | .87 | 0 |
| 2 | stateful | 1.5% | 13.8% | .87 | .83 |
| 3 | stateless | 0.2% | 2.5% | 1.0 | .01 |
| 3 | stateful | 2.9% | 23.3% | 1.0 | .97 |
| 5 | stateless | 13.5% | 75.7% | 1.0 | .60 |
| 5 | stateful | 13.5% | 75.7% | 1.0 | 1.0 |

Takeaways:

1. Anchoring to the purpose separates scope creep that per-query sensitive-column rules miss entirely (0.66 vs 0).
2. State helps where the violation is cumulative: gradual drift (.66 vs .47), enumeration (1.0 vs .87 TPR), and differencing.
3. For differencing, keeping session state and refusing at query level is what prevents tracker attacks. Per-query cell suppression stops almost none (.97 vs .01 at k = 3). The price is that about 3% of honest analyst queries get refused.

## Threats to validity

* The same authors wrote the purposes, the attacks and the monitor; the attacks
  are scripted, not adaptive. Next: LLM-agent attackers that see refusals and
  retry, and attacks that difference across three or more queries (the monitor
  checks pairs only).
* Benign sessions are templates, not logged analyst work. Benign cost should be
  re-measured on real query logs (e.g., BIRD/Spider question sequences or
  consented SQLclMCP lab sessions).
* One synthetic database and one seed; purposes are hand-written. Deriving
  purposes from role grants or query history is open.
* `cell` / `breadth` run extra key-only queries per checked query; the cost on a
  production warehouse is unmeasured.

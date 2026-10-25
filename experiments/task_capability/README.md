# Step 1: task → least-privilege data capability (BIRD)

Question: can an agent data-access gateway turn a task description into a
column-level capability that is much narrower than a database role, without
breaking the task?

## Setup

* 220 BIRD dev tasks (20 per database, seed 13). BIRD is CC BY-SA 4.0 and not
  redistributed here; set `BIRD_DIR` to a local copy.
* gpt-4o-mini, temperature 0, derives a capability: `table.column → filter | agg | raw`.
  * filter: WHERE / JOIN / GROUP BY / ORDER BY, or inside COUNT
  * agg: also SUM / AVG and similar
  * raw: values returned, including a projected MIN / MAX
* Grants compared:
  * `full`: every column, raw (today's agent role)
  * `tables`: all columns of the LLM-chosen tables
  * `columns`: the LLM capability
  * `columns+keys`: the capability plus primary / foreign / shared-name keys as filter
  * `gold`: the gold SQL's own footprint, an oracle reference
* Agent: the same model writes SQL seeing only the granted columns and their
  modes. The gateway refuses SQL that uses anything else. A query that names
  columns that don't exist counts as invalid.
* Escalation: one round. A refused query's out-of-grant uses are granted, as
  an approver inside the user's role would, and the query runs.

```bash
BIRD_DIR=~/SQLclMCP/data/bird_data guardrail-service/.venv/bin/python experiments/task_capability/run_capability_eval.py
```

LLM responses are cached in `experiments/results/task_capability/llm_cache.json`,
so reruns are free and deterministic.

## Results

| Grant | Gold fits | Agent EX | Refused | Columns | Raw columns | Beyond gold |
|---|---|---|---|---|---|---|
| full (role) | 0.995 | **0.527** | 0.005 | 72.5 | 72.5 | 71.6 |
| tables | 0.777 | 0.459 | 0.005 | 27.1 | 27.1 | 26.3 |
| columns | 0.168 | 0.150 | 0.359 | 3.7 | 0.9 | 1.1 |
| columns+keys | 0.518 | 0.405 | 0.273 | 7.4 | 0.9 | 3.8 |
| **columns+keys + 1 escalation** | — | **0.477** | — | **7.5** | ~0.9 | 3.9 |
| gold (oracle) | 1.0 | 0.555 | 0.173 | 4.6 | 0.9 | 0 |

Paired differences against `full` (bootstrap 95% CI):

* escalation: −5.0 points [−9.5, −0.5]
* gold: +2.7 points [−3.6, +9.1]

## What it shows

1. **Least privilege is not inherently costly.** The oracle grant (4.6 columns)
   is at least as accurate as the full role (72.5 columns); narrowing the
   schema even helps the agent.
2. **The cost comes from deriving the capability.** The LLM's raw column picks
   miss join keys and alternative columns, and the gold SQL fits them only 17%
   of the time. Deterministic keys plus one escalation round recover most of
   it: about 10× fewer columns and about 80× fewer raw-returnable columns than
   the role, for −5 points of accuracy. 27% of tasks escalate, requesting 0.4
   columns on average.
3. **Hard tasks suffer most.** On challenging tasks, `columns+keys` without
   escalation drops to 0.33 from 0.52.

## Caveats

* One model (gpt-4o-mini) and one sample. Stronger models may derive better
  capabilities and write better SQL.
* Escalation is auto-approved inside the role. It is also an attack channel:
  injected tasks or SQL could request broad access. Testing that is the next step.
* "Gold fits" undercounts valid capabilities when another column set would
  also answer the task (e.g. `schools.School` instead of `frpm."School Name"`);
  agent EX is the primary metric.
* Rows are not scoped yet (no row predicates in the capability).

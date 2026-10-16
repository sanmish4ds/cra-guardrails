# Guardrail service

A small HTTP service any application can call to check what goes into and
comes out of an LLM, and what the LLM is about to do to a database. It
packages the controls that held up in our confound-controlled evaluation
(`paper/v2`): judge the reply and the action, not only the prompt; enforce
rules deterministically where possible; keep state only for violations that
are defined over a session.

| Endpoint | Checks | How |
|---|---|---|
| `POST /v1/check/input` | A user message | Llama Guard 3 probability vs a calibrated threshold |
| `POST /v1/check/output` | A model reply, given its prompt | Same, on the prompt–reply pair |
| `POST /v1/check/sql` | SQL before it runs | Parsed allow-list: one read statement, allowed tables, no system objects or packages (`UTL_*`, `DBMS_*`, `V$`, `DBA_*`), row limits, sensitive columns; plus the caller's exposure budget |
| `POST /v1/report/sql-result` | Rows a query returned | Adds to a per-user, time-windowed budget of sensitive rows |
| `GET /v1/health` | — | Liveness, no auth |

Every check answers `{"decision": "allow" | "warn" | "block", "reasons": [...]}`.
Your app decides what each decision means for its users.

## Run it

```bash
cd guardrail-service
python3 -m venv .venv && .venv/bin/pip install -e .
cp policy.example.yaml policy.yaml            # edit tables, columns, budgets
export GUARDRAIL_POLICY=policy.yaml GUARDRAIL_API_KEYS=change-me
ollama pull llama-guard3:1b                    # or set text_guard.backend: none
.venv/bin/uvicorn guardrail.app:app --port 8787
.venv/bin/pytest -q                            # 26 tests
```

`Dockerfile` builds the same service (not exercised in this repository; the
container needs network access to an Ollama host via `text_guard.url`).

## Plug it into an app

Call it **from your server**, never from the browser, and pass as `subject`
an identity your server controls (the logged-in user id). A session id sent
by the client lets an attacker reset their budget by rotating ids.

A Node.js route that runs user or model SQL (`clients/js/guardrail-client.js`):

```js
const { GuardrailClient } = require('./guardrail-client');
const guard = new GuardrailClient({ baseUrl: process.env.GUARDRAIL_URL, apiKey: process.env.GUARDRAIL_KEY });

app.post('/run-sql', requireLogin, async (req, res) => {
  const subject = req.user.id;
  const verdict = await guard.checkSql(subject, req.body.sql);
  if (verdict.decision === 'block') return res.status(403).json({ error: verdict.reasons });
  const result = await readOnlyDb.execute(req.body.sql, [], { maxRows: 500 });
  await guard.reportSqlResult(subject, req.body.sql, result.rows.length);
  res.json({ rows: result.rows, warnings: verdict.decision === 'warn' ? verdict.reasons : [] });
});
```

Wrap the model the same way: `checkInput` before the call, `checkOutput`
on the reply, `checkSql` on any SQL it generates. The Python client
(`clients/python/guardrail_client.py`) has the same four methods.

Both clients fail **closed** by default: if the service is unreachable they
return `block`. Pass `failOpen: true` / `fail_open=True` only for checks
you can afford to skip.

## Calibrate the thresholds

The policy's `0.5` thresholds are placeholders. Collect conversations from
your own traffic that should *not* be flagged, hold them out from any tuning,
and run:

```bash
python -m guardrail.calibrate benign_inputs.jsonl --kind input --alpha 0.05
python -m guardrail.calibrate benign_outputs.jsonl --kind output --alpha 0.05
```

Each prints the threshold that flags at most 5% of those conversations. In
our study, thresholds set on training data instead gave 18.5% false alarms
against a nominal 5%.

## What it deliberately does not do

- **No "intent drift" or other conversation-shape scores.** In our evaluation
  they tracked session length, not risk. The only state kept is the data
  exposure budget, a violation that is defined over a session.
- **It is not a substitute for database permissions.** Give the application
  a read-only database user; the SQL guard is the second line, not the first.
- **The exposure ledger is in memory, per process.** Run one replica, or
  replace `ExposureLedger` with a shared store before scaling out.
- **Llama Guard 3 1B is a compact classifier.** It misses things and has
  false alarms; measure it on your traffic, and swap the `Scorer` for a
  larger guard if you need to.

## Audit log

Every decision is appended to `service.audit_log` as JSON: time, endpoint,
a hash of the subject and of the checked text, its length, the decision,
reasons and scores. No raw text is stored.

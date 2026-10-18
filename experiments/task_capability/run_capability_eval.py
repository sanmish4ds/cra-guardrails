"""Step 1 of the agent data-access gateway: can a task be turned into a least-privilege data capability?

For a stratified sample of BIRD dev tasks, an LLM derives a column-level
capability (column -> filter | agg | raw) from the task text and the schema.
We compare grants of decreasing breadth:

  full     every column, raw            (what an agent's database role gets today)
  tables   every column of the tables the LLM picked, raw   (table-level least privilege)
  columns  the LLM's column-level capability with modes
  columns+keys  plus primary / foreign keys (and same-name columns) of the granted tables, as filter
  gold     the gold SQL's own footprint (oracle minimum; reference only)

For columns+keys we also report one escalation round: a refused query's
out-of-grant uses are granted (as an approver inside the user's role would)
and the query runs; we count how many columns that adds.

and measure, per grant:

  admissible  the gold SQL fits inside the grant (a correct answer exists under it)
  agent EX    a text-to-SQL agent that sees only the grant answers correctly; a query
              that uses columns beyond the grant is refused and counts as wrong
  privilege   columns granted, raw columns granted, and columns beyond the gold footprint

    BIRD_DIR=~/SQLclMCP/data/bird_data python experiments/task_capability/run_capability_eval.py
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "guardrail-service"))

from guardrail.purpose import LEVEL, PurposePolicy, PurposeSpec, analyze, violations   # noqa: E402

BIRD = Path(os.environ.get("BIRD_DIR", Path.home() / "SQLclMCP" / "data" / "bird_data")).expanduser()
OUT = ROOT / "experiments" / "results" / "task_capability"
CACHE = OUT / "llm_cache.json"
GRANTS = ("full", "tables", "columns", "columns+keys", "gold")

CAP_PROMPT = """You grant a data-analysis agent the minimum database access it needs for ONE task.

Access modes per column (each implies the ones before it):
  filter  usable only in WHERE / JOIN / GROUP BY / ORDER BY or inside COUNT; values are never returned
  agg     may also appear inside SUM / AVG and similar aggregates
  raw     values may be returned to the user (a returned MIN / MAX counts as raw)

Grant only columns the task needs, at the lowest mode that still lets it be answered.
Join keys need "filter". Columns whose values the answer must show need "raw".

Database schema (table.column: type -- description):
{schema}

Task: {question}
Hint: {evidence}

Reply with JSON only: {{"columns": {{"table.column": "filter|agg|raw", ...}}}}"""

SQL_PROMPT = """Write one SQLite query that answers the task. You may use only the columns listed,
and only in the way their mode allows:
  filter  only in WHERE / JOIN / GROUP BY / ORDER BY, or inside COUNT
  agg     also inside SUM / AVG and similar aggregates
  raw     anywhere, including the SELECT list (and returned MIN / MAX)
Queries that use anything else are refused.

Columns (table.column [mode]: type -- description):
{schema}

Task: {question}
Hint: {evidence}

Reply with the SQL only, no explanation."""


# ── BIRD loading ─────────────────────────────────────────────────────────────

def db_path(db: str) -> Path:
    return BIRD / "dev_databases" / db / f"{db}.sqlite"


def load_schema(db: str) -> dict[str, dict[str, str]]:
    con = sqlite3.connect(f"file:{db_path(db)}?mode=ro", uri=True)
    out = {}
    for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        out[t] = {r[1]: (r[2] or "text") for r in con.execute(f'PRAGMA table_info("{t}")')}
    return out


def join_keys(db: str, tables: set[str]) -> dict[str, str]:
    """PK / FK columns of the given tables, plus column names shared by two of them, as filter."""
    con = sqlite3.connect(f"file:{db_path(db)}?mode=ro", uri=True)
    names = {t.lower(): t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    out, seen = {}, defaultdict(set)
    for t in tables:
        real = names.get(t)
        if real is None:
            continue
        for r in con.execute(f'PRAGMA table_info("{real}")'):
            seen[r[1].lower()].add(t)
            if r[5]:
                out[f"{t}.{r[1]}".lower()] = "filter"
        for r in con.execute(f'PRAGMA foreign_key_list("{real}")'):
            out[f"{t}.{r[3]}".lower()] = "filter"
            if r[2].lower() in tables and r[4]:
                out[f"{r[2]}.{r[4]}".lower()] = "filter"
    for col, ts in seen.items():
        if len(ts) > 1:
            for t in ts:
                out[f"{t}.{col}"] = "filter"
    return out


def load_descriptions(db: str) -> dict[str, str]:
    """lower(table.column) -> short description from BIRD's database_description CSVs."""
    out = {}
    d = BIRD / "dev_databases" / db / "database_description"
    for f in d.glob("*.csv"):
        table = f.stem.lower()
        try:
            rows = list(csv.DictReader(f.open(encoding="utf-8-sig", errors="replace")))
        except Exception:
            continue
        for r in rows:
            col = (r.get("original_column_name") or "").strip().lower()
            desc = " ".join(x.strip() for x in (r.get("column_description"), r.get("value_description")) if x and x.strip())
            if col:
                out[f"{table}.{col}"] = re.sub(r"\s+", " ", desc)[:160]
    return out


def schema_text(schema, desc, allowed: dict[str, str] | None = None) -> str:
    lines = []
    for t, cols in schema.items():
        for c, typ in cols.items():
            key = f"{t}.{c}".lower()
            if allowed is not None and key not in allowed:
                continue
            mode = f" [{allowed[key]}]" if allowed is not None else ""
            name = f'"{t}"."{c}"' if (" " in t or " " in c or "-" in c) else f"{t}.{c}"
            lines.append(f"{name}{mode}: {typ} -- {desc.get(key, '')}".rstrip(" -"))
    return "\n".join(lines)


def sample_tasks(n_per_db: int, seed: int) -> list[dict]:
    data = json.loads((BIRD / "dev.json").read_text())
    by_db = defaultdict(list)
    for x in data:
        by_db[x["db_id"]].append(x)
    rng = random.Random(seed)
    out = []
    for db in sorted(by_db):
        out += rng.sample(by_db[db], n_per_db)
    return out


# ── LLM ──────────────────────────────────────────────────────────────────────

class LLM:
    def __init__(self, model: str):
        self.model = model
        self.url = os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        self.key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        self.cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self.sem = asyncio.Semaphore(8)
        self.calls = 0

    async def __call__(self, client: httpx.AsyncClient, prompt: str) -> str:
        h = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        if h in self.cache:
            return self.cache[h]
        if not self.key:
            raise SystemExit("set OPENAI_API_KEY (or LLM_API_KEY) in .env")
        body = {"model": self.model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
        async with self.sem:
            for attempt in range(5):
                r = await client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=120)
                if r.status_code == 200:
                    break
                await asyncio.sleep(2 ** attempt)
            r.raise_for_status()
        self.calls += 1
        text = r.json()["choices"][0]["message"]["content"]
        self.cache[h] = text
        return text

    def save(self) -> None:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(self.cache, indent=0))


def parse_capability(text: str, schema) -> dict[str, str]:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        cols = json.loads(m.group(0))["columns"] if m else {}
    except (json.JSONDecodeError, KeyError, TypeError):
        cols = {}
    valid = {f"{t}.{c}".lower() for t, cs in schema.items() for c in cs}
    out = {}
    for k, v in cols.items():
        key = k.replace('"', "").replace("`", "").lower().strip()
        if key in valid and v in ("filter", "agg", "raw"):
            out[key] = v
    return out


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";")


# ── evaluation ───────────────────────────────────────────────────────────────

def run_sql(db: str, sql: str, timeout_s: float = 30.0):
    con = sqlite3.connect(f"file:{db_path(db)}?mode=ro", uri=True)
    start = time.monotonic()
    con.set_progress_handler(lambda: int(time.monotonic() - start > timeout_s), 10_000)
    try:
        return set(con.execute(sql).fetchall())
    except Exception:
        return None
    finally:
        con.close()


def fits(sql: str, grant: dict[str, str], policy: PurposePolicy) -> bool:
    q = analyze(sql, policy)
    return q.tree is not None and not violations(q, PurposeSpec(columns=grant), policy)


async def evaluate(tasks, llm: LLM) -> list[dict]:
    schemas = {t["db_id"]: load_schema(t["db_id"]) for t in tasks}
    descs = {db: load_descriptions(db) for db in schemas}
    policies = {db: PurposePolicy.model_validate({"schema": s, "purposes": {"_": {"columns": {}}}})
                for db, s in schemas.items()}
    async with httpx.AsyncClient() as client:
        caps = await asyncio.gather(*[
            llm(client, CAP_PROMPT.format(schema=schema_text(schemas[t["db_id"]], descs[t["db_id"]]),
                                          question=t["question"], evidence=t["evidence"] or "none"))
            for t in tasks])
        grants_all = []
        for t, cap_text in zip(tasks, caps):
            sch, pol = schemas[t["db_id"]], policies[t["db_id"]]
            allcols = {f"{tb}.{c}".lower(): "raw" for tb, cs in sch.items() for c in cs}
            cols = parse_capability(cap_text, sch)
            tables = {k.split(".")[0] for k in cols}
            gold = {c: m for c, m in analyze(t["SQL"], pol).uses.items()}
            keys = join_keys(t["db_id"], tables)
            grants_all.append({
                "full": allcols,
                "tables": {k: "raw" for k in allcols if k.split(".")[0] in tables},
                "columns": cols,
                "columns+keys": keys | cols,
                "gold": gold,
            })
        jobs = [(i, g) for i in range(len(tasks)) for g in GRANTS]
        sqls = await asyncio.gather(*[
            llm(client, SQL_PROMPT.format(
                schema=schema_text(schemas[tasks[i]["db_id"]], descs[tasks[i]["db_id"]], grants_all[i][g]),
                question=tasks[i]["question"], evidence=tasks[i]["evidence"] or "none"))
            for i, g in jobs])
    rows = []
    for i, t in enumerate(tasks):
        db, pol = t["db_id"], policies[t["db_id"]]
        gold_rows = run_sql(db, t["SQL"])
        n_cols = sum(len(c) for c in schemas[db].values())
        rec = {"question_id": t["question_id"], "db_id": db, "difficulty": t["difficulty"],
               "n_schema_columns": n_cols, "gold_footprint": grants_all[i]["gold"], "grants": {}}
        for g in GRANTS:
            grant = grants_all[i][g]
            sql = extract_sql(sqls[jobs.index((i, g))])
            valid = analyze(sql, pol).tree is not None      # resolves against the real schema
            allowed = valid and fits(sql, grant, pol)
            res = run_sql(db, sql) if allowed else None
            gold_fp = grants_all[i]["gold"]
            rec["grants"][g] = {
                "n_columns": len(grant),
                "n_raw": sum(m == "raw" for m in grant.values()),
                "beyond_gold": sum(1 for c, m in grant.items() if LEVEL[m] > LEVEL[gold_fp.get(c, "none")]),
                "admissible": fits(t["SQL"], grant, pol),
                "agent_sql": sql,
                "agent_valid": valid,
                "agent_allowed": allowed,
                "agent_correct": bool(allowed and res is not None and gold_rows is not None and res == gold_rows),
            }
            if g == "columns":
                rec["capability"] = grant
            if g == "columns+keys":
                q = analyze(sql, pol)
                extra = violations(q, PurposeSpec(columns=grant), pol) if q.tree is not None else {}
                esc = dict(grant)
                for k in extra:
                    col, mode = k.rsplit(":", 1)
                    esc[col] = mode
                res2 = run_sql(db, sql) if q.tree is not None else None
                rec["escalation"] = {
                    "requested_columns": len(extra),
                    "n_columns": len(esc),
                    "beyond_gold": sum(1 for c, m in esc.items() if LEVEL[m] > LEVEL[gold_fp.get(c, "none")]),
                    "agent_correct": bool(res2 is not None and gold_rows is not None and res2 == gold_rows),
                }
        rows.append(rec)
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    out = {"n_tasks": n, "grants": {}}
    for g in GRANTS:
        G = [r["grants"][g] for r in rows]
        out["grants"][g] = {
            "admissible": round(sum(x["admissible"] for x in G) / n, 3),
            "agent_ex": round(sum(x["agent_correct"] for x in G) / n, 3),
            "agent_refused": round(sum(x["agent_valid"] and not x["agent_allowed"] for x in G) / n, 3),
            "agent_invalid": round(sum(not x["agent_valid"] for x in G) / n, 3),
            "mean_columns": round(sum(x["n_columns"] for x in G) / n, 1),
            "mean_raw": round(sum(x["n_raw"] for x in G) / n, 1),
            "mean_beyond_gold": round(sum(x["beyond_gold"] for x in G) / n, 1),
            "share_of_schema": round(sum(x["n_columns"] / r["n_schema_columns"] for x, r in zip(G, rows)) / n, 3),
        }
    E = [r["escalation"] for r in rows]
    out["columns+keys+escalation"] = {
        "agent_ex": round(sum(e["agent_correct"] for e in E) / n, 3),
        "tasks_escalating": round(sum(e["requested_columns"] > 0 for e in E) / n, 3),
        "mean_requested": round(sum(e["requested_columns"] for e in E) / n, 2),
        "mean_columns": round(sum(e["n_columns"] for e in E) / n, 1),
        "mean_beyond_gold": round(sum(e["beyond_gold"] for e in E) / n, 1),
    }
    by_diff = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    out["by_difficulty"] = {d: {g: {"admissible": round(sum(r["grants"][g]["admissible"] for r in rs) / len(rs), 3),
                                    "agent_ex": round(sum(r["grants"][g]["agent_correct"] for r in rs) / len(rs), 3)}
                                for g in GRANTS} | {"n": len(rs)} for d, rs in sorted(by_diff.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-db", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    a = ap.parse_args()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if not (BIRD / "dev.json").exists():
        raise SystemExit(f"BIRD dev not found under {BIRD}; set BIRD_DIR")
    tasks = sample_tasks(a.per_db, a.seed)
    llm = LLM(a.model)
    try:
        rows = asyncio.run(evaluate(tasks, llm))
    finally:
        llm.save()
    summary = summarize(rows) | {"model": a.model, "seed": a.seed, "per_db": a.per_db, "llm_calls": llm.calls}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "per_task.json").write_text(json.dumps(rows, indent=1))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{summary['n_tasks']} tasks, model {a.model}, {llm.calls} new LLM calls\n")
    print(f"{'grant':13}{'admissible':>11}{'agent EX':>10}{'refused':>9}{'invalid':>9}{'columns':>9}{'raw':>6}{'beyond gold':>13}{'% schema':>10}")
    for g, s in summary["grants"].items():
        print(f"{g:13}{s['admissible']:>11}{s['agent_ex']:>10}{s['agent_refused']:>9}{s['agent_invalid']:>9}{s['mean_columns']:>9}"
              f"{s['mean_raw']:>6}{s['mean_beyond_gold']:>13}{100 * s['share_of_schema']:>9.1f}%")
    e = summary["columns+keys+escalation"]
    print(f"\ncolumns+keys with one escalation round: agent EX {e['agent_ex']}, "
          f"{100 * e['tasks_escalating']:.0f}% of tasks escalate, {e['mean_requested']} columns requested on average, "
          f"final grant {e['mean_columns']} columns ({e['mean_beyond_gold']} beyond gold)")
    print("\nby difficulty (admissible / agent EX):")
    for d, s in summary["by_difficulty"].items():
        print(f"  {d:12} n={s['n']:3}  " + "  ".join(f"{g} {s[g]['admissible']}/{s[g]['agent_ex']}" for g in GRANTS))


if __name__ == "__main__":
    main()

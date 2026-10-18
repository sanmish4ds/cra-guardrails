"""Purpose-anchored session monitor for SQL agents (research prototype).

Each session declares a purpose. A purpose lists the columns it may touch and
how: ``filter`` (WHERE / JOIN / GROUP BY only), ``agg`` (inside an aggregate)
or ``raw`` (values returned to the caller). The monitor parses every query
before it runs and reports, per query (stateless) and per session (stateful):

* drift   -- sensitivity-weighted use of columns beyond what the purpose
             allows; the session score counts each (column, mode) once, so a
             long session that repeats the same slip does not grow.
* cell    -- 1 / size of the smallest group an aggregate over a protected
             column is computed on (small cells reveal individuals). The
             session score also catches differencing ("tracker") attacks: two
             aggregates whose entity sets are nested and differ by d entities
             score 1 / d.
* breadth -- distinct entities whose sensitive columns were returned raw,
             divided by the purpose's entity budget.

``cell`` and ``breadth`` need the entity keys a query would touch. The monitor
gets them by running a rewritten key-only query through ``executor`` (a
read-only connection), before the original query runs. Without an executor
those two signals stay 0.

Limits of the prototype: CTEs, set operations and correlated subqueries are
not rewritten; HAVING is dropped from key-set rewrites (groups it would hide
still count); differencing is checked between pairs of queries only.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.optimizer.qualify import qualify

LEVEL = {"none": 0, "filter": 1, "agg": 2, "raw": 3}

Executor = Callable[[str], list[tuple]]


class PurposeSpec(BaseModel):
    columns: dict[str, str]                                   # "table.column" -> filter | agg | raw
    entity_budget: dict[str, int] = Field(default_factory=dict)  # entity table -> max entities seen raw


class PurposePolicy(BaseModel):
    dialect: str = "sqlite"
    schema_: dict[str, dict[str, str]] = Field(alias="schema")   # table -> column -> type
    entities: dict[str, str] = Field(default_factory=dict)       # entity table -> key column
    sensitivity: dict[str, float] = Field(default_factory=dict)  # "table.column" -> weight
    default_sensitivity: float = 0.1
    protected_threshold: float = 0.5   # sensitivity at which a column gets cell / breadth checks
    default_entity_budget: int = 25
    max_tracked_sets: int = 200
    min_isolation: int | None = None   # audit mode: refuse a query whose aggregate isolates fewer entities
    purposes: dict[str, PurposeSpec]

    def weight(self, col: str) -> float:
        return self.sensitivity.get(col, self.default_sensitivity)


@dataclass
class QueryUses:
    uses: dict[str, str]          # "table.column" -> highest mode it is used in
    aliases: dict[str, str]       # alias -> table
    tree: exp.Expression | None   # qualified tree, None if unparseable


@dataclass
class PurposeSignals:
    drift_q: float = 0.0
    cell_q: float = 0.0
    breadth_q: float = 0.0
    drift: float = 0.0
    cell: float = 0.0
    breadth: float = 0.0
    refuse: bool = False          # audit mode: answering would isolate < min_isolation entities
    violations: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def analyze(sql: str, policy: PurposePolicy) -> QueryUses:
    """Resolve every column to table.column and the mode it is used in."""
    try:
        tree = qualify(sqlglot.parse_one(sql, read=policy.dialect), schema=policy.schema_,
                       dialect=policy.dialect, validate_qualify_columns=False)
    except sqlglot.errors.SqlglotError:
        return QueryUses({}, {}, None)
    aliases = {t.alias_or_name: t.name.lower() for t in tree.find_all(exp.Table)}
    top = [tree] if isinstance(tree, exp.Select) else [s for s in tree.find_all(exp.Select)
                                                       if s.parent_select is None]
    projected: set[int] = set()
    uses: dict[str, str] = {}

    def note(col: exp.Column, mode: str) -> None:
        table = aliases.get(col.table)
        if not table:
            return                                     # output alias or CTE column
        key = f"{table}.{col.name.lower()}"
        if LEVEL[mode] > LEVEL.get(uses.get(key, "none"), 0):
            uses[key] = mode

    for sel in top:
        for e in sel.expressions:
            for c in e.find_all(exp.Column):
                projected.add(id(c))
                note(c, "agg" if c.find_ancestor(exp.AggFunc) else "raw")
    for c in tree.find_all(exp.Column):
        if id(c) not in projected:
            note(c, "agg" if c.find_ancestor(exp.AggFunc) else "filter")
    return QueryUses(uses, aliases, tree)


def violations(q: QueryUses, spec: PurposeSpec, policy: PurposePolicy) -> dict[str, float]:
    """(column:mode) -> sensitivity weight, for uses beyond the purpose's allowance."""
    out = {}
    for col, mode in q.uses.items():
        if LEVEL[mode] > LEVEL[spec.columns.get(col, "none")]:
            out[f"{col}:{mode}"] = policy.weight(col)
    return out


def _key_sets(tree: exp.Expression, alias: str, key: str, executor: Executor, dialect: str,
              grouped: bool) -> list[frozenset]:
    """Entity keys each output group (or the whole result) is computed from."""
    if not isinstance(tree, exp.Select):
        return []
    q = tree.copy()
    group = q.args.get("group")
    group_exprs = [g.copy() for g in group.expressions] if (grouped and group) else []
    q.set("expressions", [exp.column(key, table=alias)] + group_exprs)
    for arg in ("group", "having", "order", "distinct"):
        q.set(arg, None)
    if grouped:
        q.set("limit", None)        # LIMIT applies to groups, not to contributing rows
    try:
        rows = executor(q.sql(dialect=dialect))
    except Exception:
        return []
    sets: dict[tuple, set] = defaultdict(set)
    for r in rows:
        sets[tuple(r[1:])].add(r[0])
    return [frozenset(s) for s in sets.values() if s]


class PurposeMonitor:
    """One session (subject + declared purpose). Call ``check`` before each query runs."""

    def __init__(self, policy: PurposePolicy, purpose: str, executor: Executor | None = None):
        if purpose not in policy.purposes:
            raise KeyError(f"unknown purpose {purpose!r}")
        self.policy, self.spec, self.executor = policy, policy.purposes[purpose], executor
        self.seen_violations: dict[str, float] = {}
        self.agg_sets: dict[str, list[frozenset]] = defaultdict(list)   # entity.protected col -> sets
        self.revealed: dict[str, set] = defaultdict(set)                 # entity table -> keys
        self.cell = 0.0

    def _protected_agg(self, q: QueryUses) -> list[str]:
        p = self.policy
        return [c for c, m in q.uses.items() if m == "agg" and p.weight(c) >= p.protected_threshold
                and c.split(".")[0] in p.entities]

    def _raw_sensitive(self, q: QueryUses) -> list[str]:
        p = self.policy
        return [c for c, m in q.uses.items() if m == "raw" and p.weight(c) >= p.protected_threshold
                and c.split(".")[0] in p.entities]

    def check(self, sql: str) -> PurposeSignals:
        p, s = self.policy, PurposeSignals()
        q = analyze(sql, p)
        v = violations(q, self.spec, p)
        s.violations = v
        s.drift_q = sum(v.values())
        self.seen_violations.update(v)
        if v:
            s.reasons.append("beyond purpose: " + ", ".join(sorted(v)))

        if self.executor is not None and q.tree is not None:
            by_table = {t: a for a, t in q.aliases.items()}
            for col in self._protected_agg(q):
                table = col.split(".")[0]
                alias, key = by_table.get(table), p.entities[table]
                if alias is None:
                    continue
                sets = _key_sets(q.tree, alias, key, self.executor, p.dialect, grouped=True)
                if not sets:
                    continue
                isolated = min(len(x) for x in sets)
                s.cell_q = max(s.cell_q, 1.0 / isolated)
                prior = self.agg_sets[col]
                for new in sets:
                    for old in prior:
                        small, big = (new, old) if len(new) <= len(old) else (old, new)
                        d = len(big) - len(small)
                        if d and small <= big:
                            isolated = min(isolated, d)
                if p.min_isolation and isolated < p.min_isolation:
                    s.refuse = True
                    s.reasons.append(f"refused: aggregate over {col} would isolate {isolated} entities "
                                     f"(minimum {p.min_isolation})")
                    continue                     # a refused query reveals nothing; keep state unchanged
                prior.extend(sets)
                del prior[:-p.max_tracked_sets]
                if 1.0 / isolated > self.cell:
                    self.cell = 1.0 / isolated
                    s.reasons.append(f"aggregate over {col} isolates {isolated} entities")

            for col in self._raw_sensitive(q):
                table = col.split(".")[0]
                alias = by_table.get(table)
                if alias is None:
                    continue
                keys = set().union(*_key_sets(q.tree, alias, p.entities[table], self.executor,
                                              p.dialect, grouped=False) or [set()])
                budget = self.spec.entity_budget.get(table, p.default_entity_budget)
                s.breadth_q = max(s.breadth_q, len(keys) / budget)
                self.revealed[table] |= keys
                if len(self.revealed[table]) > budget:
                    s.reasons.append(f"{table}: sensitive columns returned for "
                                     f"{len(self.revealed[table])} entities (budget {budget})")

        s.drift = sum(self.seen_violations.values())
        s.cell = self.cell
        s.breadth = max((len(k) / self.spec.entity_budget.get(t, p.default_entity_budget)
                         for t, k in self.revealed.items()), default=0.0)
        return s

"""Purpose-anchored session monitor for SQL agents (research prototype).

Each session declares a purpose. A purpose lists the columns it may touch and
how: ``filter`` (WHERE / JOIN / GROUP BY only, or inside COUNT), ``agg``
(inside SUM / AVG and similar) or ``raw`` (values returned to the caller,
including a projected MIN / MAX, which returns one row's value). The monitor parses every query
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

With ``linear_audit`` the monitor also treats every answered aggregate over a
protected column as a linear equation over per-entity values (SUM directly;
AVG together with the COUNT any caller can obtain) and refuses a query that
would make some entity's value exactly derivable from the answered set, by
any combination of queries (the classic audit-expert test of Chin and
Ozsoyoglu). Pairwise differencing remains the k-anonymity-style check for
"fewer than k" rather than "exactly one".

Limits of the prototype: CTEs, set operations and correlated subqueries are
not rewritten; HAVING is dropped from key-set rewrites (groups it would hide
still count); entities joined several times count once; refusal decisions
depend on which entities match a query, so a refusal can itself leak
membership (audits are not simulatable).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
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
    max_tracked_sets: int = 5000       # aggregate entity sets kept per protected column for pairwise checks
    min_isolation: int | None = None   # audit mode: refuse a query whose aggregate isolates fewer entities
    linear_audit: bool = False         # also refuse a query that makes one entity's value exactly derivable
    max_linear_sets: int = 2000        # groups per query the linear audit will handle; more are refused
    strict_protected: bool = False     # refuse predicates on protected columns: audit decisions then
                                       # depend only on non-protected data (simulatable)
    purposes: dict[str, PurposeSpec]

    def weight(self, col: str) -> float:
        return self.sensitivity.get(col, self.default_sensitivity)


@dataclass
class QueryUses:
    uses: dict[str, str]          # "table.column" -> highest mode it is used in
    aliases: dict[str, str]       # alias -> table
    tree: exp.Expression | None   # qualified tree, None if unparseable
    predicates: set = field(default_factory=set)   # columns used outside aggregates and the select list


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


def _mode(col: exp.Column, projected: bool) -> str:
    """COUNT reveals cardinality only (filter); MIN / MAX return an actual value (raw when projected)."""
    agg = col.find_ancestor(exp.AggFunc)
    if agg is None:
        return "raw" if projected else "filter"
    if isinstance(agg, exp.Count):
        return "filter"
    if projected and isinstance(agg, (exp.Min, exp.Max)):
        return "raw"
    return "agg"


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
    predicates: set[str] = set()

    def note(col: exp.Column, mode: str) -> None:
        table = aliases.get(col.table)
        if not table:
            return                                     # output alias or CTE column
        key = f"{table}.{col.name.lower()}"
        if mode == "filter" and col.find_ancestor(exp.AggFunc) is None:
            predicates.add(key)
        if LEVEL[mode] > LEVEL.get(uses.get(key, "none"), 0):
            uses[key] = mode

    for sel in top:
        for e in sel.expressions:
            for c in e.find_all(exp.Column):
                projected.add(id(c))
                note(c, _mode(c, projected=True))
    for c in tree.find_all(exp.Column):
        if id(c) not in projected:
            note(c, _mode(c, projected=False))
    return QueryUses(uses, aliases, tree, predicates)


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


class LinearAuditor:
    """Orthonormal basis of the answered query-set indicator vectors.

    Entity i's value is exactly derivable iff the unit vector e_i lies in the
    span, i.e. the i-th diagonal entry of the projection Q Q^T equals 1.
    """

    TOL = 1e-9

    def __init__(self):
        self.index: dict = {}
        self.Q = np.zeros((0, 0))

    def _grow(self, keys) -> None:
        new = [k for k in keys if k not in self.index]
        for k in new:
            self.index[k] = len(self.index)
        if new:
            self.Q = np.vstack([self.Q, np.zeros((len(new), self.Q.shape[1]))])

    def _add(self, Q: np.ndarray, keys) -> np.ndarray:
        v = np.zeros(Q.shape[0])
        v[[self.index[k] for k in keys]] = 1.0
        for _ in range(2):                       # re-orthogonalise for stability
            v = v - Q @ (Q.T @ v)
        n = np.linalg.norm(v)
        return np.hstack([Q, (v / n)[:, None]]) if n > 1e-7 else Q

    def disclosed(self, sets: list[frozenset]) -> list:
        """Entities whose value would become exactly derivable if ``sets`` were answered."""
        self._grow(set().union(*sets))
        Q = self.Q
        for s in sets:
            Q = self._add(Q, s)
        diag = (Q ** 2).sum(axis=1)
        keys = list(self.index)
        return [keys[i] for i in np.nonzero(diag > 1 - self.TOL)[0]]

    def commit(self, sets: list[frozenset]) -> None:
        self._grow(set().union(*sets))
        for s in sets:
            self.Q = self._add(self.Q, s)


class PurposeMonitor:
    """One session (subject + declared purpose). Call ``check`` before each query runs."""

    def __init__(self, policy: PurposePolicy, purpose: str, executor: Executor | None = None):
        if purpose not in policy.purposes:
            raise KeyError(f"unknown purpose {purpose!r}")
        self.policy, self.spec, self.executor = policy, policy.purposes[purpose], executor
        self.seen_violations: dict[str, float] = {}
        self.agg_sets: dict[str, list[frozenset]] = defaultdict(list)   # entity.protected col -> sets
        self.linear: dict[str, LinearAuditor] = defaultdict(LinearAuditor)
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

        if p.strict_protected:
            bad = sorted(c for c in q.predicates if c in self._protected_cols())
            if bad:
                s.refuse = True
                s.reasons.append("refused: predicate on protected column " + ", ".join(bad))
                return self._finish(s)

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
                exact = []
                if p.linear_audit and not (p.min_isolation and isolated < p.min_isolation):
                    if len(sets) > p.max_linear_sets:     # too many groups to audit exactly: refuse
                        isolated = 0
                        s.reasons.append(f"{len(sets)} groups exceed the linear audit limit")
                    else:
                        exact = self.linear[col].disclosed(sets)
                if exact:
                    isolated = 1
                if p.min_isolation and isolated < p.min_isolation:
                    s.refuse = True
                    why = (f"would make {len(exact)} entities' {col} exactly derivable" if exact
                           else "exceeds the audit limit" if isolated == 0
                           else f"would isolate {isolated} entities")
                    s.reasons.append(f"refused: aggregate over {col} {why} (minimum {p.min_isolation})")
                    continue                     # a refused query reveals nothing; keep state unchanged
                if p.linear_audit:
                    self.linear[col].commit(sets)
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

        return self._finish(s)

    def _protected_cols(self) -> set[str]:
        p = self.policy
        return {c for c, m in self.spec.columns.items() if m == "agg" and p.weight(c) >= p.protected_threshold
                and c.split(".")[0] in p.entities}

    def _finish(self, s: PurposeSignals) -> PurposeSignals:
        p = self.policy
        s.drift = sum(self.seen_violations.values())
        s.cell = self.cell
        s.breadth = max((len(k) / self.spec.entity_budget.get(t, p.default_entity_budget)
                         for t, k in self.revealed.items()), default=0.0)
        return s

"""Action guard for generated or user-typed SQL.

Allow-list design: a statement passes only if it parses as a single read
query over allowed tables. Everything else is refused, so a statement type
nobody thought of (TRUNCATE, MERGE, a vendor extension) fails closed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

from .config import SqlGuardConfig

READS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


@dataclass
class SqlVerdict:
    decision: str                                   # allow | warn | block
    reasons: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    sensitive_columns: list[str] = field(default_factory=list)
    row_limit: int | None = None


def _row_limit(tree: exp.Expression) -> int | None:
    """Literal row cap from LIMIT, FETCH FIRST n, or WHERE ROWNUM <= n."""
    lim = tree.args.get("limit")
    node = None
    if isinstance(lim, exp.Fetch):
        node = lim.args.get("count")
    elif isinstance(lim, exp.Limit):
        node = lim.args.get("expression")
    if isinstance(node, exp.Literal) and not node.is_string:
        return int(node.this)
    for cmp in tree.find_all(exp.LTE, exp.LT):
        left, right = cmp.this, cmp.expression
        if isinstance(left, exp.Column) and left.name.lower() == "rownum" and isinstance(right, exp.Literal):
            n = int(right.this)
            return n if isinstance(cmp, exp.LTE) else n - 1
    return None


def _identifier_tokens(sql: str, dialect: str) -> list[str]:
    """Raw identifier text, read before the parser normalizes anything
    (sqlglot rewrites e.g. DBMS_RANDOM.VALUE to RAND())."""
    toks = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
    out, prev = [], None
    for t in toks:
        if t.token_type in (TokenType.STRING, TokenType.NATIONAL_STRING):
            prev = None
            continue
        text = t.text.lower()
        if prev is not None and t.token_type == TokenType.DOT:
            out.append(prev + ".")
        out.append(text)
        prev = text
    return out


def check_sql(sql: str, cfg: SqlGuardConfig) -> SqlVerdict:
    try:
        trees = [t for t in sqlglot.parse(sql, read=cfg.dialect) if t is not None]
    except sqlglot.errors.SqlglotError as e:
        return SqlVerdict("block", [f"unparseable SQL: {str(e).splitlines()[0][:120]}"])
    if len(trees) != 1:
        return SqlVerdict("block", ["exactly one statement is allowed"])
    tree = trees[0]
    if not isinstance(tree, READS):
        return SqlVerdict("block", ["only read queries (SELECT / WITH ... SELECT) are allowed"])

    block, warn = [], []
    idents = _identifier_tokens(sql, cfg.dialect)
    hits = sorted({i.rstrip(".") for i in idents for p in cfg.blocked_identifier_prefixes if i.startswith(p)})
    if hits:
        block.append(f"blocked system object or package: {', '.join(hits)}")

    ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
    tables = sorted({t.name.lower() for t in tree.find_all(exp.Table)} - ctes)
    schemas = {t.db.lower() for t in tree.find_all(exp.Table) if t.db}
    if cfg.allowed_tables:
        outside = [t for t in tables if t not in {a.lower() for a in cfg.allowed_tables}]
        if outside:
            block.append(f"tables outside the policy: {', '.join(outside)}")
    if cfg.allowed_schemas and schemas - {s.lower() for s in cfg.allowed_schemas}:
        block.append(f"schemas outside the policy: {', '.join(sorted(schemas))}")

    sens_map = {t.lower(): {c.lower() for c in cols} for t, cols in cfg.sensitive_columns.items()}
    all_sensitive = set().union(*sens_map.values()) if sens_map else set()
    cols = {c.name.lower() for c in tree.find_all(exp.Column)}
    sensitive = cols & all_sensitive
    # a projected * or t.* -- not COUNT(*) or another function argument
    has_star = any(not isinstance(s.parent, exp.Func) for s in tree.find_all(exp.Star))
    if has_star:  # SELECT * / t.* exposes every sensitive column of the tables read
        for t in tables:
            sensitive |= sens_map.get(t, set())
    if sensitive:
        msg = f"sensitive columns: {', '.join(sorted(sensitive))}"
        (block if cfg.sensitive_action == "block" else warn).append(msg)

    limit = _row_limit(tree)
    if has_star and cfg.require_limit_for_star and limit is None:
        block.append("SELECT * without a row limit")
    if cfg.max_rows is not None and limit is not None and limit > cfg.max_rows:
        block.append(f"row limit {limit} exceeds the maximum of {cfg.max_rows}")

    decision = "block" if block else "warn" if warn else "allow"
    return SqlVerdict(decision, block + warn, tables, sorted(sensitive), limit)

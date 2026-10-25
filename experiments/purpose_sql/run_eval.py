"""Evaluate purpose-anchored session signals against per-query and length baselines.

For every session, each detector produces a score trajectory over its queries
(computed before each query runs). Thresholds are set per purpose on a held-out
half of the benign sessions at a target false-alarm rate; attack families are
scored against the other half of the benign sessions *of the same purpose*.

    python experiments/purpose_sql/run_eval.py
writes experiments/results/purpose_sql/results.json and prints a summary.
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "guardrail-service"))
sys.path.insert(0, str(HERE))

import tpch_db as db                                                     # noqa: E402
from guardrail.config import SqlGuardConfig                               # noqa: E402
from guardrail.purpose import PurposeMonitor, PurposePolicy                # noqa: E402
from guardrail.sql_guard import check_sql                                  # noqa: E402

SESSIONS = ROOT / "data" / "purpose_sql" / "sessions.jsonl"
OUT = ROOT / "experiments" / "results" / "purpose_sql" / "results.json"
ALPHAS = (0.01, 0.05)
FAMILY_PURPOSES = {"drift": ("sales_reporting", "supply_chain"), "small_cell": ("finance_analytics",),
                   "tracker": ("finance_analytics",), "enumeration": ("customer_support",),
                   "tracker3": ("finance_analytics",)}

SINGLE = ["queries", "sqlguard_any", "sqlguard_count", "row_ledger",
          "drift_q", "cell_q", "breadth_q", "drift", "cell", "breadth"]
COMBOS = {"stateless_or": ["drift_q", "cell_q", "breadth_q"], "stateful_or": ["drift", "cell", "breadth"],
          "existing_or": ["sqlguard_count", "row_ledger"]}


def load_policy() -> PurposePolicy:
    raw = yaml.safe_load((HERE / "purpose_policy.yaml").read_text())
    raw["schema"] = db.SCHEMA
    return PurposePolicy.model_validate(raw)


def score_session(s: dict, policy: PurposePolicy, con: sqlite3.Connection, guard: SqlGuardConfig) -> dict:
    run = lambda sql: con.execute(sql).fetchall()                       # noqa: E731
    mon = PurposeMonitor(policy, s["purpose"], executor=run)
    traj = defaultdict(list)
    sens_count = ledger = 0
    for i, q in enumerate(s["queries"]):
        sig = mon.check(q["sql"])                     # before the query runs
        v = check_sql(q["sql"], guard)
        hit = bool(v.sensitive_columns)
        sens_count += hit
        traj["queries"].append(i + 1)
        traj["sqlguard_any"].append(float(hit))
        traj["sqlguard_count"].append(sens_count)
        traj["row_ledger"].append(ledger)             # rows seen before this query
        for k in ("drift_q", "cell_q", "breadth_q", "drift", "cell", "breadth"):
            traj[k].append(getattr(sig, k))
        rows = run(q["sql"])
        if hit:
            ledger += len(rows)
    # the ledger can only react after rows come back: shift so query i sees rows up to i
    traj["row_ledger"] = traj["row_ledger"][1:] + [ledger]
    return dict(traj)


def auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def threshold(scores: list[float], alpha: float) -> float:
    """Smallest t with mean(score > t) <= alpha on the calibration set."""
    xs = sorted(scores)
    k = math.ceil((1 - alpha) * len(xs)) - 1
    return xs[max(0, min(k, len(xs) - 1))]


def first_alarm(traj: list[float], t: float) -> int | None:
    return next((i for i, x in enumerate(traj) if x > t), None)


def main() -> None:
    policy = load_policy()
    guard = SqlGuardConfig(dialect="sqlite", max_rows=None, require_limit_for_star=False,
                           sensitive_columns={"customer": ["c_phone", "c_address", "c_acctbal"],
                                              "supplier": ["s_phone", "s_address", "s_acctbal"]})
    con = sqlite3.connect(f"file:{db.DEFAULT_OUT}?mode=ro", uri=True)
    sessions = [json.loads(l) for l in SESSIONS.read_text().splitlines()]
    for s in sessions:
        s["traj"] = score_session(s, policy, con, guard)

    rng = random.Random(5)
    benign = defaultdict(list)
    for s in sessions:
        if s["label"] == 0:
            benign[s["purpose"]].append(s)
    calib, test = {}, {}
    for p, lst in benign.items():
        rng.shuffle(lst)
        calib[p], test[p] = lst[: len(lst) // 2], lst[len(lst) // 2:]
    attacks = defaultdict(list)
    for s in sessions:
        if s["label"] == 1:
            attacks[s["family"]].append(s)

    peak = lambda s, k: max(s["traj"][k])                                  # noqa: E731
    results = {"n_sessions": len(sessions), "auroc": {}, "operating": {}}

    for fam, purposes in FAMILY_PURPOSES.items():
        neg = [s for p in purposes for s in test[p]]
        results["auroc"][fam] = {k: round(auroc([peak(s, k) for s in attacks[fam]], [peak(s, k) for s in neg]), 3)
                                 for k in SINGLE}

    for alpha in ALPHAS:
        # per purpose, per signal thresholds; OR detectors split alpha across their members
        th = {p: {k: threshold([peak(s, k) for s in calib[p]], alpha) for k in SINGLE} for p in calib}
        th_split = {p: {k: threshold([peak(s, k) for s in calib[p]], alpha / 3) for k in SINGLE} for p in calib}

        def alarm(s, det):
            if det in COMBOS:
                idx = [first_alarm(s["traj"][k], th_split[s["purpose"]][k]) for k in COMBOS[det]]
                idx = [i for i in idx if i is not None]
                return min(idx) if idx else None
            return first_alarm(s["traj"][det], th[s["purpose"]][det])

        dets = SINGLE + list(COMBOS)
        op = {"benign_fpr": {}, "tpr": {}, "prevented": {}}
        for det in dets:
            op["benign_fpr"][det] = {p: round(sum(alarm(s, det) is not None for s in test[p]) / len(test[p]), 3)
                                     for p in test}
            op["tpr"][det], op["prevented"][det] = {}, {}
            for fam, lst in attacks.items():
                hits = [(alarm(s, det), s["harm_index"]) for s in lst]
                op["tpr"][det][fam] = round(sum(a is not None for a, _ in hits) / len(lst), 3)
                op["prevented"][det][fam] = round(sum(a is not None and a <= h for a, h in hits) / len(lst), 3)
        results["operating"][f"alpha={alpha}"] = op

    results["audit"] = audit_eval(sessions, policy, con)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print_summary(results)


def audit_eval(sessions: list[dict], policy: PurposePolicy, con: sqlite3.Connection) -> dict:
    """Query-level refusal instead of session alarms, for the aggregate families.

    stateless: refuse a query whose own smallest group has < k entities (cell suppression).
    pairwise:  also refuse a query that, with one earlier answered aggregate, isolates < k entities.
    linear:    pairwise, plus refuse a query that makes any entity's value exactly derivable
               from all answered aggregates together.
    strict:    linear, plus refuse predicates on protected columns (simulatable decisions).
    """
    run = lambda sql: con.execute(sql).fetchall()                       # noqa: E731
    fin = [s for s in sessions if s["purpose"] == "finance_analytics"]
    modes = ("stateless", "pairwise", "linear", "strict")
    out = {}
    for k in (2, 3, 5):
        pols = {"pairwise": policy.model_copy(update={"min_isolation": k}),
                "linear": policy.model_copy(update={"min_isolation": k, "linear_audit": True}),
                "strict": policy.model_copy(update={"min_isolation": k, "linear_audit": True,
                                                    "strict_protected": True})}
        stats = {m: {"benign_queries_refused": 0, "benign_sessions_hit": 0,
                     "prevented": defaultdict(int)} for m in modes}
        n_bq = n_bs = 0
        n_att = defaultdict(int)
        for s in fin:
            mons = {m: PurposeMonitor(pols[m], s["purpose"], executor=run) for m in pols}
            first = dict.fromkeys(modes)
            refused = dict.fromkeys(modes, 0)
            for i, q in enumerate(s["queries"]):
                sigs = {m: mon.check(q["sql"]) for m, mon in mons.items()}
                flags = {"stateless": sigs["pairwise"].cell_q > 1.0 / k,
                         "pairwise": sigs["pairwise"].refuse, "linear": sigs["linear"].refuse,
                         "strict": sigs["strict"].refuse}
                for m, r in flags.items():
                    if r:
                        refused[m] += 1
                        first[m] = i if first[m] is None else first[m]
            if s["label"] == 0:
                n_bq += len(s["queries"])
                n_bs += 1
                for m in modes:
                    stats[m]["benign_queries_refused"] += refused[m]
                    stats[m]["benign_sessions_hit"] += refused[m] > 0
            else:
                n_att[s["family"]] += 1
                for m in modes:
                    if first[m] is not None and first[m] <= s["harm_index"]:
                        stats[m]["prevented"][s["family"]] += 1
        out[f"k={k}"] = {m: {"benign_query_refusal_rate": round(v["benign_queries_refused"] / n_bq, 4),
                             "benign_session_hit_rate": round(v["benign_sessions_hit"] / n_bs, 3),
                             "attack_prevented": {f: round(v["prevented"][f] / n_att[f], 3) for f in n_att}}
                         for m, v in stats.items()}
    return out


def print_summary(r: dict) -> None:
    fams = list(FAMILY_PURPOSES)
    print("\nAudit mode (finance): refuse the query, not the session")
    print(f"{'k':4}{'mode':11}{'benign queries refused':>24}{'benign sessions hit':>21}"
          f"{'small_cell':>12}{'tracker':>9}{'tracker3':>10}")
    for k, modes in r["audit"].items():
        for m, v in modes.items():
            ap = v["attack_prevented"]
            print(f"{k:4}{m:11}{v['benign_query_refusal_rate']:>24}{v['benign_session_hit_rate']:>21}"
                  f"{ap.get('small_cell', 0):>12}{ap.get('tracker', 0):>9}{ap.get('tracker3', 0):>10}")
    print("\nAUROC (attack family vs held-out benign of the same purpose)")
    print(f"{'detector':16}" + "".join(f"{f:>13}" for f in fams))
    for det in SINGLE:
        print(f"{det:16}" + "".join(f"{r['auroc'][f][det]:>13}" for f in fams))
    for a in ALPHAS:
        op = r["operating"][f"alpha={a}"]
        print(f"\nTarget benign FPR {a}: TPR / prevented-before-harm, and test benign FPR per purpose")
        print(f"{'detector':16}" + "".join(f"{f:>15}" for f in fams) + "   benign FPR")
        for det in op["tpr"]:
            cells = "".join(f"{op['tpr'][det][f]:>8} / {op['prevented'][det][f]:<4}" for f in fams)
            fpr = " ".join(f"{v:.2f}" for v in op["benign_fpr"][det].values())
            print(f"{det:16}{cells}   {fpr}")


if __name__ == "__main__":
    main()

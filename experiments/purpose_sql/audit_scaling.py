"""Cost of the incremental linear audit as entities (n) and answered groups (rank r) grow.

Random range-style entity sets (contiguous key ranges of random length, as SQL
range predicates produce) are checked and committed one at a time; we time the
check (`disclosed`) at given ranks and report the basis memory n * r * 8 bytes.

    python experiments/purpose_sql/audit_scaling.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "guardrail-service"))
from guardrail.purpose import LinearAuditor                                  # noqa: E402

OUT = ROOT / "experiments" / "results" / "purpose_sql" / "audit_scaling.json"
NS, RANKS, REPS = (1_000, 10_000, 50_000), (10, 100, 500), 5


def main() -> None:
    rng = random.Random(0)
    res = []
    for n in NS:
        aud = LinearAuditor()
        aud.disclosed([frozenset(range(n))])          # register all entities once
        aud.commit([frozenset(range(n))])
        added, marks = 1, {}
        while added < max(RANKS):
            lo = rng.randrange(n)
            hi = min(n, lo + rng.randint(5, max(6, n // 10)))
            s = frozenset(range(lo, hi))
            if added + 1 in RANKS:
                ts = []
                for _ in range(REPS):
                    t0 = time.perf_counter()
                    aud.disclosed([s])
                    ts.append(1000 * (time.perf_counter() - t0))
                marks[added + 1] = statistics.median(ts)
            aud.commit([s])
            added = aud.Q.shape[1]
        for r in RANKS:
            res.append({"n": n, "rank": r, "check_ms": round(marks.get(r, float("nan")), 2),
                        "basis_mb": round(n * r * 8 / 1e6, 1)})
            print(res[-1])
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

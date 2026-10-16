"""Per-subject data-exposure budget.

This is the one kind of state worth keeping: a violation defined over the
whole session ("this user has pulled 5,000 customer phone numbers today"),
which no single query reveals. The subject must be an identity the calling
app controls (user id, API client), never a session id sent by the browser,
or an attacker can reset the budget by rotating ids.

In-memory and per-process; run one replica or swap in a shared store.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from .config import SessionConfig


@dataclass
class Exposure:
    sensitive_rows: int
    total_rows: int


class ExposureLedger:
    def __init__(self, cfg: SessionConfig, clock=time.time):
        self.cfg = cfg
        self.clock = clock
        self._events: dict[str, deque] = defaultdict(deque)   # subject -> (ts, sensitive, total)
        self._lock = threading.Lock()

    def _prune(self, q: deque, now: float) -> None:
        while q and q[0][0] < now - self.cfg.window_seconds:
            q.popleft()

    def record(self, subject: str, rows: int, sensitive: bool) -> Exposure:
        now = self.clock()
        with self._lock:
            q = self._events[subject]
            self._prune(q, now)
            q.append((now, rows if sensitive else 0, rows))
            return self._total(q)

    def current(self, subject: str) -> Exposure:
        with self._lock:
            q = self._events[subject]
            self._prune(q, self.clock())
            return self._total(q)

    @staticmethod
    def _total(q: deque) -> Exposure:
        return Exposure(sum(e[1] for e in q), sum(e[2] for e in q))

    def verdict(self, exp: Exposure) -> tuple[str, list[str]]:
        c = self.cfg
        if exp.sensitive_rows >= c.sensitive_rows_block:
            return "block", [f"sensitive-row budget exhausted ({exp.sensitive_rows}/{c.sensitive_rows_block} in window)"]
        if exp.total_rows >= c.total_rows_block:
            return "block", [f"row budget exhausted ({exp.total_rows}/{c.total_rows_block} in window)"]
        if exp.sensitive_rows >= c.sensitive_rows_warn:
            return "warn", [f"sensitive rows approaching budget ({exp.sensitive_rows}/{c.sensitive_rows_block})"]
        return "allow", []

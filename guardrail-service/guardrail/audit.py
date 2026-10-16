"""Append-only audit log of every decision. Stores hashes and lengths, never
raw text, so the log can be kept and shared without holding user content."""
from __future__ import annotations

import hashlib
import json
import threading
import time


def sha(text: str | None, n: int = 16) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


class AuditLog:
    def __init__(self, path: str | None):
        self.path = path
        self._lock = threading.Lock()

    def write(self, endpoint: str, subject: str, decision: str, reasons: list[str],
              text: str | None = None, **extra) -> None:
        if not self.path:
            return
        rec = {"ts": round(time.time(), 3), "endpoint": endpoint, "subject": sha(subject),
               "input_sha256": sha(text), "input_len": len(text) if text else 0,
               "decision": decision, "reasons": reasons, **extra}
        line = json.dumps(rec) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

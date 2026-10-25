"""Text guard: a safety classifier read as a probability, not a yes/no.

The Ollama backend asks Llama Guard 3 for one token and reads the first-token
log-probabilities: score = P(unsafe) / (P(unsafe) + P(safe)). A continuous
score lets the threshold be calibrated to a false-alarm budget.
"""
from __future__ import annotations

import math
from typing import Protocol

import httpx

from .config import TextGuardConfig


class Scorer(Protocol):
    def score(self, messages: list[dict]) -> float: ...


class NullScorer:
    """Scores everything 0.0. For tests, or when only the SQL guard is used."""

    def score(self, messages: list[dict]) -> float:
        return 0.0


class OllamaLlamaGuard:
    def __init__(self, cfg: TextGuardConfig):
        self.cfg = cfg
        self.client = httpx.Client(base_url=cfg.url, timeout=cfg.timeout_seconds)

    def score(self, messages: list[dict]) -> float:
        msgs = [{"role": m["role"], "content": (m.get("content") or "(empty)")[:4000]} for m in messages]
        r = self.client.post("/api/chat", json={
            "model": self.cfg.model, "messages": msgs, "stream": False,
            "logprobs": True, "top_logprobs": 10,
            "options": {"temperature": 0, "num_predict": 1, "num_ctx": 8192}})
        r.raise_for_status()
        out = r.json()
        top = (out.get("logprobs") or [{}])[0].get("top_logprobs") or []
        p_unsafe = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() == "unsafe")
        p_safe = sum(math.exp(t["logprob"]) for t in top if t["token"].strip() == "safe")
        if p_unsafe + p_safe <= 0:  # neither token in the top 10: fall back to the text
            return 1.0 if out["message"]["content"].strip().startswith("unsafe") else 0.0
        return p_unsafe / (p_unsafe + p_safe)


def make_scorer(cfg: TextGuardConfig) -> Scorer:
    if cfg.backend == "none":
        return NullScorer()
    if cfg.backend == "ollama":
        return OllamaLlamaGuard(cfg)
    raise ValueError(f"unknown text_guard.backend: {cfg.backend}")


def decide(score: float, threshold: float, warn_margin: float) -> str:
    if score >= threshold:
        return "block"
    if score >= threshold - warn_margin:
        return "warn"
    return "allow"

"""Minimal Ollama chat client with retries, and first-token probabilities."""
from __future__ import annotations

import json
import math
import time
import urllib.request

CHAT_URL = "http://localhost:11434/api/chat"


def chat(payload: dict, timeout: float = 600.0, retries: int = 3, backoff: float = 5.0) -> dict:
    """POST /api/chat; retries transient failures, re-raises the last one."""
    data = json.dumps({"stream": False, **payload}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(CHAT_URL, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)


def token_mass(top_logprobs: list[dict], word: str, case_insensitive: bool = False) -> float:
    """Probability mass of one verdict word among a position's top tokens."""
    norm = (lambda s: s.strip().lower()) if case_insensitive else (lambda s: s.strip())
    target = norm(word)
    return sum(math.exp(t["logprob"]) for t in top_logprobs if norm(t["token"]) == target)


def first_token_prob(out: dict, positive: str, negative: str, case_insensitive: bool = False) -> float:
    """P(positive) / (P(positive) + P(negative)) at the first generated token.
    If neither word is in the top tokens, fall back to the generated text."""
    top = out["logprobs"][0]["top_logprobs"]
    p = token_mass(top, positive, case_insensitive)
    n = token_mass(top, negative, case_insensitive)
    if p + n <= 0:
        text = out["message"]["content"].strip()
        return 1.0 if (text.lower() if case_insensitive else text).startswith(
            positive.lower() if case_insensitive else positive) else 0.0
    return p / (p + n)

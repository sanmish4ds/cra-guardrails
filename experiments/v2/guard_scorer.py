"""Continuous turn-level and prefix-level risk scores from a guard model.

Llama Guard 3 answers with ``safe`` or ``unsafe`` as its first generated
token. Parsing the generated text yields a binary label, which makes every
operating point degenerate (this is why earlier tables in this repository
report sFPR = 1.0 for the guard baselines). Instead, we request the first
token's log-probabilities from Ollama and return

    g = P(unsafe) / (P(unsafe) + P(safe)),

a continuous score that supports ROC analysis and thresholding. Granite
Guardian 3 answers ``Yes``/``No`` (risk present) and is read the same way.

Two modes are used by the v2 protocol:
  * stateless  - the guard sees only the current exchange
                 (user turn, or user turn + assistant reply);
  * prefix     - the guard sees the whole conversation up to the turn
                 (the model-native "stateful" baseline).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from ollama_client import chat, first_token_prob

MODEL = "llama-guard3:1b"
GRANITE = "granite3-guardian:2b"
MAX_CHARS = 4000         # per message; bounds prompt length on long transcripts
MAX_PROMPT_CHARS = 16000  # whole conversation; oldest messages dropped first
OPTIONS = {"temperature": 0, "num_predict": 1, "num_ctx": 8192}


def _normalise(messages: list[dict]) -> list[dict]:
    """Llama Guard's template requires strictly alternating user/assistant
    roles starting with the user; merge consecutive same-role messages."""
    out: list[dict] = []
    for m in messages:
        content = (m["content"] or "").strip()[:MAX_CHARS] or "(empty)"
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] += "\n" + content
        else:
            out.append({"role": m["role"], "content": content})
    # Keep the most recent messages within MAX_PROMPT_CHARS (whole exchanges).
    while len(out) > 2 and sum(len(m["content"]) for m in out) > MAX_PROMPT_CHARS:
        out = out[2:] if out[0]["role"] == "user" else out[1:]
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(conversation start)"})
    return out


def _granite_messages(messages: list[dict]) -> list[dict]:
    """Granite Guardian's template reads only the last (user) or last two
    (user, assistant) messages. For a context-aware view we prepend earlier
    exchanges, labelled, to the final user message."""
    msgs = _normalise(messages)
    if len(msgs) <= 2 and not (len(msgs) == 2 and msgs[-1]["role"] == "user"):
        return msgs
    if msgs[-1]["role"] == "user":
        hist, last = msgs[:-1], msgs[-1]
        lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in hist]
        return [{"role": "user", "content": "Conversation so far:\n" + "\n".join(lines)
                 + "\n\nLatest user message: " + last["content"]}]
    return msgs[-2:]


def score_one(messages: list[dict], model: str = MODEL) -> float:
    """P(violation) for one rendered conversation, in [0, 1]."""
    if model == GRANITE:
        out = chat({"model": GRANITE, "messages": _granite_messages(messages),
                    "logprobs": True, "top_logprobs": 10, "options": OPTIONS})
        return first_token_prob(out, "yes", "no", case_insensitive=True)
    out = chat({"model": model, "messages": _normalise(messages),
                "logprobs": True, "top_logprobs": 10, "options": OPTIONS})
    return first_token_prob(out, "unsafe", "safe")


def score_many(batch: list[list[dict]], workers: int = 4, model: str = MODEL) -> list[float]:
    """Score message lists with the given guard model (Llama Guard 3 or Granite Guardian 3)."""
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda m: score_one(m, model), batch))

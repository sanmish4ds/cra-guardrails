#!/usr/bin/env python3
"""Build LLM-paraphrased template pool for CRA-Bench v0.2.

For every fixed text template in generate_cra_bench_v01.py that can vary
without changing meaning, ask gpt-4o-mini for N=8 paraphrases (preserving any
{curly_brace} placeholders verbatim). Results are cached to JSON so the
benchmark generator stays fully deterministic given (seed, cache).

Cost: ~75 templates x 1 LLM call = ~75 calls @ gpt-4o-mini (~$0.01 total).
Run time: ~2-3 min over sequential HTTPS calls.

Usage:
    python experiments/build_paraphrases_v0_2.py
    python experiments/build_paraphrases_v0_2.py --n-variants 8 --force
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUT_DEFAULT = PROJECT_ROOT / "data" / "cra_bench_v01" / "paraphrases_v0_2.json"
_LAST_RESPONSE_MODEL: str | None = None


PARAPHRASE_PROMPT = """You are a paraphrase generator for a benchmark.
Produce exactly N={n} distinct paraphrases of the input text.

Rules:
- Preserve meaning, register, and approximate length.
- If the input contains literal placeholders inside curly braces (for example
  {{k}}, {{obj}}, {{ent}}, {{topic}}, {{target}}, {{benign_topic}},
  {{name}}, {{employer}}, {{city}}, {{neighborhood}}, {{role}},
  {{schedule}}, {{phone_area}}, {{family}}, {{commute}}, {{persona}},
  {{domain}}, {{harmful_ask}}, {{benign_ask}}, {{harmful_topic}},
  {{distractor_a}}, {{distractor_b}}, {{distractor_c}}, {{distractor_d}}),
  every placeholder must appear EXACTLY ONCE in every paraphrase, with the
  same name and the same curly braces. Do not add new placeholders; do not
  remove any.
- Do not change the role implied by the text (a user prompt stays a user
  prompt; an assistant reply stays an assistant reply).
- Vary syntactic structure and word choice. Do not just rearrange clauses.
- Do not add markdown, quotes, bullets, or numbering.
- Output STRICT JSON of the form: {{"paraphrases": ["...", "...", ...]}}

Input:
\"\"\"{text}\"\"\""""


# ─── Template inventory (must match field names in generate_cra_bench_v01.py)

def _load_v01():
    spec = importlib.util.spec_from_file_location(
        "g01", SCRIPT_DIR / "generate_cra_bench_v01.py"
    )
    g01 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g01)
    return g01


def collect_templates(g01) -> dict[str, list[str]]:
    """Return {field_name: [text, ...]} listing every text template to
    paraphrase. We deliberately skip:
      - FRAGMENT_OBJECTIVES / FRAGMENT_BENIGN_TOPIC entity pools
      - AGGREGATION_TARGETS / AGGREGATION_PUBLIC_TARGETS record dicts
    because those are stand-alone domain entities, not paraphrasable text.
    """
    return {
        "FRAGMENT_WRAPPERS_POS":              list(g01.FRAGMENT_WRAPPERS_POS),
        "FRAGMENT_WRAPPERS_BENIGN":           list(g01.FRAGMENT_WRAPPERS_BENIGN),
        "FRAGMENT_ASSISTANT_BENIGN_WARMUP":   list(g01.FRAGMENT_ASSISTANT_BENIGN_WARMUP),
        "FRAGMENT_ASSISTANT_POS_FRAGMENT":    list(g01.FRAGMENT_ASSISTANT_POS_FRAGMENT),
        "FRAGMENT_ASSISTANT_BENIGN_SECTION":  list(g01.FRAGMENT_ASSISTANT_BENIGN_SECTION),
        "CONDITIONING_USER_ESCALATION":       list(g01.CONDITIONING_USER_ESCALATION),
        "CONDITIONING_ASSISTANT_TRAJECTORY":  list(g01.CONDITIONING_ASSISTANT_TRAJECTORY),
        "CONDITIONING_BENIGN_USER":           list(g01.CONDITIONING_BENIGN_USER),
        "CONDITIONING_BENIGN_ASSISTANT":      list(g01.CONDITIONING_BENIGN_ASSISTANT),
        "AGGREGATION_USER_TURNS":             list(g01.AGGREGATION_USER_TURNS),
        "AGGREGATION_ASSISTANT_TURNS":        list(g01.AGGREGATION_ASSISTANT_TURNS),
        "AGGREGATION_BENIGN_USER":            list(g01.AGGREGATION_BENIGN_USER),
        "AGGREGATION_BENIGN_ASSISTANT":       list(g01.AGGREGATION_BENIGN_ASSISTANT),
        "PERSONA_USER_TURNS":                 list(g01.PERSONA_USER_TURNS),
        "PERSONA_ASSISTANT_TURNS":            list(g01.PERSONA_ASSISTANT_TURNS),
        "PERSONA_BENIGN_USER":                list(g01.PERSONA_BENIGN_USER),
        "PERSONA_BENIGN_ASSISTANT":           list(g01.PERSONA_BENIGN_ASSISTANT),
        "STUFFING_USER_TURNS":                list(g01.STUFFING_USER_TURNS),
        "STUFFING_ASSISTANT_TURNS":           list(g01.STUFFING_ASSISTANT_TURNS),
        "STUFFING_BENIGN_USER":               list(g01.STUFFING_BENIGN_USER),
        "STUFFING_BENIGN_ASSISTANT":          list(g01.STUFFING_BENIGN_ASSISTANT),
    }


PLACEHOLDER_RE = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")


def extract_placeholders(text: str) -> list[str]:
    return PLACEHOLDER_RE.findall(text)


def validate_paraphrase(original: str, paraphrase: str) -> tuple[bool, str]:
    o = sorted(extract_placeholders(original))
    p = sorted(extract_placeholders(paraphrase))
    if o != p:
        return False, f"placeholder mismatch: {o} vs {p}"
    if not paraphrase.strip():
        return False, "empty"
    if paraphrase.strip() == original.strip():
        return False, "identical to original"
    # length sanity: paraphrase shouldn't differ by more than 3x
    lo, lp = len(original), len(paraphrase)
    if lp < lo / 3 or lp > lo * 3:
        return False, f"length out of range ({lo} vs {lp})"
    return True, "ok"


def call_llm(text: str, n: int, model: str, api_url: str, api_key: str,
             timeout: int = 60) -> list[str]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise paraphrase generator that always emits strict JSON."},
            {"role": "user", "content": PARAPHRASE_PROMPT.format(n=n, text=text)},
        ],
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
    }
    resp = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    global _LAST_RESPONSE_MODEL
    _LAST_RESPONSE_MODEL = data.get("model", model)
    content = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Non-JSON LLM output: {content[:200]!r}") from e
    paraphrases = parsed.get("paraphrases") or parsed.get("variants") or []
    if not isinstance(paraphrases, list):
        raise RuntimeError(f"Bad paraphrases field: {paraphrases!r}")
    return [str(p).strip() for p in paraphrases if str(p).strip()]


def paraphrase_template(text: str, n: int, model: str, api_url: str,
                        api_key: str, max_retries: int = 3) -> list[str]:
    """Return up to n validated paraphrases. Retries up to max_retries times
    if too few valid variants come back."""
    accepted: list[str] = []
    seen: set[str] = {text.strip()}
    for attempt in range(max_retries):
        try:
            candidates = call_llm(text, n + 2, model, api_url, api_key)
        except Exception as e:
            print(f"      LLM call failed (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
            continue
        for cand in candidates:
            if cand.strip() in seen:
                continue
            ok, reason = validate_paraphrase(text, cand)
            if not ok:
                continue
            accepted.append(cand)
            seen.add(cand.strip())
            if len(accepted) >= n:
                return accepted
        if len(accepted) >= n:
            break
        time.sleep(1)
    return accepted


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--n-variants", type=int, default=8)
    ap.add_argument("--model", type=str,
                    default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    ap.add_argument("--api-url", type=str,
                    default=os.environ.get(
                        "LLM_API_URL",
                        "https://api.openai.com/v1/chat/completions"))
    ap.add_argument("--force", action="store_true",
                    help="Re-paraphrase even if cache exists.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        print("ERROR: no OPENAI_API_KEY / LLM_API_KEY in environment", file=sys.stderr)
        return 1

    g01 = _load_v01()
    inventory = collect_templates(g01)
    n_total = sum(len(v) for v in inventory.values())
    print(f"[Paraphrase] {n_total} templates across {len(inventory)} fields.")
    print(f"[Paraphrase] Model: {args.model}, N variants/template: {args.n_variants}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Incremental cache: re-use any (field, original_text) that already has
    # >=n_variants validated variants from a previous run. This keeps the
    # cost of adding new families bounded to the new templates only.
    prev_fields: dict[str, dict[str, list[str]]] = {}
    if args.out.exists() and not args.force:
        try:
            prev_cache = json.loads(args.out.read_text())
            for fname, entries in prev_cache.get("fields", {}).items():
                prev_fields[fname] = {
                    e["original"]: list(e.get("variants", []))
                    for e in entries
                }
            n_cached = sum(1 for f in inventory
                           for text in inventory[f]
                           if text in prev_fields.get(f, {})
                           and len(prev_fields[f][text]) >= args.n_variants)
            print(f"[Paraphrase] Cache hits: {n_cached}/{n_total} "
                  f"(remaining {n_total - n_cached} new templates).")
        except Exception as e:
            print(f"[Paraphrase] Could not parse existing cache ({e}); "
                  f"starting fresh.")
            prev_fields = {}

    result: dict[str, Any] = {
        "version": "v0.2",
        "source_module": "generate_cra_bench_v01.py",
        "model_requested": args.model,
        "model_resolved": None,
        "api_url": args.api_url,
        "temperature": 0.8,
        "n_variants": args.n_variants,
        "fields": {},
    }

    t0 = time.time()
    done = 0
    for field, texts in inventory.items():
        print(f"\n[Paraphrase] {field} ({len(texts)} templates)")
        field_result = []
        for i, text in enumerate(texts):
            sub_t0 = time.time()
            cached = prev_fields.get(field, {}).get(text, [])
            if len(cached) >= args.n_variants and not args.force:
                variants = cached[:args.n_variants]
                done += 1
                print(f"  [{done}/{n_total}] cached ({len(variants)} variants) "
                      f": {text[:60]!r}")
            else:
                variants = paraphrase_template(
                    text, args.n_variants, args.model, args.api_url, api_key,
                )
                done += 1
                elapsed = time.time() - sub_t0
                print(f"  [{done}/{n_total}] +{len(variants)} variants "
                      f"in {elapsed:.1f}s : {text[:60]!r}")
            field_result.append({
                "original": text,
                "variants": variants,
            })
        result["fields"][field] = field_result

    result["wall_seconds"] = round(time.time() - t0, 1)
    result["model_resolved"] = _LAST_RESPONSE_MODEL or args.model
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n[Paraphrase] Wrote {args.out}")
    print(f"[Paraphrase] Wall time: {result['wall_seconds']}s")
    print(f"[Paraphrase] Average variants/template: "
          f"{sum(len(t['variants']) for f in result['fields'].values() for t in f) / n_total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

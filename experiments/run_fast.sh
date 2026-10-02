#!/usr/bin/env bash
# Fast experiment path (~5–15 min multiseed after first signal cache build).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "=== 1/2 Multiseed (fast: 3 seeds, 15 epochs, cached signals) ==="
python3 experiments/run_cra_multiseed.py --fast --split-seed 42

echo "=== 2/2 Optional: v0.2 standard split only (skip LOFO + benign) ==="
# python3 experiments/run_cra_extended_protocol.py \
#   --bench data/cra_bench_v02_5fam/sessions.jsonl \
#   --out experiments/results/cra_extended_protocol_v02_5fam_fast.json \
#   --fast --lam-fam-std 0.1

echo "Done. Check experiments/results/cra_multiseed_v02_5fam.json"

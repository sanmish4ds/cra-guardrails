#!/usr/bin/env bash
# S3 gate + full protocol rerun for paper Tables (primary-main, v01-vs-v02, 5-fam, LOFO, human-transfer).
# Prerequisites: Ollama + llama-guard3:1b for training; GPU optional.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

LOG_DIR="${LOG_DIR:-$ROOT/experiments/logs}"
mkdir -p "$LOG_DIR"

echo "=== Phase 0: Ollama check ==="
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "ERROR: Ollama not running. Run: ollama serve && ollama pull llama-guard3:1b"
  exit 1
fi

echo "=== Phase 1: Train + calibrate S3 classifier (kappa >= 0.7 gate) ==="
set +e
python3 experiments/train_s3_refusal_classifier.py \
  --bench data/cra_bench_v02_5fam/sessions.jsonl \
  --max-train 1500 \
  --epochs 5 \
  2>&1 | tee "$LOG_DIR/s3_train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
if [[ "$TRAIN_RC" -ne 0 ]]; then
  echo "S3 training failed or kappa gate not met (exit $TRAIN_RC)."
  echo "  Inspect $LOG_DIR/s3_train.log and experiments/results/s3_classifier_audit.json"
  echo "  Primary reruns below are SKIPPED until gate passes."
  exit "$TRAIN_RC"
fi

S3_FLAG=(--s3-mode classifier)
echo "=== Phase 2: S3 comparison (keyword vs classifier, quick) ==="
python3 experiments/run_s3_comparison_protocol.py \
  2>&1 | tee "$LOG_DIR/s3_comparison.log"

echo "=== Phase 3: CRA-Bench v0.2 5-family extended protocol (classifier S3) ==="
python3 experiments/run_cra_extended_protocol.py \
  --bench data/cra_bench_v02_5fam/sessions.jsonl \
  --out experiments/results/cra_extended_protocol_v02_5fam_s3clf.json \
  "${S3_FLAG[@]}" \
  --lam-fam-std 0.1 \
  2>&1 | tee "$LOG_DIR/ext_v02_5fam_s3clf.log"

echo "=== Phase 4: CRA-Bench v0.3 aggregation fix (classifier S3) ==="
python3 experiments/run_cra_extended_protocol.py \
  --bench data/cra_bench_v03_5fam/sessions.jsonl \
  --out experiments/results/cra_extended_protocol_v03_5fam_s3clf.json \
  "${S3_FLAG[@]}" \
  --lam-fam-std 0.1 \
  2>&1 | tee "$LOG_DIR/ext_v03_5fam_s3clf.log"

echo "=== Phase 5: v0.1 three-family protocol (classifier S3) ==="
python3 experiments/run_cra_extended_protocol.py \
  --bench data/cra_bench_v01/sessions.jsonl \
  --out experiments/results/cra_extended_protocol_v01_s3clf.json \
  "${S3_FLAG[@]}" \
  2>&1 | tee "$LOG_DIR/ext_v01_s3clf.log"

echo "=== Phase 6: Multiseed variance (5 seeds, classifier S3 signals) ==="
python3 experiments/run_cra_multiseed.py \
  --split-seed 42 \
  --bench data/cra_bench_v02_5fam/sessions.jsonl \
  --s3-mode classifier \
  --out experiments/results/cra_multiseed_v02_5fam_s3clf.json \
  2>&1 | tee "$LOG_DIR/multiseed_s3clf.log"

echo "=== Phase 7: Human-CRA-Transfer (classifier S3) ==="
if [[ -f experiments/run_human_transfer_eval.py ]]; then
  python3 experiments/run_human_transfer_eval.py \
    2>&1 | tee "$LOG_DIR/human_transfer_s3clf.log" || true
fi

echo "=== Phase 8: v0.3 aggregation spot-check ==="
python3 experiments/run_v03_aggregation_column.py \
  --s3-mode classifier \
  2>&1 | tee "$LOG_DIR/v03_agg_s3clf.log" || \
python3 experiments/run_v03_aggregation_column.py \
  2>&1 | tee "$LOG_DIR/v03_agg.log"

echo "Done. Update CRA_FRAMEWORK_latest/Gaurdrails_Paper.tex from:"
echo "  experiments/results/s3_classifier_audit.json"
echo "  experiments/results/cra_extended_protocol_v02_5fam_s3clf.json"
echo "  experiments/results/cra_extended_protocol_v03_5fam_s3clf.json"

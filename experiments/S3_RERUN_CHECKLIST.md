# S₃ classifier + protocol rerun checklist (C1.1 / C1.2)

## Prerequisites

```bash
ollama serve
ollama pull llama-guard3:1b
cd /path/to/SQLclMCP
python3 -m pip install -r experiments/cra-requirements.txt
python3 -m spacy download en_core_web_sm
```

## Phase 1 — Train & κ gate (blocks everything else)

```bash
python3 experiments/train_s3_refusal_classifier.py \
  --bench data/cra_bench_v02_5fam/sessions.jsonl \
  --max-train 1500 \
  --epochs 5
```

**Pass criterion:** `experiments/results/s3_classifier_audit.json` has  
`"meets_kappa_07_classifier": true` (κ ≥ 0.7 on 200-window audit vs LG3).

**Calibrate only** (after weights exist):

```bash
python3 experiments/train_s3_refusal_classifier.py --skip-train
```

Outputs:
- `experiments/models/s3_refusal_distilbert/calibration.json` (threshold)
- `experiments/results/s3_classifier_audit.json`

## Phase 2 — One-command rerun (after gate passes)

```bash
bash experiments/run_s3_rerun_protocols.sh
```

Logs: `experiments/logs/s3_train.log`, `ext_v02_5fam_s3clf.log`, etc.

## Phase 3 — Manual step list (same as script)

| Step | Command | Paper tables |
|------|---------|--------------|
| 1 | `run_s3_comparison_protocol.py` | S₃ comparison |
| 2 | `run_cra_extended_protocol.py --bench v02_5fam --s3-mode classifier` | 10, 13, 16 (5-fam) |
| 3 | `run_cra_extended_protocol.py --bench v03_5fam --s3-mode classifier` | v0.3 aggregation |
| 4 | `run_cra_extended_protocol.py --bench v01 --s3-mode classifier` | 10, 14 (v0.1) |
| 5 | `run_cra_multiseed.py --s3-mode classifier` | 17 multiseed |
| 6 | `run_human_transfer_eval.py` (if wired) | 17 human transfer |
| 7 | `run_v03_aggregation_column.py --s3-mode classifier` | v0.3 column |

## Phase 4 — Update manuscript

Edit `CRA_FRAMEWORK_latest/Gaurdrails_Paper.tex` from new JSON paths (`*_s3clf.json`).

**Do not** claim classifier S₃ in abstract if κ gate still fails.

## If κ stays below 0.7

1. Increase `--max-train` (e.g. 2500) and `--epochs` (8–10).
2. Confirm Ollama LG3 labels are stable (`unsafe`/`safe` in response).
3. Keep keyword S₃ in primary tables; report classifier in appendix only (current paper stance).

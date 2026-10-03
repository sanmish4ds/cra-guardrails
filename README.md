# Conversational Risk Accumulation (CRA)

Research code and benchmark artifacts for evaluating multi-turn risk
accumulation in conversational AI systems.

> **Research artifact, not a production safety product.** The included CRA-Bench
> sessions are generated examples. They do not establish real-world or
> organizational effectiveness. The existing CoSafe/ShareGPT transfer
> comparison is source-confounded and must not be presented as independent
> validation.

## Repository contents

| Path | Contents |
|---|---|
| `experiments/` | Evaluation, baseline, ablation, and benchmark-generation scripts |
| `experiments/generate_cra_bench_v01.py` | Deterministic generator for CRA-Bench session templates |
| `experiments/results/` | Saved experiment outputs; see the paper/protocol before comparing runs |
| `data/cra_bench_*` | Generated CRA-Bench v0.1–v0.3 sessions and selected paraphrase caches |
| `data/human_cra_transfer/` | Local-only CoSafe/ShareGPT-derived transfer corpus; excluded from Git distribution pending rights review |
| `paper/figures/` | Figures used in the manuscript |
| `integration/cra-telemetry.js` | Reference telemetry integration; not a production-ready enforcement layer |

## What the benchmark does and does not show

CRA-Bench is a controlled, generated benchmark intended to exercise five
multi-turn patterns: fragmentation, conditioning, aggregation, persona, and
stuffing. Sessions are built from authored templates and, for some versions,
paraphrase variants. They are useful for pipeline checks and controlled
comparisons, but are not independently collected organizational conversations.

The existing transfer corpus combines 750 CoSafe positive sessions with 222
ShareGPT benign sessions. Because each source contains only one label, dataset
source perfectly identifies the class. The pooled transfer metrics therefore
cannot distinguish risk detection from source recognition. The transfer
corpus is not included in the repository distribution; obtain and use source
datasets only under their respective access and license terms.

Before interpreting any saved result, read the associated protocol and run the
source-confounding audit:

```bash
node experiments/audit_transfer_confounding.mjs
node --test experiments/tests/audit_transfer_confounding.test.mjs
```

## Quick start

The metadata audit and its tests require only Node.js 18 or newer. To run the
Python experiments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/cra-requirements.txt
python -m spacy download en_core_web_sm
```

The `.env` file is optional for non-LLM experiments. Copy `.env.example` only
if you need an API-backed judge or paraphrase-generation script. Keep credentials
in the ignored `.env` or `.env.local`; never commit them.

Run the repository's quick multiseed path from the root:

```bash
./experiments/run_fast.sh
```

This trains/evaluates models and may download model weights on first run; it
is not a tiny smoke test. For the longer S3 protocol, install and start Ollama,
pull `llama-guard3:1b`, and follow
[the S3 rerun checklist](experiments/S3_RERUN_CHECKLIST.md). Some optional
baselines use external datasets or services and have separate prerequisites.

### Synthetic fallback warning

Some data loaders can fall back to generated demo sessions when an external
dataset is unavailable. A successful script exit does not by itself prove that
the named external dataset was used. Inspect the run log and result metadata;
record the actual dataset ID, version, and whether fallback data was generated
before reporting results. Do not mix fallback runs with external-corpus
evaluations.

## Reproducibility and release notes

- The checked-in benchmark files are deterministic/generated artifacts, but
  some variants use paraphrases. Record the generator, seed, benchmark version,
  split, and model/software versions for each result.
- The standard randomized train/validation/test split is session-level.
  Generated variants can share templates, so these results do not establish
  generalization to unseen organizations, users, or attack strategies.
- Results are historical outputs; reproduce them with the matching script and
  protocol rather than assuming every JSON file came from the same run.
- Do not use the existing transfer result as evidence of deployment readiness.
  A real organizational evaluation requires approvals, de-identification,
  independently reviewed labels, and user/time/cohort-separated evaluation.

See [data provenance and licensing](DATA_PROVENANCE.md), the
[code license](LICENSE), and the
[benchmark-data license](data/LICENSE.md) before reuse.

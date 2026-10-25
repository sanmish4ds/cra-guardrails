# Data provenance and release status

This note distinguishes generated CRA benchmark artifacts from records derived
from external datasets. It is not a substitute for checking the source dataset's
current terms before downloading or using that dataset.

| Artifact | Provenance known from this repository | Distribution status |
|---|---|---|
| `data/cra_bench_v01/sessions.jsonl`, `data/cra_bench_v01_5fam/sessions.jsonl`, `data/cra_bench_v02/sessions.jsonl`, `data/cra_bench_v02_5fam/sessions.jsonl`, `data/cra_bench_v03_5fam/sessions.jsonl` | Generated session examples associated with `experiments/generate_cra_bench_v01.py`; some versions use paraphrase caches | Released under `data/LICENSE.md` only to the extent the licensors own or control the contributed material |
| `data/cra_bench_v01/paraphrases_v0_2.json`, `data/cra_bench_v03_5fam/paraphrases_v0_2.json` | Paraphrase caches used by generated benchmark variants; the producing model/provider and applicable terms are not recorded in this repository | Provenance/terms need confirmation before public redistribution |
| `data/human_cra_transfer/sessions.jsonl` | CoSafe positives combined with ShareGPT negatives; labels align exactly with source | Excluded from Git distribution pending upstream license/redistribution review; a pre-existing local copy may remain on a developer machine |
| `data/external/*` (HH-RLHF red-team, Bot-Adversarial Dialogue, ProsocialDialog, WildChat-1M shard, SafeMTData, locally generated MT-ActorAttack transcripts and judge verdicts, derived score files) | Downloaded third-party corpora and per-session derived scores used by the v2 study; sources, licenses, revisions, and hashes in `experiments/v2/PROTOCOL.md` | Git-ignored; never redistributed. Only aggregate metrics in `experiments/results/v2/` are released |
| `data/external/derived/actorattack*.jsonl` | Transcripts of SafeMTData ActorAttack chains played against `llama3.2` and `deepseek-r1:1.5b`, with Gemma 4 judge verdicts; replies may contain harmful content | Git-ignored; never redistributed (also subject to the target models' licenses). Only aggregate metrics are released |
| `experiments/results/*` | Experiment outputs; some results may depend on externally sourced or locally generated inputs | Verify the matching run log and inputs before redistribution or interpreting as a reproduction |

The transfer set's source-label confounding is measurable without distributing
or reading its conversation text:

```bash
node experiments/audit_transfer_confounding.mjs
```

When the restricted transfer corpus is not present, the audit uses the
aggregate source counts in the checked-in transfer result and identifies that
basis in its output. The audit is diagnostic only. Its in-sample source-only
score is not an out-of-sample estimate.

## Before making a public release

1. Confirm all authors, employers, funders, and other rights holders approve
   release under the selected licenses.
2. Confirm provenance and provider terms for both paraphrase caches.
3. Verify citations and current terms for CoSafe, ShareGPT, WildChat, and any
   other external dataset/model used by an experiment.
4. Record exact dependency, model, and dataset revisions for the reported runs.
5. Ensure no secrets, private logs, or unreviewed third-party conversation data
   are in the release commit or release archive.

If any contributor does not have authority to license a component, do not
release that component under this repository's license.

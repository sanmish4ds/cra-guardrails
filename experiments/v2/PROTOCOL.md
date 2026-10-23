# v2 evaluation protocol (confound-controlled stateful-guardrail study)

This protocol produces every result in `experiments/results/v2/`. Run from
the repository root, one stage at a time or all at once; every stage is
resumable:

```bash
bash experiments/v2/pipeline.sh corpora signals guard eval   # primary study
bash experiments/v2/pipeline.sh actorattack                  # MT-ActorAttack corpus (two judges)
bash experiments/v2/pipeline.sh granite                      # Granite Guardian replication
bash experiments/v2/pipeline.sh goal                         # goal inference, sequential monitor (~12 h on an M4)
bash experiments/v2/pipeline.sh diagnostics                  # onset, shift-length, turn location, Ollama check
```

`bash experiments/v2/pipeline.sh` with no argument lists the stages. Logs go
to `experiments/logs/v2_<stage>.log`.

Goal inference: `qwen3:4b-instruct` (digest `0edcdef34593`, Q4_K_M; the
reasoning variant `qwen3:4b` was tried first and rejected because it emitted
its reasoning in the answer). Second judge: `mistral:7b` (digest
`6577803aa9a0`). Pre-specified before results: goal detectors, Seq-Monitor,
Turn-LR-NOR. Added after inspecting MT-ActorAttack: CRA-Acc-P and the
turn-location diagnostic.

## External data (downloaded to `data/external/`, git-ignored, never redistributed)

| Corpus | Source / revision | License | SHA-256 of downloaded file(s) |
|---|---|---|---|
| HH-RLHF red-team attempts | `Anthropic/hh-rlhf` (HF), `red-team-attempts/red_team_attempts.jsonl.gz`, fetched 2026-10-03 | MIT | `4c7b0069991460f0064f279fd400b51f3f0095697d14d7793c49b0925f80814f` |
| Bot-Adversarial Dialogue v0.2 | `http://parl.ai/downloads/bot_adversarial_dialogue/dialogue_datasets_v0.2.tar.gz` | ParlAI release terms - **verify before any redistribution** | `2178b022fac154ddd9b570f6386abc4cd3e7ceb4476f0bebfbce5941424461eb` |
| ProsocialDialog | `allenai/prosocial-dialog` (HF) `train/valid/test.json` | CC-BY-4.0 | see note below table |
| WildChat-1M | `allenai/WildChat-1M` shard `train-00000-of-00014.parquet` (post-2024-10-17 content update: toxic and flagged-PII conversations removed) | ODC-BY | `abec2a13129db8c0e6a2d3a51ff12644873c748205a6fdf6551fbcb34430e51c` |
| SafeMTData Attack_600 (ActorAttack chains) | `SafeMTData` release of Ren et al. (ACL 2025), `Attack_600.json` | MIT | `58222f797df19e4e008575f8f74ad5865be914bc07952288b02c1e934b564159` |
| CRA-Bench v0.2 5-family | `data/cra_bench_v02_5fam/sessions.jsonl` (generated, this repo) | `data/LICENSE.md` | in repo |

Sampling (seed 0, `build_corpora.py`): HH 3,000 positives (rating >= 3) and
3,000 negatives (rating 0) among sessions with >= 2 exchanges; BAD 1,000
escalation positives (first human utterance OK, later utterance NOT OK) and all
402 all-OK dialogues; ProsocialDialog 1,000 casual-start escalation positives
and 1,000 all-casual negatives length-matched to them; WildChat 1,500 English
sessions with >= 2 exchanges. Sessions are truncated to 8 exchanges and
messages to 4,000 characters.

MT-ActorAttack (`build_actorattack.py`, `label_actorattack.py`,
`build_actorattack_corpus.py`): a seed-0 sample of 400 Attack_600 chains
played against `llama3.2:latest` (410 transcripts; 10 generated before the
selection was fixed are kept) and the first 200 against `deepseek-r1:1.5b`;
temperature 0, seed 0, <= 300 new tokens per reply, thinking off. Labels:
`gemma4:e4b` (digest `c6eb396dbd59`, Q4_K_M) with a HarmBench-style rubric,
temperature 0; 0 unparseable verdicts; no second judge was run. Groups =
attack chain.

## Models and software

- Guard: `llama-guard3:1b` (Ollama 0.34.2, digest `494147e06bf9`, Q8_0),
  temperature 0, `num_ctx` 8192, first-token top-10 log-probabilities;
  score = P(unsafe) / (P(unsafe) + P(safe)).
- Second guard: `granite3-guardian:2b` (digest `ba81a177bd23`, Q8_0), default
  harm risk; score = P(Yes) / (P(Yes) + P(No)) from first-token top-10
  log-probabilities. Context view: earlier exchanges prepended as text to the
  final user message (the template reads only the last message or pair). HH
  uses a stratified seed-0 subsample of 3,000 sessions.
- Larger readers (`pipeline.sh strong`): `llama-guard3:8b` (digest `46f211c3d866`,
  Q4_K_M) in all three views, and `llama3.1:8b` (digest `46e0c10c039e`, Q4_K_M)
  as a whole-transcript safety judge (`run_llm_judge.py`; P(Yes)/(P(Yes)+P(No))
  at the first token), matched by the same judge on single exchanges. Ollama
  0.40.8. Seed-0 stratified subsamples: ProsocialDialog 1,200, HH 1,500, BAD 902
  (`--max-sessions 1000`: 500 positives, all 402 negatives); all MT-ActorAttack.
  Results: `strong_guard_judge.json`, `llamaguard8b/`.
- Ollama was auto-updated from 0.34.2 to 0.35.1 during the extension runs;
  `check_ollama_version.py` re-scores stored sessions for both guards
  (max |diff| 5.6e-4; `experiments/results/v2/ollama_version_check.json`).
- Encoder: `sentence-transformers/all-MiniLM-L6-v2` (sentence-transformers 5.5.1);
  NER: spaCy `en_core_web_sm` 3.8.0.
- Python 3.14.3, numpy 2.4.3, scikit-learn 1.8.0, scipy 1.17.1, torch 2.12.0
  (CPU). Hardware: Apple M4, 16 GB.

## Statistical design

- Unit: session. Folds: `StratifiedGroupKFold(5, shuffle=True)`, seeds 0-2;
  groups = red-teamer (HH), template slot (CRA-Bench), attack chain
  (MT-ActorAttack), dialogue (others).
- AUROC point estimate: mean over seeds of pooled out-of-fold AUROC.
- 95% CI: percentile cluster bootstrap over groups, 2,000 resamples, seed-0
  out-of-fold scores; paired differences use the same resamples.
- Operating point: inside each training fold, 20% of the groups are held out
  for calibration; threshold = smallest value whose false-alarm rate on the
  held-out negatives is <= 5%; TPR/FPR measured on the test fold.
- Significance: Holm correction over the primary family only (four state
  contrasts per Llama Guard corpus, computed from the `p_le_0` fields of the results); all other
  contrasts are reported as paired-bootstrap intervals. The Granite Guardian
  runs are a replication and are not in the family.
- Caveat for MT-ActorAttack: labels come from one LLM judge without a human
  audit or second judge.

## Caveats that must accompany results

- HH labels are the red-teamer's self-rated success, not independent
  annotation; ratings 1-2 are excluded.
- The prior CoSafe and CoSafe/ShareGPT results are reported only as a
  confound audit; they are not validation (see `DATA_PROVENANCE.md`).
- CRA-Bench is generated; it tests template generalization only.

ProsocialDialog SHA-256: train `4782f339a1f01a6962289f1c7b2c0727e340e76af95a402c569a48f5df38dc75`, valid `2d937082c55dc9e5ce9416caba55dce0933df7ca768bfb76d65981dbfd3ecd9a`, test `3b6705cef4279e60bcd70b4fed3bf9a2e53283af3eda2723933d31feee220ffd`.

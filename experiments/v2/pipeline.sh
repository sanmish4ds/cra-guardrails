#!/bin/bash
# Reproduce the v2 study, stage by stage. Every stage is resumable: re-run it
# after an interruption and it continues where it stopped.
#
#   bash experiments/v2/pipeline.sh <stage> [<stage> ...]
#
# Stages, in dependency order:
#   corpora      derived corpora from the downloads in data/external (PROTOCOL.md)
#   signals      CRA reference signals, verified against run_cranet.py
#   guard        Llama Guard 3 features for every corpus
#   eval         primary evaluation, HH target-model shift, transfer and benign false alarms
#   actorattack  generate MT-ActorAttack, judge it twice, build the corpus, score and evaluate it
#   granite      Granite Guardian 3 replication
#   goal         goal-inference and sequential-monitor evidence, and its evaluation
#   strong       Llama Guard 3 8B and a full-transcript 8B judge (Llama 3.1)
#   diagnostics  onset visibility, shift-length check, turn location, Ollama version check
#   all          every stage above, in order
#
# Needs a local Ollama server with the models named in PROTOCOL.md. Logs go to
# experiments/logs/v2_<stage>.log.
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/../.."
V2=experiments/v2
mkdir -p experiments/logs

py() { python3 "$V2/$1" "${@:2}"; }

# Ollama's server log grows by gigabytes over long runs; empty it between steps.
clear_ollama_log() { [ -f ~/.ollama/logs/server.log ] && : > ~/.ollama/logs/server.log || true; }

unload() {  # free a model's memory before loading the next one
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"$1\",\"keep_alive\":0}" > /dev/null || true
}

stage_corpora()     { py build_corpora.py; }
stage_signals()     { py run_cra_signals.py; }
stage_guard()       { clear_ollama_log; py run_guard_features.py; }
stage_eval()        { py run_v2_eval.py --shift --transfer; }

stage_actorattack() {
  clear_ollama_log; py build_actorattack.py
  unload llama3.2:latest; unload deepseek-r1:1.5b
  clear_ollama_log; py label_actorattack.py --model gemma4:e4b; unload gemma4:e4b
  clear_ollama_log; py label_actorattack.py --model mistral:7b; unload mistral:7b
  py build_actorattack_corpus.py
  py run_cra_signals.py --corpora actorattack --verify 10
  clear_ollama_log; py run_guard_features.py --corpora actorattack
  py run_v2_eval.py --corpora actorattack --aa-shift
}

stage_granite() {
  local g="--model granite3-guardian:2b"
  clear_ollama_log; py run_guard_features.py $g --corpora actorattack bad prosocial
  clear_ollama_log; py run_guard_features.py $g --corpora hh_rt --max-sessions 3000
  unload granite3-guardian:2b
  py run_v2_eval.py --guard granite --corpora hh_rt bad prosocial actorattack
}

stage_goal() {
  for c in actorattack prosocial bad; do clear_ollama_log; py run_goal_features.py --corpora "$c"; done
  clear_ollama_log; py run_goal_features.py --corpora hh_rt --max-sessions 3000
  clear_ollama_log; py run_goal_features.py --corpora wildchat
  py run_v2_eval.py --goal --corpora hh_rt bad prosocial actorattack --transfer
}

stage_strong() {
  # Per corpus: Llama Guard 3 8B features, then the Llama 3.1 8B judge (seed-0 stratified subsamples).
  local spec
  for spec in actorattack:0 prosocial:1200 hh_rt:1500 bad:1000; do
    local c=${spec%%:*} n=${spec##*:} sub=""
    [ "$n" != 0 ] && sub="--max-sessions $n"
    clear_ollama_log; py run_guard_features.py --model llama-guard3:8b --corpora "$c" $sub
    unload llama-guard3:8b
    clear_ollama_log; py run_llm_judge.py --corpora "$c" $sub
    unload llama3.1:8b
  done
  py run_v2_eval.py --guard llamaguard8b --corpora actorattack prosocial hh_rt bad
  py eval_strong_guard_judge.py
}

stage_diagnostics() {
  py run_onset_visibility.py
  py run_shift_length_check.py
  py run_turn_location.py
  py check_ollama_version.py
}

[ $# -gt 0 ] || { sed -n '2,20p' "$SELF"; exit 1; }
stages=("$@")
[ "${stages[0]}" = all ] && stages=(corpora signals guard eval actorattack granite goal strong diagnostics)
for s in "${stages[@]}"; do
  declare -F "stage_$s" > /dev/null || { echo "unknown stage: $s" >&2; exit 1; }
  echo "[pipeline] $s"
  "stage_$s" >> "experiments/logs/v2_$s.log" 2>&1
  echo "[pipeline] $s done"
done

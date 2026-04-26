#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_BIN="/home/user/anaconda3/bin/conda"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

CHO_PATTERN="python -m autoeegencoder.train dataset=cho2017 experiment=cho2017_pra_mi_full trainer=gpu_external_full"
LEE_PATTERN="python -m autoeegencoder.train dataset=lee2019_mi experiment=lee2019_pra_mi_full trainer=gpu_external_full"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

latest_run_dir() {
  local pattern="$1"
  find "$PROJECT_ROOT/results" -maxdepth 1 -type d -name "$pattern" | sort | tail -n1
}

run_complete() {
  local run_dir="$1"
  [[ -n "$run_dir" ]] || return 1
  [[ -f "$run_dir/subject_metrics.csv" ]] || return 1
  [[ -f "$run_dir/diagnostics.csv" ]] || return 1
  [[ -f "$run_dir/harmonization_ledger.csv" ]] || return 1
  [[ -f "$run_dir/residual_audit.csv" ]] || return 1
  [[ -f "$run_dir/audit_summary.md" ]] || return 1
}

wait_for_pattern() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null; do
    log "Waiting on: $pattern"
    sleep 120
  done
}

run_step() {
  log "Running: $*"
  (cd "$PROJECT_ROOT" && CUDA_VISIBLE_DEVICES=0 "$CONDA_BIN" run -n AEC "$@")
}

log "Q1 overnight queue starting"

if pgrep -f "$CHO_PATTERN" >/dev/null; then
  log "Cho2017 full is already running; waiting for completion"
  wait_for_pattern "$CHO_PATTERN"
fi

CHO_RUN="$(latest_run_dir 'cho2017_pra_mi_full_cho2017_seed-2026_*')"
if ! run_complete "$CHO_RUN"; then
  log "Cho2017 full is not complete; launching a fresh run"
  if ! run_step python -m autoeegencoder.train dataset=cho2017 experiment=cho2017_pra_mi_full trainer=gpu_external_full; then
    log "Cho2017 full rerun failed; continuing with whatever results are already available"
  fi
  CHO_RUN="$(latest_run_dir 'cho2017_pra_mi_full_cho2017_seed-2026_*')"
fi

LEE_RUN="$(latest_run_dir 'lee2019_pra_mi_full_lee2019_mi_seed-2026_*')"
if ! run_complete "$LEE_RUN"; then
  if pgrep -f "$LEE_PATTERN" >/dev/null; then
    log "Lee2019 full is already running; waiting for completion"
    wait_for_pattern "$LEE_PATTERN"
  else
    log "Launching Lee2019 full"
    if ! run_step python -m autoeegencoder.train dataset=lee2019_mi experiment=lee2019_pra_mi_full trainer=gpu_external_full; then
      log "Lee2019 full failed; continuing to paper-table generation with available runs"
    fi
  fi
  LEE_RUN="$(latest_run_dir 'lee2019_pra_mi_full_lee2019_mi_seed-2026_*')"
fi

SUMMARY_ARGS=(python scripts/summarize_paper_main_tables.py --project-root . --output-dir work/paper_tables)
if run_complete "$CHO_RUN"; then
  SUMMARY_ARGS+=(--cho-full "$CHO_RUN")
fi
if run_complete "$LEE_RUN"; then
  SUMMARY_ARGS+=(--lee-full "$LEE_RUN")
fi

log "Generating paper tables"
if ! (cd "$PROJECT_ROOT" && "$CONDA_BIN" run -n AEC "${SUMMARY_ARGS[@]}"); then
  log "Paper-table generation failed"
  exit 1
fi

log "Q1 overnight queue finished"

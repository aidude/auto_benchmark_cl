#!/usr/bin/env bash
# Phase 1: compare LLMs across CL tasks.
# Usage: conda activate twopac && bash scripts/run_horse_race.sh
set -euo pipefail

source .env 2>/dev/null || { echo "ERROR: .env not found — copy .env.example first"; exit 1; }

TASKS=("sales_prediction")
MODELS=("gpt-4o-mini" "deepseek-chat" "deepseek-r1")
DATE=$(date +%Y-%m-%d)
RUN_ID=$(date +%H%M%S)
OUTDIR="results/${DATE}/horse_race_${RUN_ID}"
mkdir -p "$OUTDIR"

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== ${task} / ${model} ==="
    python -m src.experiments.run \
      --task   "$task" \
      --model  "$model" \
      --system stateless \
      --output "${OUTDIR}/${task}_${model}_stateless.json"
  done
done

echo "Done. Results in ${OUTDIR}"
echo "Logs  in  logs/${DATE}/"

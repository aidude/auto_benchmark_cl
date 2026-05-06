#!/usr/bin/env bash
# Phase 1: compare LLMs across CL tasks.
# Usage: conda activate twopac && bash scripts/run_horse_race.sh
set -euo pipefail

source .env 2>/dev/null || { echo "ERROR: .env not found — copy .env.example first"; exit 1; }

TASKS=("codebase_adaptation" "sales_prediction")
MODELS=("gpt-4o-mini" "deepseek-chat" "deepseek-r1")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/horse_race_${TIMESTAMP}"
mkdir -p "$OUTDIR"

for task in "${TASKS[@]}"; do
  for model in "${MODELS[@]}"; do
    echo "=== ${task} / ${model} ==="
    python -m src.experiments.run \
      --task  "$task" \
      --model "$model" \
      --output "${OUTDIR}/${task}_${model}.json"
  done
done

echo "Done. Results in ${OUTDIR}"

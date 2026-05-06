#!/usr/bin/env bash
# Phase 1: compare LLMs across CL tasks.
# Usage: conda activate twopac && bash scripts/run_horse_race.sh
set -euo pipefail

source .env 2>/dev/null || { echo "ERROR: .env not found — copy .env.example first"; exit 1; }

TASKS=("sales_prediction")

# Read horse_race model list from configs/models.yaml — single source of truth
mapfile -t MODELS < <(python -c "
import yaml
models = yaml.safe_load(open('configs/models.yaml')).get('horse_race', [])
print('\n'.join(models))
")

DATE=$(date +%Y-%m-%d)
RUN_ID=$(date +%H%M%S)
OUTDIR="results/${DATE}/horse_race_${RUN_ID}"
mkdir -p "$OUTDIR"

echo "Models: ${MODELS[*]}"
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

echo "Done.  Results → ${OUTDIR}"
echo "Logs  → logs/${DATE}/"

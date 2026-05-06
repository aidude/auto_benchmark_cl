#!/usr/bin/env bash
# Phase 2: run SAGE system against CL tasks.
# Usage: conda activate twopac && bash scripts/run_sage.sh
# NOTE: SAGESystem implementation must be complete before running.
set -euo pipefail

source .env 2>/dev/null || { echo "ERROR: .env not found — copy .env.example first"; exit 1; }

TASKS=("codebase_adaptation" "sales_prediction")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="results/sage_${TIMESTAMP}"
mkdir -p "$OUTDIR"

for task in "${TASKS[@]}"; do
  echo "=== SAGE / ${task} ==="
  python -m src.experiments.run \
    --task   "$task" \
    --system sage \
    --output "${OUTDIR}/${task}_sage.json"
done

echo "Done. Results in ${OUTDIR}"

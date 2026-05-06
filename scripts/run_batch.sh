#!/usr/bin/env bash
# Run one or all experiments defined in configs/experiments.yaml.
#
# Usage:
#   bash scripts/run_batch.sh                        # run ALL experiments
#   bash scripts/run_batch.sh horse_race_v1          # run one experiment
#   bash scripts/run_batch.sh horse_race_v1 --dry-run
#   bash scripts/run_batch.sh --list
#
# Cron (nightly at 2 am):
#   0 2 * * * cd /Users/champ/Documents/auto_benchmark_cl && \
#             conda run -n twopac bash scripts/run_batch.sh nightly \
#             >> logs/cron.log 2>&1
set -euo pipefail

source .env 2>/dev/null || { echo "ERROR: .env not found — copy .env.example first"; exit 1; }

EXPERIMENT="${1:-}"
shift || true           # remaining args forwarded to batch_run.py
EXTRA_ARGS="$*"

if [ "$EXPERIMENT" = "--list" ]; then
  python -m src.experiments.batch_run --list
elif [ -z "$EXPERIMENT" ]; then
  python -m src.experiments.batch_run --all $EXTRA_ARGS
else
  python -m src.experiments.batch_run --experiment "$EXPERIMENT" $EXTRA_ARGS
fi

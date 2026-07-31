#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${GMSTRAT_CONFIG:-experiments/configs/fixed_clump_k2.example.json}"
JOB_COUNT="$(python3 -m experiments.sample_fixed_clump --config "$CONFIG" --job-count)"
if [[ "$JOB_COUNT" != "28" ]]; then
  echo "Expected 28 sampling jobs for the checked-in Slurm array; found $JOB_COUNT" >&2
  exit 1
fi

mkdir -p local/experiments/fixed_clump_k2/logs
mkdir -p local/experiments/fixed_clump_k2/results

SAMPLE_JOB_ID="$(sbatch --parsable "$SCRIPT_DIR/bouchet_sample_array.sbatch")"
ANALYSIS_JOB_ID="$(
  sbatch \
    --parsable \
    --dependency="afterok:$SAMPLE_JOB_ID" \
    "$SCRIPT_DIR/bouchet_analyze.sbatch"
)"

echo "sampling_array_job=$SAMPLE_JOB_ID"
echo "dependent_analysis_job=$ANALYSIS_JOB_ID"
echo "monitor with: squeue -j $SAMPLE_JOB_ID,$ANALYSIS_JOB_ID"

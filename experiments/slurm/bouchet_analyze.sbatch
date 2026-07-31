#!/usr/bin/env bash
#SBATCH --job-name=gm-fixed-analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=local/experiments/fixed_clump_k2/logs/analysis-%j.out
#SBATCH --error=local/experiments/fixed_clump_k2/logs/analysis-%j.err

set -euo pipefail

: "${SCRATCH:?Bouchet must define SCRATCH}"
: "${SLURM_SUBMIT_DIR:?Submit this job from the gmstrat repository root}"

cd "$SLURM_SUBMIT_DIR"
module purge
module load miniconda/24.11.3
eval "$(conda shell.bash hook)"

GMSTRAT_PY_ENV="${GMSTRAT_PY_ENV:-$SCRATCH/gmstrat/envs/fixed-clump}"
GMSTRAT_CONFIG="${GMSTRAT_CONFIG:-experiments/configs/fixed_clump_k2.example.json}"
GMSTRAT_OUTPUT="${GMSTRAT_OUTPUT:-local/experiments/fixed_clump_k2/results}"
conda activate "$GMSTRAT_PY_ENV"

export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRATCH/gmstrat/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/gmstrat/cache}"

echo "job_id=$SLURM_JOB_ID host=$(hostname)"
python -m experiments.run_fixed_clump \
  --config "$GMSTRAT_CONFIG" \
  --output "$GMSTRAT_OUTPUT"

#!/usr/bin/env bash
#SBATCH --job-name=gm-fixed-clump
#SBATCH --array=0-27%4
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1-00:00:00
#SBATCH --output=local/experiments/fixed_clump_k2/logs/sample-%A_%a.out
#SBATCH --error=local/experiments/fixed_clump_k2/logs/sample-%A_%a.err

set -euo pipefail

: "${SCRATCH:?Bouchet must define SCRATCH}"
: "${SLURM_SUBMIT_DIR:?Submit this job from the gmstrat repository root}"

cd "$SLURM_SUBMIT_DIR"
module purge
module load miniconda/24.11.3
module load Julia/1.11.4-linux-x86_64
eval "$(conda shell.bash hook)"

GMSTRAT_PY_ENV="${GMSTRAT_PY_ENV:-$SCRATCH/gmstrat/envs/fixed-clump}"
GMSTRAT_CONFIG="${GMSTRAT_CONFIG:-experiments/configs/fixed_clump_k2.example.json}"
conda activate "$GMSTRAT_PY_ENV"

export JULIA_DEPOT_PATH="${JULIA_DEPOT_PATH:-$SCRATCH/gmstrat/julia_depot}"
export GMSTRAT_SKIP_INSTANTIATE=true
export JULIA_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/gmstrat/cache}"

echo "job_id=$SLURM_JOB_ID array_task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
python -m experiments.sample_fixed_clump \
  --config "$GMSTRAT_CONFIG" \
  --job-index "$SLURM_ARRAY_TASK_ID" \
  --execute
